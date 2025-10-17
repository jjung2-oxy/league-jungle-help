
#!/usr/bin/env python3
# Minimap champions watcher (no audio): detects champ circle icons and labels them
# Pipeline: (A) red ring Hough candidates -> (B) best template match over champ circle icons

import time
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mss
import numpy as np
import yaml
from pynput import keyboard

CONFIG_PATH = Path("config.yaml")

DEFAULTS = {
    "roi": {"left": None, "top": None, "width": 360, "height": 360},
    "target_fps": 18,
    "monitor_index": 1,
    "detector": {
        "ring_pad_frac": 0.35,
        "tmpl_scales": [0.9, 1.0, 1.1],
        "champ_thresh": 0.85,     # tighten/loosen as needed
        "cooldown_sec": 3.0       # visual flash cooldown on new detection
    },
    "template_dir": "assets/templates/champ_minimap"
}

# Smooth presence + UI hold of last detection
SMOOTH = {
    "frames_confirm": 2,     # frames in a row needed to "be present"
    "frames_release": 3,     # frames in a row needed to "clear"
    "box_hold_sec": 1.2
}

GLOBAL_FB = {
    "enabled": True,
    "min_interval_sec": 2.0,
    "run_every_n_frames": 6,
    "downscale": 0.7,
    "accept_thresh": 0.90  # higher for broad search
}

# ----------------- config -----------------
def load_cfg():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}
    def merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict):
                a[k] = merge(a.get(k, {}) or {}, v)
            else:
                a.setdefault(k, v)
        return a
    cfg = merge(cfg, DEFAULTS)
    Path(cfg["template_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg

# ----------------- capture -----------------
def clamp_roi_to_monitor(roi, mon):
    left = max(0, int(roi["left"])); top = max(0, int(roi["top"]))
    width = max(1, int(roi["width"])); height = max(1, int(roi["height"]))
    if left + width > mon["width"]:
        width = mon["width"] - left
    if top + height > mon["height"]:
        height = mon["height"] - top
    roi.update({"left": left, "top": top, "width": width, "height": height})
    return roi

def grab_roi_frame(sct, mon, roi):
    rect = {"left": mon["left"]+roi["left"], "top": mon["top"]+roi["top"],
            "width": roi["width"], "height": roi["height"]}
    raw = sct.grab(rect)
    arr = np.array(raw, dtype=np.uint8)
    return np.ascontiguousarray(arr[:, :, :3])

# ----------------- templates -----------------
class ChampTemplate:
    def __init__(self, alias: str, gray: np.ndarray, mask: Optional[np.ndarray]):
        self.alias = alias      # lowercase alias (e.g., 'ahri', 'monkeyking')
        self.gray = gray
        self.mask = mask

DISPLAY_OVERRIDES = {
    "monkeyking": "Wukong",
    "ksante": "K'Sante",
    "kogmaw": "Kog'Maw",
    "xinzhao": "Xin Zhao",
    "jarvaniv": "Jarvan IV",
    "leesin": "Lee Sin",
    "masteryi": "Master Yi",
    "missfortune": "Miss Fortune",
    "drmundo": "Dr. Mundo",
    "tahmkench": "Tahm Kench",
    "reksai": "Rek'Sai",
    "velkoz": "Vel'Koz"
}

def humanize_alias(alias: str) -> str:
    if alias in DISPLAY_OVERRIDES:
        return DISPLAY_OVERRIDES[alias]
    s = re.sub(r'[_\-]+', ' ', alias)
    s = re.sub(r'\biv\b', 'IV', s, flags=re.I)
    return s.title()

def _pick_lowest_circle(files: List[Path]) -> Optional[Path]:
    circle0 = [p for p in files if re.search(r'_circle_0\.png$', p.name, re.I)]
    if circle0:
        return circle0[0]
    numeric = []
    for p in files:
        m = re.search(r'_circle_(\d+)\.png$', p.name, re.I)
        if m:
            numeric.append((int(m.group(1)), p))
    if numeric:
        numeric.sort(key=lambda t: t[0])
        return numeric[0][1]
    plain = [p for p in files if re.search(r'_circle\.png$', p.name, re.I)]
    return plain[0] if plain else None

def load_champ_templates(dirpath: Path) -> List[ChampTemplate]:
    groups: Dict[str, List[Path]] = {}
    for p in sorted(dirpath.glob("*.png")):
        m = re.match(r'^(.+?)_circle(?:_\d+)?\.png$', p.name, flags=re.I)
        if not m:
            continue
        alias = m.group(1).lower()
        groups.setdefault(alias, []).append(p)

    tmpls: List[ChampTemplate] = []
    for alias, paths in groups.items():
        pick = _pick_lowest_circle(paths)
        if not pick:
            continue
        img = cv2.imread(str(pick), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[warn] failed to read template {pick}")
            continue
        bgr = img[:, :, :3] if img.ndim == 3 else img
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        a_mask = None
        if img.ndim == 3 and img.shape[2] == 4:
            a_mask = cv2.threshold(img[:, :, 3], 1, 255, cv2.THRESH_BINARY)[1]
        tmpls.append(ChampTemplate(alias, gray, a_mask))
    print(f"[info] loaded {len(tmpls)} champion template(s) from {dirpath}")
    return tmpls

# ----------------- Stage A: ring candidates -----------------
def red_dominance_mask(bgr: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(bgr)
    rg = cv2.max(b, g)
    dom = (r.astype(np.int16) - rg.astype(np.int16))
    m = (dom > 35) & (r > 110)
    mask = (m.astype(np.uint8) * 255)
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask

def find_ring_candidates(bgr: np.ndarray, pad_frac: float,
                         radius_px: Optional[Tuple[int,int]] = None):
    mask = red_dominance_mask(bgr)
    blurred = cv2.GaussianBlur(mask, (7, 7), 1.5)
    H, W = bgr.shape[:2]
    if radius_px is None:
        minR = max(8, int(round(min(H, W) * 0.032)))
        maxR = max(12, int(round(min(H, W) * 0.060)))
    else:
        minR, maxR = radius_px

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
        param1=120, param2=12, minRadius=minR, maxRadius=maxR
    )

    boxes = []
    if circles is not None:
        circles = np.uint16(np.around(circles[0, :]))
        for (cx, cy, r) in circles:
            x = int(max(0, cx - r))
            y = int(max(0, cy - r))
            w = int(min(W - x, 2 * r))
            h = int(min(H - y, 2 * r))
            pad = int(pad_frac * max(w, h))
            x = max(0, x - pad); y = max(0, y - pad)
            w = min(W - x, w + 2 * pad); h = min(H - y, h + 2 * pad)
            boxes.append((x, y, w, h))

    return mask, boxes

# ----------------- Stage B: best champ template match -----------------
def best_champ_in_patch(patch_bgr: np.ndarray, templates: List[ChampTemplate],
                        scales: Tuple[float, ...], thresh: float):
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    best = None  # (score, alias, x1, y1, x2, y2)
    for t in templates:
        timg, tmask = t.gray, t.mask
        for s in scales:
            th = int(max(1, round(timg.shape[0] * s)))
            tw = int(max(1, round(timg.shape[1] * s)))
            if th < 8 or tw < 8 or th > gray.shape[0] or tw > gray.shape[1]:
                continue
            rtmpl = cv2.resize(timg, (tw, th), interpolation=cv2.INTER_AREA)
            rmask = cv2.resize(tmask, (tw, th), interpolation=cv2.INTER_NEAREST) if tmask is not None else None
            if rmask is not None:
                res = cv2.matchTemplate(gray, rtmpl, cv2.TM_CCORR_NORMED, mask=rmask)
            else:
                res = cv2.matchTemplate(gray, rtmpl, cv2.TM_CCOEFF_NORMED)
            _, maxVal, _, maxLoc = cv2.minMaxLoc(res)
            if maxVal < thresh:
                continue
            x, y = maxLoc
            cand = (float(maxVal), t.alias, x, y, x + tw, y + th)
            if best is None or cand[0] > best[0]:
                best = cand
    return best

# ----------------- UI helpers -----------------
def ensure_window_alive(window_name, mouse_cb=None):
    try:
        visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
    except Exception:
        visible = -1
    if visible < 1:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 360, 360)
        cv2.moveWindow(window_name, 64, 64)
        try:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass
        if mouse_cb:
            cv2.setMouseCallback(window_name, mouse_cb)
        print(f"[ui] window {window_name!r} recreated")

# ----------------- main -----------------
def main():
    cfg = load_cfg()
    roi = cfg["roi"]
    if None in (roi["left"], roi["top"]):
        raise SystemExit("[error] ROI not set in config.yaml")

    det = cfg["detector"]
    tmpl_dir = Path(cfg["template_dir"])
    champ_templates = load_champ_templates(tmpl_dir)
    if not champ_templates:
        raise SystemExit(f"[error] No champion templates found in {tmpl_dir}")

    # flags + hotkeys
    show = True
    debug_on = True
    show_mask = True
    hot = {"toggle": False, "debug": False, "mask": False, "quit": False}
    mods = {"ctrl": False}
    from pynput.keyboard import Key, KeyCode

    def on_press(key):
        try:
            k = key.char.lower() if hasattr(key, "char") and key.char else ""
        except Exception:
            k = ""
        if k == "v": hot["toggle"] = True
        elif k == "d": hot["debug"] = True
        elif k == "m": hot["mask"] = True
        # Require Ctrl+Q to quit
        if key in (Key.ctrl_l, Key.ctrl_r):
            mods["ctrl"] = True
        elif isinstance(key, KeyCode) and key.char and key.char.lower() == "q" and mods["ctrl"]:
            hot["quit"] = True
    def on_release(key):
        if key in (Key.ctrl_l, Key.ctrl_r):
            mods["ctrl"] = False
    listener = keyboard.Listener(on_press=on_press, on_release=on_release, suppress=False)
    listener.daemon = True
    listener.start()

    target_period = 1.0 / max(1, int(cfg["target_fps"]))
    window = "Champ Watcher  (v toggle, d debug, m mask, q/esc quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 360, 360)
    cv2.moveWindow(window, 64, 64)
    try:
        cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    with mss.mss() as sct:
        mon = sct.monitors[cfg["monitor_index"]]
        roi = clamp_roi_to_monitor(roi, mon)
        frame_idx = 0
        last_global_try_t = 0.0
        suspend_global_until = 0.0

        present_run = 0
        absent_run = 0
        present_latched = False
        box_latched_until = 0.0
        flash_until = 0.0

        last_box = None            # (x1, y1, x2, y2)
        last_label = None
        last_score = 0.0

        while True:
            t0 = time.time()
            try:
                bgr = grab_roi_frame(sct, mon, roi)

                mask, cands = find_ring_candidates(
                    bgr,
                    pad_frac=det.get("ring_pad_frac", 0.35),
                    radius_px=None
                )
                vis = bgr.copy()
                if debug_on:
                    for (x, y, w, h) in cands:
                        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 0), 1)

                best = None  # (score, alias, (x1,y1,x2,y2))
                tmpl_scales = tuple(det.get("tmpl_scales", [0.9, 1.0, 1.1]))
                champ_thresh = float(det.get("champ_thresh", 0.85))

                for (x, y, w, h) in cands:
                    patch = bgr[y:y+h, x:x+w]
                    m = best_champ_in_patch(patch, champ_templates, tmpl_scales, champ_thresh)
                    if m is None:
                        continue
                    sc, alias, px1, py1, px2, py2 = m
                    gx1, gy1, gx2, gy2 = x + px1, y + py1, x + px2, y + py2
                    if (best is None) or (sc > best[0]):
                        best = (sc, alias, (gx1, gy1, gx2, gy2))

                now = time.time()
                if best is None:
                    should_try_global = (
                        GLOBAL_FB["enabled"]
                        and (now >= suspend_global_until)
                        and (now - last_global_try_t) >= GLOBAL_FB["min_interval_sec"]
                        and (frame_idx % GLOBAL_FB["run_every_n_frames"] == 0)
                    )
                    if should_try_global:
                        last_global_try_t = now
                        downscale = GLOBAL_FB["downscale"]
                        if downscale != 1.0:
                            small = cv2.resize(bgr, (0,0), fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
                            scale_back = 1.0 / downscale
                        else:
                            small = bgr
                            scale_back = 1.0
                        gbest = None
                        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                        for t in champ_templates:
                            timg, tmask = t.gray, t.mask
                            for s in tmpl_scales:
                                th = int(max(1, round(timg.shape[0] * s * downscale)))
                                tw = int(max(1, round(timg.shape[1] * s * downscale)))
                                if th < 8 or tw < 8 or th > gray_small.shape[0] or tw > gray_small.shape[1]:
                                    continue
                                rtmpl = cv2.resize(timg, (tw, th), interpolation=cv2.INTER_AREA)
                                rmask = cv2.resize(tmask, (tw, th), interpolation=cv2.INTER_NEAREST) if tmask is not None else None
                                if rmask is not None:
                                    res = cv2.matchTemplate(gray_small, rtmpl, cv2.TM_CCORR_NORMED, mask=rmask)
                                else:
                                    res = cv2.matchTemplate(gray_small, rtmpl, cv2.TM_CCOEFF_NORMED)
                                _, mval, _, mloc = cv2.minMaxLoc(res)
                                if mval >= GLOBAL_FB["accept_thresh"] and (gbest is None or mval > gbest[0]):
                                    xg, yg = mloc
                                    gbest = (
                                        float(mval), t.alias,
                                        int(round(xg*scale_back)), int(round(yg*scale_back)),
                                        int(round((xg+tw)*scale_back)), int(round((yg+th)*scale_back))
                                    )
                        if gbest is not None:
                            best = (gbest[0], gbest[1], (gbest[2], gbest[3], gbest[4], gbest[5]))

                present_raw = False
                if best is not None:
                    sc, alias, (x1, y1, x2, y2) = best
                    present_raw = True
                    last_box = (x1, y1, x2, y2)
                    last_label = humanize_alias(alias)
                    last_score = sc
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{last_label} ({sc:.2f})"
                    y_txt = max(12, y1 - 6)
                    cv2.putText(vis, label, (x1, y_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                if present_raw:
                    present_run = min(present_run + 1, 10**6)
                    absent_run = 0
                else:
                    absent_run = min(absent_run + 1, 10**6)
                    present_run = 0

                if not present_latched and present_run >= SMOOTH["frames_confirm"]:
                    present_latched = True
                    flash_until = now + 0.8
                    box_latched_until = now + SMOOTH["box_hold_sec"]

                if present_latched and absent_run >= SMOOTH["frames_release"]:
                    present_latched = False

                if present_raw:
                    box_latched_until = max(box_latched_until, now + SMOOTH["box_hold_sec"])

                draw_box = present_raw or (now < box_latched_until)
                if draw_box and not present_raw and last_box is not None:
                    x1, y1, x2, y2 = last_box
                    color = (0, 255, 0)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    label = f"{last_label} ({last_score:.2f})" if last_label else f"{last_score:.2f}"
                    y_txt = max(12, y1 - 6)
                    cv2.putText(vis, label, (x1, y_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if time.time() < flash_until and last_label:
                    H, W = vis.shape[:2]
                    thickness = 6
                    cv2.rectangle(vis, (0, 0), (W-1, H-1), (0, 200, 255), thickness)
                    cv2.putText(
                        vis, f"DETECTED: {last_label}",
                        (8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2
                    )

                if show_mask:
                    mask_small = cv2.resize(mask, (120, 120), interpolation=cv2.INTER_NEAREST)
                    mask_small = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
                    H, W = vis.shape[:2]
                    x0 = W - 120 - 6; y0 = 24
                    vis[y0:y0+120, x0:x0+120] = mask_small
                    cv2.rectangle(vis, (x0, y0), (x0+120, y0+120), (0,255,255), 1)
                    cv2.putText(vis, "mask", (x0+4, y0+14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

                cv2.putText(vis, datetime.now().strftime("%H:%M:%S"), (8, vis.shape[0]-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)

                if show:
                    cv2.imshow(window, vis)
                    ensure_window_alive(window)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    hot["quit"] = True
                elif key == ord('v'):
                    hot["toggle"] = True
                elif key == ord('d'):
                    hot["debug"] = True
                elif key == ord('m'):
                    hot["mask"]  = True

                if hot["quit"]:
                    break
                if hot["toggle"]:
                    show = not show; hot["toggle"] = False
                if hot["debug"]:
                    debug_on = not debug_on; hot["debug"] = False
                if hot["mask"]:
                    show_mask = not show_mask; hot["mask"] = False

            except KeyboardInterrupt:
                print("[info] interrupted; exiting.")
                break
            except Exception as e:
                print(f"[error] frame failed: {e}")
                time.sleep(0.01)
                continue

            elapsed = time.time() - t0
            if elapsed < target_period:
                time.sleep(target_period - elapsed)

            frame_idx += 1

    cv2.destroyAllWindows()
    try:
        listener.stop()
    except Exception as e:
        print(f"[error] listener failed: {e}")

if __name__ == "__main__":
    main()
