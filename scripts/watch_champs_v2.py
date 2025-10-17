
#!/usr/bin/env python3
# Minimap champions watcher v2 (ally-blue ring targeting, no audio)
# Key upgrades:
#   * Stage A uses BLUE ring HSV mask (ally champs) to avoid red jungle/camp clutter
#   * Stricter circle detection
#   * Higher template threshold; better smoothing (no flicker)
#   * Simple NMS to keep multiple champs if present

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
        "ring_color": "blue",     # "blue" for ally; "red" for enemy (revealed). Default ally.
        "ring_pad_frac": 0.30,
        "tmpl_scales": [0.9, 1.0, 1.1],
        "champ_thresh": 0.92,     # stricter to avoid mislabels
        "cooldown_sec": 3.0,
        "max_detections": 3       # draw up to K champs per frame
    },
    "smoothing": {
        "frames_confirm": 1,      # confirm on first positive to reduce flicker
        "frames_release": 3,      # clear after a few negatives
        "box_hold_sec": 1.8       # keep drawing a bit longer
    },
    "template_dir": "assets/templates/champ_minimap"
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
        self.alias = alias      # lowercase alias
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

# ----------------- Stage A: ring candidates (ALLY BLUE) -----------------
def blue_ring_mask(bgr: np.ndarray) -> np.ndarray:
    """HSV mask tuned to LoL minimap ally ring (cyan/blue)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Two lobes around cyan/blue; tweak if needed per monitor gamma
    m1 = cv2.inRange(hsv, (85, 80, 80), (105, 255, 255))   # cyan
    m2 = cv2.inRange(hsv, (106, 80, 80), (125, 255, 255))  # blue
    mask = cv2.bitwise_or(m1, m2)
    k = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask

def red_ring_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 120, 90), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, 120, 90), (180, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    k = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask

def find_ring_candidates(bgr: np.ndarray, pad_frac: float, color: str = "blue",
                         radius_px: Optional[Tuple[int,int]] = None):
    if color == "red":
        mask = red_ring_mask(bgr)
    else:
        mask = blue_ring_mask(bgr)

    blurred = cv2.GaussianBlur(mask, (7, 7), 1.5)
    H, W = bgr.shape[:2]
    if radius_px is None:
        minR = max(10, int(round(min(H, W) * 0.034)))
        maxR = max(14, int(round(min(H, W) * 0.062)))
    else:
        minR, maxR = radius_px

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=24,
        param1=140, param2=20, minRadius=minR, maxRadius=maxR
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
class ChampHit(Tuple):
    # (score, alias, x1, y1, x2, y2)
    pass

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

# ----------------- NMS -----------------
def nms_boxes(boxes: List[Tuple[int,int,int,int]], scores: List[float], iou_thresh=0.3, topk=3):
    if not boxes:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0 and len(keep) < topk:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.clip(xx2 - xx1 + 1, 0, None)
        h = np.clip(yy2 - yy1 + 1, 0, None)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

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
    SMOOTH = cfg["smoothing"]
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
    window = "Champ Watcher v2  (v toggle, d debug, m mask, q/esc quit)"
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

        present_run = 0
        absent_run = 0
        present_latched = False
        box_latched_until = 0.0
        flash_until = 0.0

        # cache last detections
        last_hits = []  # list of dicts per detection {box,label,score}

        while True:
            t0 = time.time()
            try:
                bgr = grab_roi_frame(sct, mon, roi)

                # Stage A: ally BLUE ring candidates
                mask, cands = find_ring_candidates(
                    bgr,
                    pad_frac=det.get("ring_pad_frac", 0.30),
                    color=det.get("ring_color", "blue"),
                    radius_px=None
                )
                vis = bgr.copy()
                if debug_on:
                    for (x, y, w, h) in cands:
                        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 0), 1)

                # Stage B: best champ for each candidate
                boxes = []
                scores = []
                labels = []
                tmpl_scales = tuple(det.get("tmpl_scales", [0.9, 1.0, 1.1]))
                champ_thresh = float(det.get("champ_thresh", 0.92))

                for (x, y, w, h) in cands:
                    patch = bgr[y:y+h, x:x+w]
                    m = best_champ_in_patch(patch, champ_templates, tmpl_scales, champ_thresh)
                    if m is None:
                        continue
                    sc, alias, px1, py1, px2, py2 = m
                    gx1, gy1, gx2, gy2 = x + px1, y + py1, x + px2, y + py2
                    boxes.append([gx1, gy1, gx2, gy2])
                    scores.append(sc)
                    labels.append(humanize_alias(alias))

                # NMS to filter overlaps and take top-K
                keep_idx = nms_boxes(boxes, scores, iou_thresh=0.3, topk=int(det.get("max_detections", 3))) if boxes else []
                hits = []
                for i in keep_idx:
                    x1, y1, x2, y2 = boxes[i]
                    sc = scores[i]
                    lab = labels[i]
                    hits.append({"box": (x1, y1, x2, y2), "score": sc, "label": lab})
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    y_txt = max(12, y1 - 6)
                    cv2.putText(vis, f"{lab} ({sc:.2f})", (x1, y_txt),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                now = time.time()
                present_raw = len(hits) > 0
                if present_raw:
                    present_run = min(present_run + 1, 10**6)
                    absent_run = 0
                else:
                    absent_run = min(absent_run + 1, 10**6)
                    present_run = 0

                if not present_latched and present_run >= int(SMOOTH["frames_confirm"]):
                    present_latched = True
                    flash_until = now + 0.6
                    box_latched_until = now + float(SMOOTH["box_hold_sec"])

                if present_latched and absent_run >= int(SMOOTH["frames_release"]):
                    present_latched = False

                if present_raw:
                    box_latched_until = max(box_latched_until, now + float(SMOOTH["box_hold_sec"]))

                # draw latched (last) hits if current frame empty
                if (now < box_latched_until) and not present_raw and last_hits:
                    for h in last_hits:
                        (x1, y1, x2, y2) = h["box"]
                        lab = h["label"]; sc = h["score"]
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
                        y_txt = max(12, y1 - 6)
                        cv2.putText(vis, f"{lab} ({sc:.2f})", (x1, y_txt),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 2)

                if time.time() < flash_until:
                    H, W = vis.shape[:2]
                    thickness = 5
                    cv2.rectangle(vis, (0, 0), (W-1, H-1), (0, 200, 255), thickness)

                # store current hits as last for hold
                if present_raw:
                    last_hits = hits

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
                    show_mask = False if show_mask else True  # toggle only mask for brevity
                    hot["toggle"] = False
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

    cv2.destroyAllWindows()
    try:
        listener.stop()
    except Exception as e:
        print(f"[error] listener failed: {e}")

if __name__ == "__main__":
    main()
