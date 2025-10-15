
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mss
import numpy as np
import simpleaudio as sa
import yaml
from pynput import keyboard

CONFIG_PATH = Path("config.yaml")

DEFAULTS = {
    "roi": {"left": None, "top": None, "width": 360, "height": 360},
    "target_fps": 18,
    "monitor_index": 1,
    "detector": {
        "frames_confirm": 2,
        "cooldown_sec": 6.0,
        # Stage A defaults; will be replaced by learned HSV range if possible
        "ring_pad_frac": 0.35,
        "ring_min_area": 120,      # keep broad; we refine with color
        "ring_max_area": 6000,
        "ring_min_circ": 0.60,
        # Stage B defaults
        "tmpl_scales": [0.8, 0.9, 1.0, 1.1, 1.2],
        "tmpl_thresh": 0.62
    },
    "beep_wav": "assets/beep.wav",
    "template_dir": "assets/templates/dummy"
}

# Smooth presence + UI hold
SMOOTH = {
    "frames_confirm": 2,     # frames in a row needed to "be present"
    "frames_release": 3,     # frames in a row needed to "clear"
    "box_hold_sec": 1.5      # keep drawing box this long after last positive
}

GLOBAL_FB = {
    "enabled": True,
    "min_interval_sec": 2.0,   # don’t run more than once every 2s
    "run_every_n_frames": 6,   # only on every 6th frame
    "downscale": 0.7,          # search at 70% size then map back
    "accept_thresh": 0.82      # must be strong to accept
}

# ----------------- util: audio -----------------
def play_beep(beep_path: str):
    try:
        p = Path(beep_path)
        if p.exists():
            try:
                sa.WaveObject.from_wave_file(str(p)).play()
                return
            except Exception as e:
                print(f"[warn] wav failed ({e}); falling back")
        # fallback short tone via simpleaudio
        try:
            sr = 44100
            t = np.linspace(0, 0.18, int(0.18 * sr), False)
            tone = (np.sin(2 * np.pi * 880 * t) * 0.42).astype(np.float32)
            sa.play_buffer((tone * 32767).astype(np.int16), 1, 2, sr)
            return
        except Exception as e:
            print(f"[warn] tone failed ({e}); trying winsound")
        # Windows last resort
        try:
            import winsound
            winsound.Beep(880, 180)
        except Exception as e:
            print(f"[warn] winsound failed: {e}")
    except Exception as e:
        print(f"[warn] beep failed: {e}")

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
class Template:
    def __init__(self, path: Path, kind: str, gray: np.ndarray, mask: Optional[np.ndarray]):
        self.path = path
        self.kind = kind      # "ring" or "helmet" or "unknown"
        self.gray = gray
        self.mask = mask

def load_templates(dirpath: Path) -> List[Template]:
    tmpls: List[Template] = []
    for p in sorted(dirpath.glob("*.png")):
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[warn] failed to read template {p}")
            continue
        bgr = img[:, :, :3] if img.ndim == 3 else img
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        a_mask = None
        if img.ndim == 3 and img.shape[2] == 4:
            a_mask = cv2.threshold(img[:, :, 3], 1, 255, cv2.THRESH_BINARY)[1]
        # classify heuristic: ring has strong red rim; helmet template is beige/inner
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        # "redness" quick score
        red_mask_guess = ((hsv[:, :, 1] > 80) & ((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 168))).astype(np.uint8) * 255
        red_ratio = red_mask_guess.mean() / 255.0
        kind = "ring" if red_ratio > 0.10 else "helmet"  # 10%+ red pixels → likely ring crop
        tmpls.append(Template(p, kind, gray, a_mask))
    print(f"[info] loaded {len(tmpls)} template(s) from {dirpath}")
    return tmpls

# ----------------- learn HSV from a ring template -----------------
def learn_red_hsv_from_ring(tmpls: List[Template]) -> Optional[List[Tuple[Tuple[int,int,int], Tuple[int,int,int]]]]:
    # pick the first ring-like template
    ring_bgr = None
    for t in tmpls:
        if t.kind == "ring":
            ring_bgr = cv2.imread(str(t.path), cv2.IMREAD_COLOR)
            break
    if ring_bgr is None:
        return None

    hsv = cv2.cvtColor(ring_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # candidate ring pixels: red-dominant by channel heuristic + saturated
    red_dominant = (ring_bgr[:, :, 2] > ring_bgr[:, :, 1] * 1.2) & (ring_bgr[:, :, 2] > ring_bgr[:, :, 0] * 1.2)
    sat_high = s > 90
    val_ok = v > 70
    candidates = np.where(red_dominant & sat_high & val_ok, 1, 0).astype(np.uint8)

    if candidates.sum() < 25:
        return None

    hue_vals = h[candidates == 1].astype(np.float32)
    # hues wrap at 180; handle bimodal (0..10 and 170..180)
    left = hue_vals[hue_vals < 90]
    right = hue_vals[hue_vals >= 90]
    ranges = []

    def make_range(vec, pad=10):
        if vec.size == 0:
            return None
        m = float(np.median(vec)); mad = float(np.median(np.abs(vec - m)))
        lo = int(max(0, m - (2.5 * max(5, mad)) - pad))
        hi = int(min(180, m + (2.5 * max(5, mad)) + pad))
        return (lo, hi)

    r1 = make_range(left); r2 = make_range(right)
    s_min = 90; v_min = 70

    if r1 and (r1[1] - r1[0] >= 5):
        ranges.append(((r1[0], s_min, v_min), (min(r1[1], 90), 255, 255)))
    if r2 and (r2[1] - r2[0] >= 5):
        ranges.append(((max(r2[0], 90), s_min, v_min), (r2[1], 255, 255)))

    if not ranges:
        return None
    print(f"[learn] HSV red ranges from template: {ranges}")
    return ranges  # list of (lowHSV, highHSV)

def mask_red_hsv(bgr: np.ndarray, hsv_ranges: Optional[List[Tuple[Tuple[int,int,int], Tuple[int,int,int]]]]) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    if not hsv_ranges:
        # fallback wide red
        m1 = cv2.inRange(hsv, (0, 120, 90), (10, 255, 255))
        m2 = cv2.inRange(hsv, (170, 120, 90), (180, 255, 255))
        mask = cv2.bitwise_or(m1, m2)
    else:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in hsv_ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    k = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask

def red_dominance_mask(bgr: np.ndarray) -> np.ndarray:
    """
    Return a binary mask where R channel dominates G/B.
    Tuned for LoL minimap red ring under typical gamma.
    """
    b, g, r = cv2.split(bgr)
    rg = cv2.max(b, g)
    # strong R dominance + minimum brightness
    dom = (r.astype(np.int16) - rg.astype(np.int16))  # signed
    m = (dom > 35) & (r > 110)  # loosen/tighten if needed
    mask = (m.astype(np.uint8) * 255)

    # clean up
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask

def brute_best_template_over_roi(bgr_roi, templates, scales=(0.9,1.0,1.1), downscale=1.0):
    if not templates:
        return None
    if downscale != 1.0:
        bgr_small = cv2.resize(bgr_roi, (0,0), fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
        scale_back = 1.0 / downscale
    else:
        bgr_small = bgr_roi
        scale_back = 1.0

    gray = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2GRAY)
    best = None
    for t in templates:
        timg, tmask = t.gray, t.mask
        for s in scales:
            thh = int(max(1, round(timg.shape[0] * s * downscale)))
            tww = int(max(1, round(timg.shape[1] * s * downscale)))
            if thh < 10 or tww < 10 or thh > gray.shape[0] or tww > gray.shape[1]:
                continue
            rtmpl = cv2.resize(timg, (tww, thh), interpolation=cv2.INTER_AREA)
            rmask = cv2.resize(tmask, (tww, thh), interpolation=cv2.INTER_NEAREST) if tmask is not None else None
            if rmask is not None:
                res = cv2.matchTemplate(gray, rtmpl, cv2.TM_CCORR_NORMED, mask=rmask)
            else:
                res = cv2.matchTemplate(gray, rtmpl, cv2.TM_CCOEFF_NORMED)
            _, mval, _, mloc = cv2.minMaxLoc(res)
            if (best is None) or (mval > best[0]):
                x, y = mloc
                # map back to original ROI coords
                best = (float(mval),
                        int(round(x*scale_back)), int(round(y*scale_back)),
                        int(round((x+rtmpl.shape[1])*scale_back)),
                        int(round((y+rtmpl.shape[0])*scale_back)))
    return best

# ----------------- Stage A: ring candidates -----------------
def find_ring_candidates(bgr: np.ndarray, pad_frac: float,
                         radius_px: Optional[Tuple[int,int]] = None):
    """
    Find circular red rims using HoughCircles on a red-dominance mask.
    Returns: mask, [(x,y,w,h), ...]
    """
    mask = red_dominance_mask(bgr)

    # Hough on a slightly blurred mask
    blurred = cv2.GaussianBlur(mask, (7, 7), 1.5)
    # Estimate radius range if not provided; for 400x400 minimap the ring radius is ~14–22
    H, W = bgr.shape[:2]
    if radius_px is None:
        minR = max(8, int(round(min(H, W) * 0.032)))   # ~400*0.032 ≈ 12-13
        maxR = max(12, int(round(min(H, W) * 0.060)))  # ~400*0.060 ≈ 24
    else:
        minR, maxR = radius_px

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
        param1=120, param2=12, minRadius=minR, maxRadius=maxR
    )

    boxes = []
    if circles is not None:
        circles = np.uint16(np.around(circles[0, :]))
        H, W = bgr.shape[:2]
        for (cx, cy, r) in circles:
            # convert circle to padded box
            x = int(max(0, cx - r))
            y = int(max(0, cy - r))
            w = int(min(W - x, 2 * r))
            h = int(min(H - y, 2 * r))
            pad = int(pad_frac * max(w, h))
            x = max(0, x - pad); y = max(0, y - pad)
            w = min(W - x, w + 2 * pad); h = min(H - y, h + 2 * pad)
            boxes.append((x, y, w, h))

    return mask, boxes

# ----------------- Stage B: helmet match -----------------
def match_helmet_in_patch(patch_bgr: np.ndarray, templates: List[Template],
                          scales: Tuple[float, ...], thresh: float):
    if not templates:
        return None
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    best = None  # (score, x1, y1, x2, y2)
    for t in templates:
        if t.kind == "ring":
            continue  # prefer helmet/inner for matching
        timg, tmask = t.gray, t.mask
        for s in scales:
            thh = int(max(1, round(timg.shape[0] * s)))
            tww = int(max(1, round(timg.shape[1] * s)))
            if thh < 10 or tww < 10 or thh > gray.shape[0] or tww > gray.shape[1]:
                continue
            rtmpl = cv2.resize(timg, (tww, thh), interpolation=cv2.INTER_AREA)
            rmask = cv2.resize(tmask, (tww, thh), interpolation=cv2.INTER_NEAREST) if tmask is not None else None
            if rmask is not None:
                res = cv2.matchTemplate(gray, rtmpl, cv2.TM_CCORR_NORMED, mask=rmask)
            else:
                res = cv2.matchTemplate(gray, rtmpl, cv2.TM_CCOEFF_NORMED)
            _, maxVal, _, maxLoc = cv2.minMaxLoc(res)
            if maxVal < thresh:
                continue
            x, y = maxLoc
            cand = (float(maxVal), x, y, x + tww, y + thh)
            if best is None or cand[0] > best[0]:
                best = cand
    return best

# ----------------- UI helpers -----------------
class DragBox:
    def __init__(self):
        self.dragging = False
        self.start = None
        self.end = None
        self.box = None
    def reset(self):
        self.__init__()
    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True; self.start = (x, y); self.end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False; self.end = (x, y)
            x1, y1 = self.start; x2, y2 = self.end
            x0, y0 = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)
            if w >= 8 and h >= 8:
                self.box = (x0, y0, w, h)

def save_template_from_drag(vis_img, box, out_dir: Path):
    x, y, w, h = box
    crop = vis_img[y:y+h, x:x+w].copy()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"helmet_{ts}.png"
    cv2.imwrite(str(out), crop)
    print(f"[template] saved {out}")
    return out

def ensure_window_alive(window_name, mouse_cb=None):
    """Recreate the OpenCV window if it disappears (e.g., from alt-tab or game fullscreen)."""
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
    templates = load_templates(tmpl_dir)

    # Learn HSV from any ring template (if available)
    learned_ranges = learn_red_hsv_from_ring(templates)
    if learned_ranges is None:
        print("[warn] no ring template found; using fallback wide red HSV.")

    # flags + hotkeys
    show = True
    debug_on = True
    show_mask = True
    hot = {
        "tmpl": False, "photo": False, "toggle": False,
        "debug": False, "mask": False, "reload": False, "quit": False
    }
    mods = {"ctrl": False}
    from pynput.keyboard import Key, KeyCode

    def on_press(key):
        try:
            k = key.char.lower() if hasattr(key, "char") and key.char else ""
        except Exception:
            k = ""
        if k == "t": hot["tmpl"] = True
        elif k == "p": hot["photo"] = True
        elif k == "v": hot["toggle"] = True
        elif k == "d": hot["debug"] = True
        elif k == "m": hot["mask"] = True
        elif k == "r": hot["reload"] = True
        # Require Ctrl+Q to quit (avoid accidental ESC from game)
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

    last_alert_t = 0.0
    target_period = 1.0 / max(1, int(cfg["target_fps"]))

    window = "Dummy Watcher  (t capture, r reload, d debug, m mask, p photo, v toggle, q/esc quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 360, 360)
    cv2.moveWindow(window, 64, 64)
    try:
        cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    drag = DragBox()
    cv2.setMouseCallback(window, drag.on_mouse)

    with mss.mss() as sct:
        mon = sct.monitors[cfg["monitor_index"]]
        roi = clamp_roi_to_monitor(roi, mon)
        last_raw = None
        last_overlay = None

        frame_idx = 0
        last_global_try_t = 0.0
        suspend_global_until = 0.0
        last_event = {"stage": "none", "score": 0.0, "cands": 0, "t": 0.0}
        # smoothing / latching
        present_run = 0        # consecutive positives
        absent_run = 0         # consecutive negatives
        present_latched = False
        box_latched_until = 0.0

        # cache the last confirmed box so we can draw during hold
        last_box = None            # (x1, y1, x2, y2)
        last_box_stage = None      # "A+B" or "GLOBAL"
        last_box_score = 0.0

        while True:
            t0 = time.time()
            boxes, scores = [], []
            present = False

            try:
                bgr = grab_roi_frame(sct, mon, roi)

                # Stage A
                mask, cands = find_ring_candidates(
                    bgr,
                    pad_frac=det.get("ring_pad_frac", 0.35),
                    radius_px=None  # or (12, 26) if you want to force it
                )

                vis = bgr.copy()
                if debug_on:
                    for (x, y, w, h) in cands:
                        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 0), 1)

                # ---- Stage B: best helmet match across Stage-A candidates ----
                best = None                 # (score, (x1,y1,x2,y2))
                have_helmet = any(t.kind == "helmet" for t in templates)

                # use det config if you have it; otherwise fallback defaults
                tmpl_scales = tuple(det.get("tmpl_scales", [0.8, 0.9, 1.0, 1.1, 1.2]))
                tmpl_thresh = float(det.get("tmpl_thresh", 0.62))

                for (x, y, w, h) in cands:
                    patch = bgr[y:y+h, x:x+w]
                    m = match_helmet_in_patch(
                        patch,
                        templates if have_helmet else templates,   # (kept for future branching)
                        scales=tmpl_scales,
                        thresh=tmpl_thresh,
                    )
                    if m is None:
                        continue
                    sc, px1, py1, px2, py2 = m
                    gx1, gy1, gx2, gy2 = x + px1, y + py1, x + px2, y + py2
                    if (best is None) or (sc > best[0]):
                        best = (sc, (gx1, gy1, gx2, gy2))

                boxes, scores = [], []
                present = False

                if best is not None:
                    sc, (x1, y1, x2, y2) = best
                    boxes, scores = [[x1, y1, x2, y2]], [sc]
                    present = True
                    last_event = {"stage": "A+B", "score": sc, "cands": len(cands), "t": time.time()}
                    # draw box + score
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis, f"{sc:.2f}", (x1, max(0, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    # cache last box
                    last_box = (x1, y1, x2, y2)
                    last_box_stage = "A+B"
                    last_box_score = sc

                # ---- GLOBAL fallback (rate-limited & downscaled) ----
                now = time.time()
                should_try_global = (
                    GLOBAL_FB["enabled"]
                    and not present
                    and len(cands) == 0
                    and now >= suspend_global_until
                    and (now - last_global_try_t) >= GLOBAL_FB["min_interval_sec"]
                    and (frame_idx % GLOBAL_FB["run_every_n_frames"] == 0)
                )

                if not present and should_try_global:
                    last_global_try_t = now
                    g = brute_best_template_over_roi(
                        bgr,
                        templates,
                        scales=(0.9, 1.0, 1.1),
                        downscale=GLOBAL_FB["downscale"],
                    )
                    if g and g[0] >= GLOBAL_FB["accept_thresh"]:
                        sc, x1, y1, x2, y2 = g
                        boxes, scores = [[x1, y1, x2, y2]], [sc]
                        present = True
                        last_event = {"stage": "GLOBAL", "score": sc, "cands": 0, "t": now}
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)  # yellow = fallback
                        cv2.putText(vis, f"{sc:.2f} (global)", (x1, max(0, y1 - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        # cache last box
                        last_box = (x1, y1, x2, y2)
                        last_box_stage = "GLOBAL"
                        last_box_score = sc

                # ---- Presence smoothing / hysteresis ----
                now = time.time()
                present_raw = len(boxes) > 0

                if present_raw and last_event.get("stage") == "GLOBAL":
                    # treat strong global as instant confirm
                    present_run = max(1, SMOOTH["frames_confirm"])
                else:
                    # normal update
                    pass

                if present_raw:
                    present_run = min(present_run + 1, 10**6)
                    absent_run = 0
                else:
                    absent_run = min(absent_run + 1, 10**6)
                    # don't increment present_run on negatives
                    present_run = 0

                # confirm presence if we've seen enough positives
                if not present_latched and present_run >= SMOOTH["frames_confirm"]:
                    present_latched = True            # rising edge
                    # Beep on rising edge (and apply cooldown you already track)
                    if (now - last_alert_t) >= det["cooldown_sec"]:
                        print(f"[alert] Dummy via {last_event['stage']} | score={last_event['score']:.2f}")
                        play_beep(cfg["beep_wav"])
                        last_alert_t = now
                        suspend_global_until = now + det["cooldown_sec"]  # keep GLOBAL off during cooldown
                    # keep drawing the box for a short time even if the next frame drops
                    box_latched_until = now + SMOOTH["box_hold_sec"]

                # release presence only after several negatives (prevents flicker)
                if present_latched and absent_run >= SMOOTH["frames_release"]:
                    present_latched = False

                # extend box hold while positives continue
                if present_raw:
                    box_latched_until = max(box_latched_until, now + SMOOTH["box_hold_sec"])

                # ---- Draw the last box while latched/held ----
                draw_box = present_raw or (now < box_latched_until)
                if draw_box:
                    if present_raw and boxes:
                        # current frame already drew its box above for A+B or GLOBAL
                        pass
                    elif last_box is not None:
                        x1, y1, x2, y2 = last_box
                        color = (0, 255, 0) if (last_box_stage == "A+B") else (0, 255, 255)
                        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(vis, f"{last_box_score:.2f}",
                                    (x1, max(0, y1 - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Mask inset (top-right) to visualize what Stage-A sees
                if show_mask:
                    mask_small = cv2.resize(mask, (120, 120), interpolation=cv2.INTER_NEAREST)
                    mask_small = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
                    H, W = vis.shape[:2]
                    x0 = W - 120 - 6; y0 = 24
                    vis[y0:y0+120, x0:x0+120] = mask_small
                    cv2.rectangle(vis, (x0, y0), (x0+120, y0+120), (0,255,255), 1)
                    cv2.putText(vis, "mask", (x0+4, y0+14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

                last_raw = bgr
                last_overlay = vis
                if show:
                    cv2.imshow(window, vis)
                    ensure_window_alive(window)

                # Local keys
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    hot["quit"] = True
                elif key == ord('v'):
                    hot["toggle"] = True
                elif key == ord('p'):
                    hot["photo"] = True
                elif key == ord('t'):
                    hot["tmpl"] = True
                elif key == ord('d'):
                    hot["debug"] = True
                elif key == ord('m'):
                    hot["mask"]  = True
                elif key == ord('r'):
                    hot["reload"] = True

                # Global hotkeys
                if hot["quit"]:
                    break
                if hot["toggle"]:
                    show = not show; hot["toggle"] = False
                if hot["debug"]:
                    debug_on = not debug_on; hot["debug"] = False
                if hot["mask"]:
                    show_mask = not show_mask; hot["mask"] = False
                if hot["photo"]:
                    Path("screenshots").mkdir(parents=True, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    if last_raw is not None:
                        cv2.imwrite(f"screenshots/roi_{ts}.png", last_raw)
                    if last_overlay is not None:
                        cv2.imwrite(f"screenshots/roi_overlay_{ts}.png", last_overlay)
                    print(f"[snap] screenshots/roi_{ts}.png")
                    hot["photo"] = False
                if hot["reload"]:
                    templates = load_templates(Path(cfg["template_dir"]))
                    learned = learn_red_hsv_from_ring(templates)
                    if learned is not None:
                        learned_ranges = learned
                    print("[info] reloaded templates.")
                    hot["reload"] = False

            except KeyboardInterrupt:
                print("[info] interrupted; exiting.")
                break
            except Exception as e:
                print(f"[error] frame failed: {e}")
                time.sleep(0.01)
                continue

            # pacing
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
