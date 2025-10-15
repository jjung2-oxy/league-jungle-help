import time
from pathlib import Path
import yaml
import cv2
import mss
import numpy as np
from pynput import keyboard

hot = {"photo": False, "save": False, "draw": False, "quit": False}

def on_press(key):
    try:
        k = key.char.lower() if hasattr(key, "char") and key.char else ""
    except Exception:
        k = ""
    # letter keys
    if k == "p":
        hot["photo"] = True
    elif k == "s":
        hot["save"]  = True
    elif k == "r":
        hot["draw"]  = True
    # function keys/esc (also allow F9/F10 as alternates)
    if key == keyboard.Key.esc:
        hot["quit"] = True
    elif key == keyboard.Key.f9:
        hot["photo"] = True
    elif key == keyboard.Key.f10:   
        hot["save"] = True

listener = keyboard.Listener(on_press=on_press)
listener.daemon = True
listener.start()

CONFIG_PATH = Path("config.yaml")

DEFAULT_CONFIG = {
    "roi": { "left": None, "top": None, "width": 360, "height": 360 },
    "target_fps": 20,
    "monitor_index": 1,   # 1 = primary (mss uses 1-based indexing for monitors)
    "show_scale": 1.0     # set <1.0 to shrink preview, e.g., 0.8
}

# ----------------- Config helpers -----------------
def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}
    # merge defaults
    def merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict):
                a[k] = merge(a.get(k, {}) or {}, v)
            else:
                a.setdefault(k, v)
        return a
    cfg = merge(cfg, DEFAULT_CONFIG)
    # save back to ensure missing keys are written
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(cfg, f)
    return cfg

def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(cfg, f)
    print("[info] config saved:", cfg)

# ----------------- ROI draw state -----------------
class RoiDrawer:
    def __init__(self):
        self.dragging = False
        self.start = None
        self.end = None
        self.final_rect = None  # (x, y, w, h) in preview space

    def reset(self):
        self.dragging = False
        self.start = None
        self.end = None
        self.final_rect = None

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start = (x, y)
            self.end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            self.end = (x, y)
            x1, y1 = self.start
            x2, y2 = self.end
            x_min, y_min = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)
            if w >= 10 and h >= 10:
                self.final_rect = (x_min, y_min, w, h)
                
                
                
def ensure_window_alive(window_name, mouse_cb):
    try:
        visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
    except Exception:
        visible = -1
    if visible < 1:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 360, 360)
        cv2.moveWindow(window_name, 64, 64)
        try: cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        except: pass
        cv2.setMouseCallback(window_name, mouse_cb)
        print("[ui] window recreated")

# ----------------- Main -----------------
def main():
    cfg = load_config()
    target_period = 1.0 / max(1, int(cfg["target_fps"]))
    preview_scale = float(cfg.get("show_scale", 1.0))

    # screenshot dirs
    ss_root = Path("screenshots")
    (ss_root / "raw").mkdir(parents=True, exist_ok=True)
    (ss_root / "overlay").mkdir(parents=True, exist_ok=True)

    drawer = RoiDrawer()
    window = "LOL Minimap Capture (r=draw ROI, p=photo, s=save ROI, q=quit)"
    # --- window setup ---
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 480, 480)  # small, movable
    cv2.moveWindow(window, 100, 100)    # ensure it's on-screen
    try:
        cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)  # keep on top (Windows)
    except Exception:
        pass
    cv2.setMouseCallback(window, drawer.on_mouse)


    with mss.mss() as sct:
        monitors = sct.monitors
        if not (1 <= cfg["monitor_index"] < len(monitors)):
            print(f"[warn] monitor_index={cfg['monitor_index']} invalid. Using primary (1).")
            cfg["monitor_index"] = 1
        mon = monitors[cfg["monitor_index"]]

        roi_left, roi_top, roi_w, roi_h = cfg["roi"]["left"], cfg["roi"]["top"], cfg["roi"]["width"], cfg["roi"]["height"]
        if None in (roi_left, roi_top):
            screen_w, screen_h = mon["width"], mon["height"]
            roi_left = mon["left"] + screen_w - roi_w
            roi_top  = mon["top"]  + screen_h - roi_h
            print("[info] ROI not set; defaulting to bottom-right.")
            cfg["roi"].update({"left": int(roi_left), "top": int(roi_top), "width": int(roi_w), "height": int(roi_h)})
            save_config(cfg)

        fps_smooth = None
        last_t = time.time()
        pending_save = False

        # keep most recent raw/overlay to save when 'p' is pressed
        last_raw_bgr = None
        last_overlay_bgr = None

        while True:
            t0 = time.time()
            roi = {
                "left": int(cfg["roi"]["left"]),
                "top": int(cfg["roi"]["top"]),
                "width": int(cfg["roi"]["width"]),
                "height": int(cfg["roi"]["height"]),
            }
            assert roi["width"] > 0 and roi["height"] > 0, "ROI has zero width/height"

            # ---- capture (safe & contiguous) ----
            raw = sct.grab(roi)  # BGRA
            frame = np.array(raw, dtype=np.uint8)        # materialize
            bgr = frame[:, :, :3]                        # drop alpha
            bgr = np.ascontiguousarray(bgr)              # ensure C-contiguous

            # keep a copy of the raw ROI for screenshots
            last_raw_bgr = bgr.copy()

            # ---- preview scaling ----
            if preview_scale != 1.0:
                w = max(1, int(bgr.shape[1] * preview_scale))
                h = max(1, int(bgr.shape[0] * preview_scale))
                bgr_disp = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
            else:
                bgr_disp = bgr.copy()
            bgr_disp = np.ascontiguousarray(bgr_disp)

            # ---- drawing (drag rect or persisted frame border) ----
            if drawer.dragging and drawer.start and drawer.end:
                x1, y1 = drawer.start
                x2, y2 = drawer.end
                cv2.rectangle(bgr_disp, (x1, y1), (x2, y2), (0, 255, 255), 2)

            if drawer.final_rect:
                px, py, pw, ph = drawer.final_rect
                scale_inv = 1.0 / preview_scale
                new_left = int(cfg["roi"]["left"] + px * scale_inv)
                new_top  = int(cfg["roi"]["top"] + py * scale_inv)
                new_w    = max(40, int(pw * scale_inv))
                new_h    = max(40, int(ph * scale_inv))
                cfg["roi"].update({"left": new_left, "top": new_top, "width": new_w, "height": new_h})
                drawer.reset()
                pending_save = True

            # draw border around the shown frame (for orientation)
            cv2.rectangle(bgr_disp, (0, 0), (bgr_disp.shape[1]-1, bgr_disp.shape[0]-1), (0, 200, 0), 2)

            # ---- HUD ----
            now = time.time()
            dt = now - last_t
            last_t = now
            inst_fps = 1.0 / max(dt, 1e-6)
            fps_smooth = inst_fps if fps_smooth is None else (0.9 * fps_smooth + 0.1 * inst_fps)

            hud = f"ROI=({cfg['roi']['left']},{cfg['roi']['top']},{cfg['roi']['width']},{cfg['roi']['height']})  FPS={fps_smooth:.1f}"
            if pending_save:
                hud += "  [unsaved changes: press 's']"
            cv2.putText(bgr_disp, hud, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # keep a copy of overlay for screenshots
            last_overlay_bgr = bgr_disp.copy()

            cv2.imshow(window, bgr_disp)
            ensure_window_alive(window)
            

            # Use waitKeyEx to capture more keys; only works when window has focus
            key = cv2.waitKeyEx(1)
            if key != -1:
                # debug print the keycode once per press
                print(f"[key] OpenCV keycode: {key}")
                k = key & 0xFF
                if k in (ord('q'), 27):           # q or ESC
                    hot["quit"] = True
                elif k == ord('r'):
                    hot["draw"] = True
                elif k == ord('s'):
                    hot["save"] = True
                elif k == ord('p'):
                    hot["photo"] = True

            # Also check global hotkeys (works even if window focus is elsewhere)
            if hot["quit"]:
                print("[quit] requested by hotkey (Ctrl+Q or F12).")
                break


            if hot["draw"]:
                drawer.reset()
                print("[info] Draw a rectangle in the preview to set a new ROI, then press 's' to save.")
                hot["draw"] = False

            if hot["save"]:
                if pending_save:
                    save_config(cfg)
                    pending_save = False
                else:
                    print("[info] No changes to save.")
                hot["save"] = False

            if hot["photo"]:
                # save last_raw_bgr and last_overlay_bgr (as you already implemented)
                # ... your screenshot code here ...
                hot["photo"] = False

            # optional: verify window is actually visible
            vis = cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE)
            if vis < 1:
                print("[warn] Preview window not visible (minimized or covered). Bring it to front.")

            # pacing...
            elapsed = time.time() - t0
            if elapsed < target_period:
                time.sleep(target_period - elapsed)

        cv2.destroyAllWindows()
        try:
            listener.stop()
        except Exception:
            pass

if __name__ == "__main__":
    main()