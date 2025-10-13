
import os
import json
import sys
import argparse
from pathlib import Path
import configparser

import numpy as np
import mss
import cv2
import yaml

# -------- Tunables (adjust once if needed) --------
COEFF_BASE = 0.24   # Base fraction of screen height at 100% HUD & 100% Minimap
MARGIN_PX  = 16     # Edge padding from screen borders
MIN_SIDE   = 200    # Minimum minimap side (px) safety clamp
MAX_SIDE_FRACTION = 0.42  # Avoid absurd sizes (cap to 42% of H)

CONFIG_PATH = Path("config.yaml")

# -------- Helpers --------
def load_yaml(path):
    if path.exists():
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}

def save_yaml(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f)
    print(f"[info] wrote {path}")

def get_desktop_resolution():
    # Desktop res is fine for borderless/windowed in LoL
    try:
        import ctypes
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        # Fallback via MSS
        with mss.mss() as sct:
            mon = sct.monitors[1]
            return mon["width"], mon["height"]

def read_persisted_settings():
    """
    Try common paths for PersistedSettings.json or game.cfg
    Return dict with keys: hud_scale (0..1+), minimap_scale (0..1+), left_minimap (bool), width, height
    Any missing values are None.
    """
    result = dict(hud_scale=None, minimap_scale=None, left_minimap=None, width=None, height=None)

    # Likely locations (adjust if your install differs)
    candidates = [
        # Riot install
        Path(r"C:\Riot Games\League of Legends\Config\PersistedSettings.json"),
        Path(r"C:\Riot Games\League of Legends\Config\game.cfg"),
        # OneDrive or user profile variants
        Path(os.path.expandvars(r"%LOCALAPPDATA%")) / r"Riot Games\League of Legends\Config\PersistedSettings.json",
        Path(os.path.expandvars(r"%LOCALAPPDATA%")) / r"Riot Games\League of Legends\Config\game.cfg",
        # Legacy Documents path
        Path.home() / r"Documents\League of Legends\Config\PersistedSettings.json",
        Path.home() / r"Documents\League of Legends\Config\game.cfg",
    ]

    # JSON reader
    def parse_persisted_json(p: Path):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return
        # Persisted schema varies; search heuristically
        # Look for "HUD", "HUDScale", "MinimapScale", "EnableLeftMinimap", "Height", "Width"
        s = json.dumps(data).lower()
        # naive scrapes
        def find_float(key):
            # crude pattern: "<key>": <num>
            import re
            m = re.search(rf'"{key.lower()}"\s*:\s*([0-9.]+)', s)
            return float(m.group(1)) if m else None
        def find_bool(key):
            import re
            m = re.search(rf'"{key.lower()}"\s*:\s*(true|false)', s)
            return True if (m and m.group(1) == "true") else (False if m else None)
        def find_int(key):
            import re
            m = re.search(rf'"{key.lower()}"\s*:\s*([0-9]+)', s)
            return int(m.group(1)) if m else None

        # Try common field names
        result["hud_scale"]      = result["hud_scale"]      or find_float("HudScale")
        result["minimap_scale"]  = result["minimap_scale"]  or find_float("MinimapScale")
        result["left_minimap"]   = result["left_minimap"]   if result["left_minimap"] is not None else find_bool("EnableLeftMinimap")
        result["width"]          = result["width"]          or find_int("Width")
        result["height"]         = result["height"]         or find_int("Height")

    # INI (game.cfg) reader
    def parse_game_cfg(p: Path):
        try:
            cp = configparser.ConfigParser()
            cp.read(p, encoding="utf-8")
        except Exception:
            return
        # common sections/keys (names vary by build; we try a few)
        for sec in ("General", "HUD", "UnitRenderStyle", "Performance"):
            if cp.has_section(sec):
                for k in cp[sec]:
                    v = cp[sec][k].strip()
                    lk = k.lower()
                    if lk in ("height",) and result["height"] is None:
                        try: result["height"] = int(v)
                        except: pass
                    if lk in ("width",) and result["width"] is None:
                        try: result["width"] = int(v)
                        except: pass
                    if lk in ("hudscale", "globalscale") and result["hud_scale"] is None:
                        try: result["hud_scale"] = float(v)
                        except: pass
                    if lk in ("minimapscale", "minimapsize") and result["minimap_scale"] is None:
                        try: result["minimap_scale"] = float(v)
                        except: pass
                    if lk in ("enableleftminimap", "leftminimap") and result["left_minimap"] is None:
                        lv = v.lower()
                        result["left_minimap"] = (lv in ("1", "true", "yes", "on"))

    for p in candidates:
        if not p.exists(): 
            continue
        if p.suffix.lower() == ".json":
            parse_persisted_json(p)
        elif p.suffix.lower() in (".cfg", ".ini"):
            parse_game_cfg(p)

    return result

def compute_minimap_roi(W, H, minimap_scale, hud_scale, left_minimap=False, margin=MARGIN_PX):
    # clamp and defaults
    minimap_scale = float(minimap_scale) if minimap_scale is not None else 1.0
    hud_scale = float(hud_scale) if hud_scale is not None else 1.0

    side = int(H * COEFF_BASE * hud_scale * minimap_scale)
    side = max(MIN_SIDE, min(side, int(H * MAX_SIDE_FRACTION)))

    if left_minimap:
        left = margin
    else:
        left = W - side - margin
    top = H - side - margin

    roi = dict(left=int(left), top=int(top), width=int(side), height=int(side))
    return roi, side

def grab_and_save_snapshot(roi, out_path="roi_snapshot.png", monitor_index=1):
    with mss.mss() as sct:
        mon = sct.monitors[monitor_index]
        # roi is relative to this monitor's origin; MSS expects absolute
        grab_rect = {
            "left": mon["left"] + roi["left"],
            "top":  mon["top"]  + roi["top"],
            "width": roi["width"],
            "height": roi["height"],
        }
        raw = sct.grab(grab_rect)
        img = np.array(raw, dtype=np.uint8)[:, :, :3]
        cv2.imwrite(out_path, img)
    return out_path

def main():
    global COEFF_BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--hud", type=float, default=None, help="HUD scale (1.0 = 100%) if not auto-detected")
    ap.add_argument("--minimap", type=float, default=None, help="Minimap size scale (1.0 = 100%) if not auto-detected")
    ap.add_argument("--left", action="store_true", help="Use left-side minimap")
    ap.add_argument("--coeff", type=float, default=COEFF_BASE, help="Override base coefficient (default 0.24)")
    ap.add_argument("--margin", type=int, default=MARGIN_PX, help="Edge margin in pixels")
    ap.add_argument("--monitor", type=int, default=1, help="MSS monitor index (1 = primary)")
    ap.add_argument("--snapshot", action="store_true", help="Save a snapshot of the computed ROI")
    args = ap.parse_args()

    
    COEFF_BASE = float(args.coeff)

    # 1) resolution
    W, H = get_desktop_resolution()
    print(f"[info] desktop resolution: {W}x{H}")

    # 2) try to read LoL settings
    s = read_persisted_settings()
    hud_scale = args.hud if args.hud is not None else s["hud_scale"]
    minimap_scale = args.minimap if args.minimap is not None else s["minimap_scale"]
    left_minimap = args.left if args.left else (s["left_minimap"] if s["left_minimap"] is not None else False)

    print(f"[info] detected settings -> HUD scale: {hud_scale}, Minimap scale: {minimap_scale}, Left minimap: {left_minimap}")

    # 3) compute ROI
    roi, side = compute_minimap_roi(W, H, minimap_scale, hud_scale, left_minimap, args.margin)
    print(f"[info] computed ROI: {roi} (side={side}px)")

    # 4) write config.yaml (merge if exists)
    cfg = load_yaml(CONFIG_PATH)
    cfg.setdefault("monitor_index", args.monitor)
    cfg.setdefault("target_fps", 20)
    cfg.setdefault("show_scale", 1.0)
    cfg["roi"] = roi
    save_yaml(CONFIG_PATH, cfg)

    # 5) optional snapshot
    if args.snapshot:
        out = grab_and_save_snapshot(roi, out_path="roi_snapshot.png", monitor_index=args.monitor)
        print(f"[snap] saved verification snapshot -> {out}")

if __name__ == "__main__":
    main()
