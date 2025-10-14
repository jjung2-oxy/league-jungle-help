#!/usr/bin/env python3
"""
Compute the League minimap ROI from resolution + Minimap size only (HUD scale ignored).

Usage:
  # simplest: compute and write config, plus a verification crop
  python auto_roi_from_config.py --snapshot

  # learn coeff/margin from your existing perfect ROI in config.yaml, then recompute
  python auto_roi_from_config.py --minimap 1.2 --fit-from-current --snapshot

  # override explicitly
  python auto_roi_from_config.py --minimap 1.2 --coeff 0.205761 --margin 20 --snapshot

  # if your minimap is on the left
  python auto_roi_from_config.py --left --minimap 1.2 --coeff 0.205761 --margin 20 --snapshot
"""

import os, re, json, argparse, configparser
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import cv2
import mss
import yaml

DEFAULT_COEFF = 0.206          # ~your learned coeff for 400px @ 1620p with 1.2 minimap
DEFAULT_MARGIN = 20            # px padding
MIN_SIDE = 120                 # safety clamp (don’t go smaller than this)
MAX_SIDE_FRACTION = 0.5        # cap to 50% of H just in case
DEFAULT_CONFIG_PATH = Path("config.yaml")

def load_yaml(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

def save_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(f"[info] wrote {path}")

def get_desktop_resolution(monitor_index: int = 1) -> Tuple[int, int, Dict]:
    with mss.mss() as sct:
        monitors = sct.monitors
        if not (1 <= monitor_index < len(monitors)):
            print(f"[warn] monitor_index={monitor_index} invalid, defaulting to primary (1).")
            monitor_index = 1
        mon = monitors[monitor_index]
        return mon["width"], mon["height"], mon

# Best-effort: read minimap size & left-minimap
def _read_text(p: Path) -> Optional[str]:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

def normalize_minimap(x: Optional[float]) -> Optional[float]:
    if x is None: return None
    try:
        x = float(x)
    except Exception:
        return None
    # interpret 85 → 0.85, 120 → 1.2
    if x > 5.0:
        x = x / 100.0
    # reject nonsense (we’ll default later)
    if x <= 0.1 or x > 3.0:
        return None
    return x

def read_lol_minimap_settings() -> dict:
    result = dict(minimap_scale=None, left_minimap=None, width=None, height=None)
    candidates = [
        Path(r"C:\Riot Games\League of Legends\Config\PersistedSettings.json"),
        Path(r"C:\Riot Games\League of Legends\Config\game.cfg"),
        Path(os.path.expandvars(r"%LOCALAPPDATA%")) / r"Riot Games\League of Legends\Config\PersistedSettings.json",
        Path(os.path.expandvars(r"%LOCALAPPDATA%")) / r"Riot Games\League of Legends\Config\game.cfg",
        Path.home() / r"Documents\League of Legends\Config\PersistedSettings.json",
        Path.home() / r"Documents\League of Legends\Config\game.cfg",
    ]

    def parse_json(p: Path):
        s = _read_text(p)
        if not s: return
        low = s.lower()
        def ffloat(key): 
            m = re.search(rf'"{re.escape(key.lower())}"\s*:\s*([0-9.]+)', low)
            return float(m.group(1)) if m else None
        def fbool(key):
            m = re.search(rf'"{re.escape(key.lower())}"\s*:\s*(true|false)', low)
            return (m.group(1) == "true") if m else None
        def fint(key):
            m = re.search(rf'"{re.escape(key.lower())}"\s*:\s*([0-9]+)', low)
            return int(m.group(1)) if m else None

        result["minimap_scale"] = result["minimap_scale"] or ffloat("MinimapScale")
        if result["left_minimap"] is None:
            result["left_minimap"] = fbool("EnableLeftMinimap")
        result["width"]  = result["width"]  or fint("Width")
        result["height"] = result["height"] or fint("Height")

    def parse_cfg(p: Path):
        cp = configparser.ConfigParser()
        try: cp.read(p, encoding="utf-8")
        except Exception: return
        for sec in ("General","HUD","Performance","UnitRenderStyle"):
            if not cp.has_section(sec): continue
            for k,v in cp[sec].items():
                lk, lv = k.lower().strip(), v.strip()
                if lk == "width" and result["width"] is None:
                    try: result["width"]=int(lv)
                    except: pass
                if lk == "height" and result["height"] is None:
                    try: result["height"]=int(lv)
                    except: pass
                if lk in ("minimapscale","minimapsize") and result["minimap_scale"] is None:
                    try: result["minimap_scale"]=float(lv)
                    except: pass
                if lk in ("enableleftminimap","leftminimap") and result["left_minimap"] is None:
                    result["left_minimap"] = lv.lower() in ("1","true","yes","on")

    for p in candidates:
        if not p.exists(): continue
        if p.suffix.lower()==".json": parse_json(p)
        elif p.suffix.lower() in (".cfg",".ini"): parse_cfg(p)

    result["minimap_scale"] = normalize_minimap(result["minimap_scale"])
    return result

def compute_minimap_roi(W:int, H:int, minimap_scale: Optional[float], left_minimap: bool, coeff: float, margin: int) -> Dict[str,int]:
    ms = normalize_minimap(minimap_scale) or 1.0
    side = int(H * float(coeff) * ms)
    side = max(MIN_SIDE, min(side, int(H * MAX_SIDE_FRACTION)))
    left = margin if left_minimap else (W - side - margin)
    top  = H - side - margin
    return {"left": int(left), "top": int(top), "width": int(side), "height": int(side)}

def learn_from_existing_roi(W:int, H:int, roi:Dict[str,int], minimap_scale: Optional[float]) -> Tuple[float,int]:
    side = int(roi["width"])
    margin_x = W - (roi["left"] + side)
    margin_y = H - (roi["top"]  + side)
    margin = int(round((margin_x + margin_y) / 2))
    ms = normalize_minimap(minimap_scale) or 1.0
    coeff = (side / float(H)) / max(ms, 1e-6)
    return float(coeff), int(margin)

def save_roi_snapshot(roi:Dict[str,int], mon:Dict, out:Path) -> Path:
    with mss.mss() as sct:
        rect = {"left": mon["left"]+roi["left"], "top": mon["top"]+roi["top"], "width": roi["width"], "height": roi["height"]}
        raw = sct.grab(rect)
        img = np.array(raw, dtype=np.uint8)[:, :, :3]
        cv2.imwrite(str(out), img)
    return out

def save_fullscreen_with_box(roi:Dict[str,int], mon:Dict, out:Path) -> Path:
    with mss.mss() as sct:
        rect = {"left": mon["left"], "top": mon["top"], "width": mon["width"], "height": mon["height"]}
        raw = sct.grab(rect)
        img = np.array(raw, dtype=np.uint8)[:, :, :3]
        x1,y1 = roi["left"], roi["top"]
        x2,y2 = x1+roi["width"], y1+roi["height"]
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 3)
        cv2.putText(img, f"{roi}", (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        cv2.imwrite(str(out), img)
    return out

def main():
    ap = argparse.ArgumentParser(description="Compute & persist League minimap ROI (HUD ignored).")
    ap.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to config.yaml")
    ap.add_argument("--monitor", type=int, default=1, help="MSS monitor index (1 = primary)")
    ap.add_argument("--minimap", type=float, default=None, help="Minimap size (e.g., 1.2 for 120%, or 120).")
    ap.add_argument("--left", action="store_true", help="Minimap on left side")
    ap.add_argument("--coeff", type=float, default=None, help="Base coefficient (fraction of H at minimap=1.0).")
    ap.add_argument("--margin", type=int, default=None, help="Edge margin in pixels.")
    ap.add_argument("--snapshot", action="store_true", help="Save tight ROI snapshot (roi_snapshot.png)")
    ap.add_argument("--full-snapshot", action="store_true", help="Save full-screen snapshot with ROI (desktop_with_roi.png)")
    ap.add_argument("--fit-from-current", action="store_true", help="Infer coeff & margin from existing config ROI")
    ap.add_argument("--print-only", action="store_true", help="Print computed ROI without writing config")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)

    # Read in-game settings (minimap + left)
    auto = read_lol_minimap_settings()
    minimap_scale = normalize_minimap(args.minimap) if args.minimap is not None else auto["minimap_scale"]
    left_minimap = bool(args.left or (auto["left_minimap"] if auto["left_minimap"] is not None else False))

    # Resolution from monitor
    W, H, mon = get_desktop_resolution(args.monitor)
    print(f"[info] monitor {args.monitor} resolution: {W}x{H}")
    print(f"[info] Minimap scale={minimap_scale} (HUD ignored), Left minimap={left_minimap}")

    # Choose coeff/margin: CLI > persisted > defaults
    persisted = cfg.get("minimap_params", {}) if isinstance(cfg.get("minimap_params"), dict) else {}
    coeff = args.coeff if args.coeff is not None else float(persisted.get("coeff", DEFAULT_COEFF))
    margin = args.margin if args.margin is not None else int(persisted.get("margin", DEFAULT_MARGIN))

    # Optionally learn from current perfect ROI
    if args.fit_from_current and isinstance(cfg.get("roi"), dict):
        learned_coeff, learned_margin = learn_from_existing_roi(W, H, cfg["roi"], minimap_scale)
        print(f"[fit] learned coeff ~ {learned_coeff:.6f}, margin ~ {learned_margin}px from existing ROI")
        if args.coeff is None: coeff = learned_coeff
        if args.margin is None: margin = learned_margin

    # Compute ROI
    roi = compute_minimap_roi(W, H, minimap_scale, left_minimap, coeff, margin)
    print(f"[info] computed ROI: {roi} (side={roi['width']}px) using coeff={coeff:.6f}, margin={margin}")

    if not args.print_only:
        cfg.setdefault("monitor_index", args.monitor)
        cfg.setdefault("target_fps", 20)
        cfg.setdefault("show_scale", 1.0)
        cfg["roi"] = roi
        cfg["minimap_params"] = {"coeff": float(coeff), "margin": int(margin)}
        save_yaml(cfg_path, cfg)
    else:
        print("[info] print-only: not writing config.yaml")

    if args.snapshot:
        out = save_roi_snapshot(roi, mon, Path("roi_snapshot.png"))
        print(f"[snap] saved tight ROI snapshot -> {out}")
    if args.full_snapshot:
        out = save_fullscreen_with_box(roi, mon, Path("desktop_with_roi.png"))
        print(f"[snap] saved full-screen overlay snapshot -> {out}")

if __name__ == "__main__":
    main()
