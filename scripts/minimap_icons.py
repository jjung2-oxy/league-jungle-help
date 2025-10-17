#!/usr/bin/env python3
"""
Fetch each champion's **default/base** minimap round portrait from CommunityDragon.

Output: assets/templates/champ_minimap/{alias}_circle_0.png (or a fallback if 0 not present)
"""
import json, time, re
from pathlib import Path
from urllib.parse import quote
import requests

CD_BASE = "https://raw.communitydragon.org"
PATCH   = "latest"   # Pin to a patch like "15.20" if you want frozen assets
OUT_DIR = Path("assets/templates/champ_minimap")
TIMEOUT = 20

SLOW = 0.05  # small delay to be polite

def get(url, ok_404=False):
    r = requests.get(url, timeout=TIMEOUT)
    if ok_404 and r.status_code == 404:
        return None
    r.raise_for_status()
    return r

def list_dir(url):  # HTML directory listing -> scrape file names
    html = get(url).text
    # crude but effective: look for lines like '>{alias}_circle_12.png<'
    return set(re.findall(r'>([^<]+circle[^<]+\.png)<', html, flags=re.I))

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # champion summary = list of champs with 'alias'
    summary_url = f"{CD_BASE}/{PATCH}/plugins/rcp-be-lol-game-data/global/default/v1/champion-summary.json"
    champs = get(summary_url).json()

    ok, miss = 0, 0
    for c in champs:
        alias = (c.get("alias") or "").lower()
        if not alias:
            continue

        hud_dir = f"{CD_BASE}/{PATCH}/game/assets/characters/{quote(alias)}/hud/"
        target  = f"{hud_dir}{alias}_circle_0.png"

        # 1) try base-skin file directly
        r = get(target, ok_404=True)
        if r is not None:
            out = OUT_DIR / f"{alias}_circle_0.png"
            out.write_bytes(r.content)
            print(f"[ok] {alias:<16} -> {out.name}")
            ok += 1
            time.sleep(SLOW)
            continue

        # 2) fallback: older naming without suffix
        r = get(f"{hud_dir}{alias}_circle.png", ok_404=True)
        if r is not None:
            out = OUT_DIR / f"{alias}_circle.png"
            out.write_bytes(r.content)
            print(f"[ok:fallback] {alias:<16} -> {out.name}")
            ok += 1
            time.sleep(SLOW)
            continue

        # 3) last resort: list dir, pick the **lowest** circle_N
        try:
            names = list_dir(hud_dir)
            numeric = sorted(
                (n for n in names if re.search(rf'^{re.escape(alias)}_circle_(\d+)\.png$', n, flags=re.I)),
                key=lambda fn: int(re.search(r'_(\d+)\.png$', fn).group(1))
            )
            if numeric:
                fname = numeric[0]  # lowest available number; 0 is ideal
                r2 = get(f"{hud_dir}{fname}", ok_404=True)
                if r2 is not None:
                    out = OUT_DIR / fname
                    out.write_bytes(r2.content)
                    print(f"[ok:dirpick] {alias:<16} -> {out.name}")
                    ok += 1
                    time.sleep(SLOW)
                    continue
        except Exception as e:
            pass

        print(f"[miss] {alias} (no circle icon found)")
        miss += 1
        time.sleep(SLOW)

    print(f"\nDone. Saved {ok} default icons, {miss} missing.")
    print(f"Output -> {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
