"""
MicroSR — BioSR Dataset Downloader
Downloads the BioSR microscopy dataset from Figshare.
Falls back to synthetic data generation if Figshare is blocked.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import zipfile

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# BioSR Figshare file IDs
BIOSR_FILES = {
    "MTs":      "25503987",
    "F-actin":  "25503984",
    "ER":       "25503981",
    "CCPs":     "25503978",
}

BASE_URL  = "https://figshare.com/ndownloader/files/"
HEADERS   = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://figshare.com/",
}


def download_with_requests(url, dest, chunk=1 << 20):
    if not HAS_REQUESTS:
        raise RuntimeError("requests not installed")
    session = requests.Session()
    # First hit the article page to get cookies
    session.get("https://figshare.com/", headers=HEADERS, timeout=15)
    r = session.get(url, headers=HEADERS, stream=True, timeout=60, allow_redirects=True)
    if r.status_code == 202 or len(r.content) < 100:
        raise RuntimeError(f"WAF blocked (HTTP {r.status_code})")
    total = int(r.headers.get("content-length", 0))
    done  = 0
    with open(dest, "wb") as f:
        for block in r.iter_content(chunk):
            f.write(block)
            done += len(block)
            if total:
                pct = done / total * 100
                bar = "█" * int(pct/2) + "░" * (50 - int(pct/2))
                sys.stdout.write(f"\r  [{bar}] {pct:.1f}% ({done/1e6:.1f} MB)")
                sys.stdout.flush()
    print()


def download_structure(structure: str, file_id: str, out_dir: Path):
    url      = BASE_URL + file_id
    zip_path = out_dir / f"{structure}.zip"
    out_path = out_dir / structure

    if out_path.exists() and any(out_path.rglob("*.tif")):
        print(f"  ✓ {structure} already exists — skipping")
        return True

    print(f"\n  Downloading {structure} ...")
    try:
        download_with_requests(url, zip_path)
        # Validate zip
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(out_dir)
        zip_path.unlink()
        print(f"  ✓ {structure} extracted")
        return True
    except Exception as e:
        if zip_path.exists():
            zip_path.unlink()
        print(f"  ✗ Failed: {e}")
        return False


def generate_synthetic(out_dir, structures):
    print("\n[Download] Falling back to synthetic data generation...")
    script = Path(__file__).parent / "generate_synthetic_data.py"
    python  = sys.executable
    cmd = [python, str(script),
           "--out_dir", str(out_dir),
           "--structures"] + structures + [
           "--n_train", "500", "--n_val", "50"]
    subprocess.run(cmd, check=True)


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Download] BioSR dataset → {out_dir}")
    print(f"[Download] Structures: {args.structures}\n")

    failed = []
    for struct in args.structures:
        if struct not in BIOSR_FILES:
            print(f"  Unknown: {struct}"); continue
        ok = download_structure(struct, BIOSR_FILES[struct], out_dir)
        if not ok:
            failed.append(struct)

    if failed:
        print(f"\n⚠  Figshare WAF blocked download for: {failed}")
        print("   Generating synthetic microscopy data instead...")
        generate_synthetic(out_dir, failed)

    print(f"\n[Download] Complete. Dataset at: {out_dir}")
    for struct in args.structures:
        p = out_dir / struct
        if p.exists():
            lr_n = len(list((p/"LR").glob("*.tif"))) if (p/"LR").exists() else 0
            hr_n = len(list((p/"HR").glob("*.tif"))) if (p/"HR").exists() else 0
            print(f"  {struct}/  LR:{lr_n}  HR:{hr_n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--structures", nargs="+", default=["MTs","ER","F-actin","CCPs"],
                        choices=list(BIOSR_FILES.keys()))
    parser.add_argument("--out_dir", default="data/BioSR")
    main(parser.parse_args())
