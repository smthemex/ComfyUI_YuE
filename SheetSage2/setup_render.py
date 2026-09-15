"""Install the optional renderer and its matching headless browser."""
from pathlib import Path
import argparse
import subprocess
import sys

SCRIPT_DIR = Path(__file__).absolute().parent
sys.path.insert(0, str(SCRIPT_DIR))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-deps", action="store_true", help="Also install Chromium OS libraries (may require sudo).")
    args = parser.parse_args()
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(SCRIPT_DIR / "requirements-render.txt")], check=True)
    command = [sys.executable, "-m", "playwright", "install", "--only-shell", "chromium"]
    if args.with_deps:
        command.append("--with-deps")
    subprocess.run(command, check=True)
    check = """from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=['--disable-gpu'])
    context = browser.new_context(offline=True)
    page = context.new_page()
    assert page.evaluate('typeof OfflineAudioContext') == 'function'
    browser.close()
"""
    result = subprocess.run([sys.executable, "-c", check], capture_output=True, text=True)
    if result.returncode:
        print("Chromium could not start. On Linux, run python setup_render.py --with-deps to install its system libraries.", file=sys.stderr)
        return 1
    print("Rendering is ready. Example: python render.py --input output --audio --score pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
