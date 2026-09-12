"""Install the hash-pinned official Linux binary distribution locally."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time
import urllib.request

root = Path(__file__).resolve().parents[1]
config = json.loads((root / "benchmark.json").read_text())
vendor = root / "vendor"
vendor.mkdir(exist_ok=True)
archive = vendor / "mp-spdz.tar.xz"
started = time.perf_counter()
urllib.request.urlretrieve(config["archive_url"], archive)
digest = hashlib.file_digest(archive.open("rb"), "sha256").hexdigest()
if digest != config["archive_sha256"]:
    raise SystemExit("Official release archive SHA-256 mismatch")
with tarfile.open(archive) as package:
    package.extractall(vendor, filter="data")
roots = [p.parent for p in vendor.glob("*/compile.py")]
if len(roots) != 1:
    raise SystemExit("Expected exactly one MP-SPDZ distribution root")
dist = roots[0]
subprocess.run(["bash", "Scripts/tldr.sh"], cwd=dist, check=True)
subprocess.run(["bash", "Scripts/setup-ssl.sh", "2"], cwd=dist, check=True)
subprocess.run(["ldd", "semi2k-party.x"], cwd=dist, check=True)
(vendor / "install.json").write_text(json.dumps({
    "distribution": dist.name,
    "release": config["release"],
    "commit": config["commit"],
    "archive_sha256": digest,
    "download_extract_binary_and_tls_setup_seconds": time.perf_counter() - started,
}, indent=2) + "\n")
print("Installed verified MP-SPDZ release:", dist.name)
