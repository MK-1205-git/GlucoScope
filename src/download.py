"""Download the exact UCI dataset used by this investigation."""
from pathlib import Path
import hashlib
import urllib.request

URL = 'https://archive.ics.uci.edu/static/public/891/data.csv'
ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA256 = '9f71fda9d4ae5f4878c99b9233b6a16accfa9a17c194116a6b78100540934964'

def download():
    target = ROOT / 'data/source.csv'
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        with urllib.request.urlopen(URL, timeout=120) as response:
            target.write_bytes(response.read())
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError('Dataset checksum differs from the researched version. Audit the change before training.')
    print('Dataset SHA-256:', digest)
    return target

if __name__ == '__main__':
    download()
