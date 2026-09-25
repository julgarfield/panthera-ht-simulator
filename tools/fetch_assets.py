"""Restore exact official files from their pinned commit. No repository checkout needed."""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true', help='Verify bundled assets without downloading')
    args = parser.parse_args()
    folder = ROOT/'assets/panthera'
    manifest = json.loads((folder/'provenance.json').read_text(encoding='utf-8'))
    repository = manifest['repository'].removeprefix('https://github.com/')
    for item in manifest['files']:
        target = folder/item['path']
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == item['sha256']:
            continue
        if args.verify:
            raise ValueError(f'Missing or modified official file: {target}')
        url = f"https://raw.githubusercontent.com/{repository}/{manifest['commit']}/{item['upstream_path']}"
        request = urllib.request.Request(url, headers={'User-Agent': 'Panthera-local-simulator/1.0'})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError(f'Official asset checksum mismatch: {url}')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        print(f'Downloaded {item["path"]}')
    print(f'Verified {len(manifest["files"])} official assets at {manifest["commit"]}')


if __name__ == '__main__':
    main()
