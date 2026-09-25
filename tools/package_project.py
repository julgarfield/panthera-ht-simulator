"""Create a portable source archive, including assets but no installed runtimes."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.venv', 'build', 'datasets', 'validation_runs', '__pycache__', '.pytest_cache', '.git'}


def main():
    target = ROOT.parent/'Panthera-HT-Simulator.zip'
    with ZipFile(target, 'w', compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(ROOT.rglob('*')):
            relative = path.relative_to(ROOT)
            if path.is_file() and not any(part in EXCLUDED for part in relative.parts):
                archive.write(path, Path('panthera_sim')/relative)
    with ZipFile(target) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f'Corrupt archive member: {bad}')
        print(f'Packaged {len(archive.namelist())} files: {target} ({target.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
