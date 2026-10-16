"""Build a source-only archive and verify packaged bytes against disk."""
from pathlib import Path
import hashlib
import json
import os
import tempfile
import zipfile


def build(root):
    root = Path(root).resolve()
    folders = {'references', 'docs', 'examples', 'scripts', 'tests', 'assets', 'agents'}
    excluded = {'data', 'results', '__pycache__', '.git', 'dist', '.venv'}
    suffixes = {'.md', '.py', '.png', '.jpg', '.jpeg', '.svg', '.yaml', '.yml'}
    sources = {}
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.is_symlink():
            continue
        rel = p.relative_to(root)
        if any(x in excluded for x in rel.parts[:-1]):
            continue
        if rel.as_posix() in {'SKILL.md', 'README.md', 'LICENSE'} or (
                rel.parts[0] in folders and p.suffix in suffixes):
            sources[rel.as_posix()] = p.read_bytes()
    if 'SKILL.md' not in sources:
        raise ValueError('SKILL.md missing')
    hashes = {k: hashlib.sha256(v).hexdigest() for k, v in sources.items()}
    dist = root / 'dist'
    dist.mkdir(exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='aivc-', suffix='.zip', dir=dist)
    os.close(fd)
    pending = Path(name)
    try:
        with zipfile.ZipFile(pending, 'w', zipfile.ZIP_DEFLATED) as z:
            for rel, data in sources.items():
                z.writestr('aivc/' + rel, data)
            z.writestr('aivc/package_manifest.json', json.dumps(hashes, indent=2, ensure_ascii=False))
        with zipfile.ZipFile(pending) as z:
            if z.testzip() is not None:
                raise ValueError('archive CRC failed')
            for rel, digest in hashes.items():
                if hashlib.sha256(z.read('aivc/' + rel)).hexdigest() != digest:
                    raise ValueError('archive mismatch: ' + rel)
                if hashlib.sha256((root / rel).read_bytes()).hexdigest() != digest:
                    raise ValueError('source changed during build: ' + rel)
        os.replace(pending, dist / 'aivc.zip')
    finally:
        if pending.exists():
            pending.unlink()
    return len(sources)


if __name__ == '__main__':
    print(f'Packaged and verified {build(Path(__file__).resolve().parents[1])} files.')
