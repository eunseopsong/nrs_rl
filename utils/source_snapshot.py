"""Archive executable task sources with relative paths and hashes for new runs."""
import hashlib
import json
from pathlib import Path
import shutil
from ..paths import ROOT


def snapshot_sources(output):
    output = Path(output)
    package = ROOT / 'source/nrs_rl/nrs_rl'
    snapshot = output / 'package_source_snapshot'
    snapshot.mkdir(exist_ok=False)
    manifest = {}
    for source in sorted(package.rglob('*')):
        if not source.is_file() or source.suffix not in {'.py', '.yaml', '.cpp', '.hpp', '.toml'}:
            continue
        relative = source.relative_to(ROOT)
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        manifest[str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()
    (snapshot / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest
