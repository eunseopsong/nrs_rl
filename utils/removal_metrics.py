"""Common entry-excluded ROI and artifact I/O used by both algorithm families."""
import hashlib
import json
from pathlib import Path
import numpy as np

def save(path, data):
    path = Path(path)
    tmp = path.with_suffix('.next.json')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def processing_mask(surface, path):
    mask = surface.roi.copy()
    arc = np.r_[0., np.linalg.norm(np.diff(path, axis=0), axis=1).cumsum()]
    for point in path[arc <= 18.]:
        rr, cc, _, _, w, _ = surface._footprint((point-surface.origin) @ surface.basis.T)
        mask[rr[w > 0], cc[w > 0]] = False
    return mask

