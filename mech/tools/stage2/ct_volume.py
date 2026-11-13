"""Stack the DigiMorph coronal slice frames into a volume (local only) and label the bones.
Volume axes: [z, y, x] = [frame (cranial->caudal, 0.234 mm), row (dorsal->ventral, 0.0959 mm), col (0.0959 mm)]."""
import glob, pathlib, numpy as np
from PIL import Image
from scipy import ndimage as ndi
ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / 'ref' / 'models' / 'digimorph_acristata' / 'cor'
PX, DZ = 0.0959, 0.234
OUT = ROOT / 'results' / 'local' / 'ct_volume.npz'


def build():
    V = np.stack([np.asarray(Image.open(f)) for f in sorted(glob.glob(str(SRC / '*.png')))]).astype(np.uint8)
    V[:, :25, :60] = 0                                   # slice-number overlay
    bone = ndi.median_filter(V, size=(1, 3, 3)) > 140
    bone = ndi.binary_closing(bone, structure=np.ones((3, 3, 3)))
    lab, n = ndi.label(bone, structure=np.ones((3, 3, 3)))
    sizes = ndi.sum(bone, lab, range(1, n + 1))
    np.savez_compressed(OUT, V=V, lab=lab.astype(np.int32), sizes=sizes)
    return V, lab, sizes


if __name__ == '__main__':
    V, lab, sizes = build()
    order = np.argsort(-sizes)[:40]
    sl = ndi.find_objects(lab)
    print('components', len(sizes))
    for i in order:
        s = sl[i]; z0, z1 = s[0].start * DZ, s[0].stop * DZ; y0, y1 = s[1].start * PX, s[1].stop * PX; x0, x1 = s[2].start * PX, s[2].stop * PX
        print('%3d size %7d  z %5.1f-%5.1f  y %5.1f-%5.1f  x %5.1f-%5.1f' % (i + 1, sizes[i], z0, z1, y0, y1, x0, x1))
