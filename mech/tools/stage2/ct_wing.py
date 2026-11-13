"""Measure, on the DigiMorph CT of Alcedo cristata (FMNH 319990, coronal slice movie), how far the folded-wing bones lie below the
outer surface (skin + feathers). Input: mech/ref/models/digimorph_acristata/cor/*.png (frames of CorFlipBodySkel.mp4, local only).
Scale: the 44 mm reconstruction field fits a circle of r = 229.4 px -> 0.0959 mm/px (cross-checked against the sagittal/horizontal
movies within ~4 %). One frame = 2 slices x 0.117 mm = 0.234 mm.
Output: results/local/ct_wing_*.png (CT images, local only) and results/stage2/ct_wing.json (numbers only)."""
import glob, json, pathlib, numpy as np
from PIL import Image
from scipy import ndimage as ndi
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / 'ref' / 'models' / 'digimorph_acristata' / 'cor'
PX, DZ = 0.0959, 0.234
T_TISSUE, T_BONE = 35, 150
files = sorted(glob.glob(str(SRC / '*.png')))
disk = lambda r: (np.add.outer(np.arange(-r, r + 1) ** 2, np.arange(-r, r + 1) ** 2) <= r * r)


def segment(a):
    a = a.copy(); a[:25, :60] = 0
    m = ndi.median_filter(a, 3)
    tissue = ndi.binary_closing(m > T_TISSUE, disk(4))          # bridge the gaps between feather barbs (0.4 mm)
    lab, n = ndi.label(tissue)
    if n == 0: return None
    sz = ndi.sum(tissue, lab, range(1, n + 1)); body = lab == (np.argmax(sz) + 1)
    body = ndi.binary_fill_holes(body)
    core = ndi.binary_opening(ndi.median_filter(a, 5) > 80, disk(6))    # solid muscle/organ mass (feathers are not solid)
    lab2, n2 = ndi.label(core)
    if n2: core = lab2 == (np.argmax(ndi.sum(core, lab2, range(1, n2 + 1))) + 1)
    return a, body, core


rows = []
show = {}
for k, f in enumerate(files):
    a = np.asarray(Image.open(f), float)
    s = segment(a)
    if s is None: continue
    a, body, core = s
    if core.sum() < 2000: continue
    d_out = ndi.distance_transform_edt(body) * PX                   # distance to the outside air
    d_core = ndi.distance_transform_edt(~core) * PX                  # distance to the muscle/organ mass
    cy, cx = np.array(np.where(core)).mean(1)
    bone = a > T_BONE; lab, n = ndi.label(bone)
    for i in range(1, n + 1):
        comp = lab == i; ar = comp.sum()
        if ar < 6 or ar > 400: continue
        filled = ndi.binary_fill_holes(comp)
        if filled.sum() < 1.25 * ar: continue                        # long bones are hollow rings
        y, x = np.array(np.where(filled)).mean(1)
        if y > cy: continue                                          # folded wing bones lie dorsal to the body centre
        if abs(x - cx) * PX < 3 or d_out[int(round(y)), int(round(x))] > 3: continue   # lateral and near the surface
        dia = 2 * np.sqrt(filled.sum() / np.pi) * PX
        # outward direction: from the core centre through the bone
        iy, ix = int(round(y)), int(round(x)); sd = int(np.sign(x - cx))
        def ray(dy, dx):                                              # distance along a direction to the first outside pixel
            for s in range(1, 200):
                yy, xx = int(round(y + dy * s)), int(round(x + dx * s))
                if not (0 <= yy < body.shape[0] and 0 <= xx < body.shape[1]) or not body[yy, xx]: return s * PX
            return np.nan
        rows.append(dict(frame=k, z=k * DZ, side=sd, x=(x - cx) * PX, y=(cy - y) * PX, dia=dia,
                         d_out=float(d_out[iy, ix]), d_core=float(d_core[iy, ix]), up=ray(-1, 0), lat=ray(0, sd),
                         down=ray(1, 0)))
    if k in (180, 200, 220, 241, 260, 280, 300, 320): show[k] = (a, body, core, cx, cy)

out = ROOT / 'results' / 'local'
fig, axs = plt.subplots(2, 4, figsize=(20, 12))
for ax, (k, (a, body, core, cx, cy)) in zip(axs.ravel(), sorted(show.items())):
    ax.imshow(a, cmap='gray', vmax=200)
    ax.contour(body, [0.5], colors='c', linewidths=0.8); ax.contour(core, [0.5], colors='y', linewidths=0.8)
    for r in rows:
        if r['frame'] == k:
            ax.plot(cx + r['x'] / PX, cy - r['y'] / PX, 'r+', ms=10)
            ax.text(cx + r['x'] / PX + 4, cy - r['y'] / PX, f"{r['d_out']:.1f}", color='r', fontsize=9)
    ax.set_title(f'frame {k} (z = {k * DZ:.1f} mm) cyan: outer surface, yellow: muscle mass, red: wing bone (mm to outside)', fontsize=8)
    ax.axis('off')
plt.tight_layout(); plt.savefig(out / 'ct_wing_sections.png', dpi=70); plt.close(fig)

R = rows
json.dump(R, open(out / 'ct_wing_rows.json', 'w'))
z = np.array([r['z'] for r in R]); print('wing-bone sections', len(R), 'z range %.1f-%.1f mm' % (z.min(), z.max()))
for side in (-1, 1):
    S = [r for r in R if r['side'] == side]
    d = np.array([r['d_out'] for r in S]); dc = np.array([r['d_core'] for r in S]); dia = np.array([r['dia'] for r in S])
    print('side', side, 'n', len(S), 'bone centre -> outside: median %.2f  p10 %.2f  p90 %.2f mm' % tuple(np.percentile(d, [50, 10, 90])),
          '| -> muscle mass median %.2f mm | bone dia median %.2f' % (np.median(dc), np.median(dia)))
    for key in ('up', 'lat', 'down'):
        v = np.array([r[key] for r in S]); print('   ray', key, 'median %.2f p10 %.2f p90 %.2f' % tuple(np.nanpercentile(v, [50, 10, 90])))
zz = np.array([r['z'] for r in R]); dd = np.array([r['d_out'] for r in R])
for z0 in np.arange(np.floor(zz.min()), zz.max(), 4):
    s = (zz >= z0) & (zz < z0 + 4)
    if s.any(): print('z %4.0f-%4.0f mm: n %3d, bone->outside median %.2f, max %.2f, lateral ray median %.2f, down ray median %.2f' % (
        z0, z0 + 4, s.sum(), np.median(dd[s]), dd[s].max(), np.nanmedian(np.array([R[i]['lat'] for i in np.where(s)[0]])),
        np.nanmedian(np.array([R[i]['down'] for i in np.where(s)[0]]))))

summary = dict(source='DigiMorph Alcedo cristata FMNH 319990, CorFlipBodySkel.mp4 (fluid-preserved whole specimen, scan 2004)',
               scale_mm_per_px=PX, frame_spacing_mm=DZ, n_sections=len(R),
               bone_centre_to_outside_median_mm=float(np.median(dd)), bone_centre_to_outside_p90_mm=float(np.percentile(dd, 90)),
               lateral_ray_median_mm=float(np.nanmedian([r['lat'] for r in R])), bone_diameter_median_mm=float(np.median([r['dia'] for r in R])),
               note='wet specimen: feathers are flattened, a live bird is thicker (not measured). Scale to A. atthis ~x1.3 (estimate).')
json.dump(summary, open(ROOT / 'results' / 'stage2' / 'ct_wing.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
