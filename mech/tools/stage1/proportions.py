"""Compare body proportions of the two reference models, normalised by wing length.
Common Kingfisher (CG): body frame from ck_align (scaled to wing 78.8 mm).
Belted Kingfisher (scan): body frame from bk_sym (scan units taken as mm; wing length 155-165 mm, Featherbase/AVONET -> 160).
Profiles: for slices along the body axis y, lateral width (x extent) and dorso-ventral depth (z extent) of the feathered outline
(torso + folded wings; head, bill, tail and feet excluded where they can be separated)."""
import sys, json, pathlib, numpy as np, trimesh, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ck_align import transform, aligned_parts, ROOT
from bk_body import bk_bird_mesh

OUT = ROOT / 'results' / 'stage1'


def profile(P, ys, half=1.5):
    w, h, xc = [], [], []
    for y in ys:
        s = P[np.abs(P[:, 1] - y) < half]
        if len(s) < 20:
            w.append(np.nan); h.append(np.nan); xc.append(np.nan); continue
        lo, hi = np.percentile(s[:, 0], [0.5, 99.5]); zl, zh = np.percentile(s[:, 2], [0.5, 99.5])
        w.append(hi - lo); h.append(zh - zl); xc.append((hi + lo) / 2)
    return np.array(w), np.array(h)


if __name__ == '__main__':
    tr = transform(); A = aligned_parts(tr)
    ck = trimesh.util.concatenate([A['torso'], A['wing'], A['wing_other']])
    Pck, _ = trimesh.sample.sample_surface_even(ck, 80000, seed=0)
    hd = A['head']
    # BK
    bk = bk_bird_mesh()
    Pbk, _ = trimesh.sample.sample_surface_even(bk, 120000, seed=0)
    res = {}
    # CK: body length axis range from torso
    yck = np.arange(A['torso'].bounds[0, 1] + 2, A['torso'].bounds[1, 1] - 2, 1.0)
    wck, hck = profile(Pck, yck)
    # BK: find torso y-range: exclude the head (head turned) -> use y range where slice width is "torso-like";
    # simple: cranial limit = y where the neck narrows (min width between torso and head), caudal limit = where tail/wing tips start.
    ybk = np.arange(Pbk[:, 1].min() + 2, Pbk[:, 1].max() - 2, 1.0)
    wbk, hbk = profile(Pbk, ybk)
    np.savez(OUT / 'profiles.npz', yck=yck, wck=wck, hck=hck, ybk=ybk, wbk=wbk, hbk=hbk)
    WL_CK, WL_BK = 78.8, 160.0
    res['CK_max_width_mm'] = float(np.nanmax(wck)); res['CK_max_depth_mm'] = float(np.nanmax(hck))
    res['CK_head_width_mm'] = float(hd.extents[0]); res['CK_torso_len_mm'] = float(A['torso'].extents[1])
    res['CK_width/wing'] = res['CK_max_width_mm'] / WL_CK; res['CK_depth/wing'] = res['CK_max_depth_mm'] / WL_CK
    print(json.dumps(res, indent=1))
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].plot(yck / WL_CK, wck / WL_CK, label='CK width'); ax[0].plot(yck / WL_CK, hck / WL_CK, label='CK depth')
    ax[0].plot(ybk / WL_BK, wbk / WL_BK, label='BK width'); ax[0].plot(ybk / WL_BK, hbk / WL_BK, label='BK depth')
    ax[0].set_xlabel('y / wing length'); ax[0].set_ylabel('extent / wing length'); ax[0].grid(alpha=.3); ax[0].legend()
    ax[1].plot(ybk, wbk, label='BK width (mm)'); ax[1].plot(ybk, hbk, label='BK depth (mm)'); ax[1].grid(alpha=.3); ax[1].legend(); ax[1].set_xlabel('BK y (mm)')
    plt.tight_layout(); plt.savefig(OUT / 'profiles_raw.png', dpi=70)
