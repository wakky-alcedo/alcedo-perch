"""v4 elbow chain, variant B: bellcrank 1 sits ON THE HUMERUS (pivot axis = wing-plane normal), driven by a rod from a
body-fixed point A (ball ends: the rod swings out of plane because the shoulder hinge is oblique).
Stage 2 (bellcrank 1 -> rod2 -> ulna lever) is planar in the wing plane: plain pins. Circle fit in the bellcrank frame."""
import json, os, sys
sys.argv = [sys.argv[0], 'elbow'] + sys.argv[1:]
src = open('v4opt.py').read().split("def eval_elbow")[0]
exec(src)


def eval_elbowB(x, detail=False):
    Ax, Ay, Az, Gx, Gy, rb1, Lc1, wx, wy, br = x
    A = np.array([Ax, Ay, Az]); G = np.array([Gx, Gy, 0.0])
    Ah = np.einsum('kji,j->ki', Rh, A)               # body point seen from the humerus
    d = Ah - G; rho = np.hypot(d[:, 0], d[:, 1])
    r = (rb1 ** 2 + np.sum(d * d, 1) - Lc1 ** 2) / (2 * rb1 * np.maximum(rho, 1e-9))
    margin = (1 - np.abs(r)).min()
    if margin < 0.02:
        return 1e3 if not detail else None
    beta = np.arctan2(d[:, 1], d[:, 0]) + (1 if br > 0.5 else -1) * np.arccos(np.clip(r, -1, 1))
    beta = np.unwrap(beta)
    if np.max(np.abs(np.diff(beta))) > 6 * D:
        return 1e3 if not detail else None
    B1 = G + rb1 * np.c_[np.cos(beta), np.sin(beta), np.zeros(N)]
    tang = np.c_[-np.sin(beta), np.cos(beta), np.zeros(N)]
    rod = Ah - B1; rod /= np.linalg.norm(rod, axis=1)[:, None]
    t1 = np.degrees(np.arcsin(np.clip(np.abs(np.sum(rod * tang, 1)), 0, 1))).min()
    E2 = np.array([LH, 0.0])
    U = E2 + np.c_[np.cos(phiE) * wx - np.sin(phiE) * wy, np.sin(phiE) * wx + np.cos(phiE) * wy]
    Fb = np.array([[[np.cos(b), -np.sin(b)], [np.sin(b), np.cos(b)]] for b in beta])
    Ul = np.einsum('kji,kj->ki', Fb, U - G[:2])
    C, L2, res = circ_fit(Ul)
    B2 = G[:2] + np.einsum('kij,j->ki', Fb, C)
    t2 = trans_planar(B2, U, np.repeat(E2[None], N, 0))
    resmax = np.abs(res).max(); arm2 = np.linalg.norm(C); swing = abs(beta[-1] - beta[0]) / D
    pen = 6 * max(0, resmax - 0.05) + 0.4 * max(0, 40 - t1) + 0.4 * max(0, 40 - t2)
    pen += 0.6 * max(0, arm2 - 14) + 0.6 * max(0, L2 - 38) + 0.6 * max(0, 6 - L2) + 0.6 * max(0, rb1 - 14) + 0.6 * max(0, Lc1 - 38)
    pen += 0.2 * max(0, 3 - arm2) + 0.2 * max(0, 3 - rb1) + 1.5 * max(0, 4.5 - np.hypot(wx, wy)) + 0.05 * max(0, swing - 140)
    pen += 2 * max(0, body_sd(A) + 1.5)              # the body anchor must be inside the torso
    hid = 0.0
    z3 = lambda v: np.array([v[0], v[1], 0.0])
    for k in SUB:
        loc = [G, B1[k], z3(B2[k]), z3(U[k])] + seg(G, z3(B2[k]), 2) + seg(B1[k], z3(B2[k]), 2) + seg(z3(B2[k]), z3(U[k]), 6)
        pts = [Rh[k] @ v for v in loc] + seg(A, Rh[k] @ B1[k], 6)
        hid += sum(outside(X, k) for X in pts)
    pen += 0.08 * hid
    score = pen - 0.02 * min(t1, t2)
    if detail:
        return dict(A=[Ax, Ay, Az], G=[Gx, Gy], rb1=rb1, Lc1=Lc1, w=[wx, wy], br=int(br > 0.5), b2=C.tolist(), L2=float(L2), t1=float(t1), t2=float(t2),
                    resmax=float(resmax), swing=float(swing), margin=float(margin), hidden=float(hid), bodyA=float(body_sd(A)))
    return score


if __name__ == '__main__':
    B = [(-20, 4), (-40, 10), (-16, 6), (-6, 24), (-14, 6), (3, 14), (5, 38), (-12, 10), (-12, 6), (0, 1)]
    seeds = [int(s) for s in os.environ.get('SEEDS', '0,1').split(',')]
    out = []
    for seed in seeds:
        r = differential_evolution(eval_elbowB, B, seed=seed, maxiter=int(os.environ.get('MAXIT', '250')), popsize=26, tol=1e-12, polish=False,
                                   mutation=(0.5, 1.0), recombination=0.8)
        d = eval_elbowB(r.x, detail=True)
        out.append(dict(score=float(r.fun), x=r.x.tolist(), d=d))
        print('elbowB seed', seed, 'score %.3f' % r.fun, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in (d or {}).items()}), flush=True)
    json.dump(out, open('v4_elbowB_all.json', 'w'), default=float)
    print('BEST elbowB', min(o['score'] for o in out), flush=True)
