"""Stage 3: the kinematic model of one stage-2 candidate turned into rigid parts with local frames, joint features and the
per-pose transforms (41 poses, u = 0 open .. 1 folded). Used by cad.py (solids, checks, drawings, viewer).

Frames (world = shoulder origin, x outward, y forward, z dorsal; all 1:1 kingfisher mm):
- hum   : humerus frame Rh (x along the bone, z = elbow axis), origin = shoulder.
- uln   : ulna frame Ru = Rh rz(pe), origin = elbow E (z = hum z).
- hand  : hand frame Rm = Ru Rw, origin = wrist W (hand plate is normal to the tilted wrist axis).
- bc1   : lever on the humerus, Rh rz(b1), origin = its pivot G1 (stage 3: one joint B where rod 1 (ball, below) and rod 2 (pin, above)
          meet, ELBOW_MERGE=1).
- rod2  : planar rod, x along B2 -> U, z = hum z, origin B2.
- rod1  : bamboo skewer, x along A -> B1 (ball centres), origin A.
- rod4  : wrist rod, x along B4 -> Q (ball centres), z = the humerus z made normal to x, origin B4.
Plates sit at z = level * LAYER_T in their frame (hand: along its plane normal). Run with the same environment as the stage-2
pipeline of the candidate (MIDLINE, SWIVEL, DIHEDRAL, STAGE_TAG, ELBOW_MERGE, STACK_BALL, ...)."""
import sys, json, pathlib, numpy as np
S2 = pathlib.Path(__file__).resolve().parents[1] / 'stage2'
sys.path.insert(0, str(S2))
_argv, sys.argv = sys.argv, sys.argv[:1] + ['45', '2e']
from analyze import make_ev, OUT
from stack import full_check, T, TP, R_SKEW, R_SOCK, R_SOCK_H, Geometry
from problem import R_BONE, R_ROD, R_BOSS, R_PIN, GAP, U
from kin import rz3, wrist_axis
sys.argv = _argv

unit = lambda v: v / np.linalg.norm(v)


def frame_from(p, q, zref):
    x = unit(q - p); z = unit(zref - np.dot(zref, x) * x); y = np.cross(z, x)
    return np.stack([x, y, z], 1), p


class Model:
    def __init__(self, cid, cands=None):
        cands = cands or (OUT / 'candidates.json')
        c = next(c for c in json.load(open(cands, encoding='utf-8')) if c['id'] == cid)
        self.c = c; self.x = np.asarray(c['x'], float)
        self.ev = ev = make_ev(c['ec'], c['wc'], 45, c['dsg'])
        F, G, e = ev(self.x[None], extra=True)
        self.F, self.G = F[0], G[0]
        f, d = e['f'], e['d']; self.f, self.d = f, d
        v, plan, studs = full_check(ev, self.x)
        assert v == 0, f'{cid}: no layer plan ({v} parts unplaced)'
        self.plan, self.studs = plan, studs
        N = len(U); self.N = N
        g1 = lambda a: np.asarray(a)[0]
        self.S = np.zeros((N, 3)); self.E, self.W, self.T = g1(f['E_w']), g1(f['W_w']), g1(f['T_w'])
        self.ep = {k: g1(v) for k, v in f['elbow_pts'].items()}; self.wp = {k: g1(v) for k, v in f['wrist_pts'].items()}
        self.Rh, self.Ru, self.Rm, self.b1 = g1(f['Rh']), g1(f['Ru']), g1(f['Rm']), g1(f['b1'])
        self.axis = g1(d['axis']); self.tw = float(g1(d['tw'])) if d.get('tilt') else 0.0
        # wrist axis (world) and hand plate normal (hand frame, constant)
        a_u = wrist_axis(np.array(self.tw)); self.w_world = np.einsum('kij,j->ki', self.Ru, a_u)
        n = self.Rm[0].T @ self.w_world[0]
        if np.dot(self.Rm[0] @ n, self.Rh[0][:, 2]) < 0: n = -n
        self.n_hand = n
        # parts as capsules at every pose (world), from the stage-2 evaluator
        P_, adj, _ = ev.parts(f, k=np.arange(N))
        self.caps = {nm: [(g1(p), g1(q), float(np.asarray(r).ravel()[0]) if np.ndim(r) else float(r)) for p, q, r in lst] for nm, lst in P_.items()}
        self.names = list(P_)

    # ------------------------------------------------------------ frames
    def frame(self, nm, k):
        ep, wp, Rh = self.ep, self.wp, self.Rh
        if nm == 'hum': return Rh[k], np.zeros(3)
        if nm == 'uln': return self.Ru[k], self.E[k]
        if nm == 'hand': return self.Rm[k], self.W[k]
        if nm == 'bc1': return Rh[k] @ rz3(np.array(self.b1[k])), ep['G1'][k]
        if nm == 'rod2': return frame_from(ep['B2'][k], ep['U'][k], Rh[k][:, 2])
        if nm == 'rod1': return frame_from(ep['A'][k], ep['B1'][k], Rh[k][:, 2])
        if nm == 'rod4': return frame_from(wp['B4'][k], wp['Q'][k], Rh[k][:, 2])
        if nm == 'body': return np.eye(3), np.zeros(3)
        raise KeyError(nm)

    def to_local(self, nm, X, k=0):
        R, t = self.frame(nm, k); return R.T @ (np.asarray(X) - t)

    def to_world(self, nm, Xl, k):
        R, t = self.frame(nm, k); return R @ np.asarray(Xl) + t

    def M(self, nm, k):
        R, t = self.frame(nm, k); m = np.eye(4); m[:3, :3] = R; m[:3, 3] = t; return m

    def level_z(self, nm):
        return self.plan.get(nm, 0) * T

    # ------------------------------------------------------------ joints
    def joints(self):
        """(name, part a, part b, world points (N,3), kind) with kind 'pin' | 'ball'; plus the body anchor A of rod 1"""
        ep, wp = self.ep, self.wp
        J = [('E', 'hum', 'uln', self.E, 'pin'), ('W', 'uln', 'hand', self.W, 'pin'),
             ('G1', 'bc1', 'hum', ep['G1'], 'pin'), ('B1', 'bc1', 'rod1', ep['B1'], 'ball'), ('B2', 'bc1', 'rod2', ep['B2'], 'pin'),
             ('U', 'rod2', 'uln', ep['U'], 'pin'), ('B4', 'rod4', 'hum', wp['B4'], 'ball'), ('Q', 'rod4', 'hand', wp['Q'], 'ball'),
             ('A', 'body', 'rod1', ep['A'], 'ball')]
        return J

    def consistency(self):
        """max over poses of the distance between a joint seen from its two parts (local at pose 0 -> world at pose k)"""
        out = {}
        for nm, a, b, X, kind in self.joints():
            la, lb = self.to_local(a, X[0]), self.to_local(b, X[0])
            out[nm] = max(float(np.linalg.norm(self.to_world(a, la, k) - self.to_world(b, lb, k))) for k in range(self.N))
        return out


if __name__ == '__main__':
    m = Model(sys.argv[1])
    print('plan', m.plan, 'studs', m.studs)
    print('F', np.round(m.F, 2), 'feasible', bool(np.all(m.G <= 0)))
    print('joint consistency (mm)', {k: round(v, 4) for k, v in m.consistency().items()})
    print('B1-B2 (mm)', float(np.linalg.norm(m.ep['B1'][0] - m.ep['B2'][0])))
    print('wrist tilt deg', np.degrees(m.tw), 'hand normal (hand frame)', np.round(m.n_hand, 3))
