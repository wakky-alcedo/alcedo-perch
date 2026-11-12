import json, sys, numpy as np, os
exec(open('synth.py').read().split('# ---------- loop 1 ----------')[0])
import importlib.util
spec = importlib.util.spec_from_file_location('f', 'fold6bar.py'); F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
F.AH = a; F.N = N

def tr_ang(Pa, Pb, Pc):
    t = trans(Pa, Pb, Pc); return min(np.nanmin(t), 180 - np.nanmax(t))

def loop1():
    out = []
    for s in np.linspace(-15, 36, 103):
        for n in np.linspace(-16, 16, 65):
            w = np.array([s, n]); Up = Epts + np.einsum('kij,j->ki', R(au), w)
            C, r, res = circfit(Up)
            if C[0] > 8 or C[0] < -32 or abs(C[1]) > 40 or r > 80 or r < 4: continue
            tm = tr_ang(Epts, Up, np.repeat(C[None], N, 0))
            out.append((np.sqrt(np.mean(res**2)), np.max(np.abs(res)), tm, s, n, C[0], C[1], r))
    return np.array(out)

def loop2():
    rel_u = au - a; rel_m = am - a; E_loc = np.array([LH, 0.0])
    W_loc = E_loc + LU * np.c_[np.cos(rel_u), np.sin(rel_u)]
    out = []
    for qx in np.linspace(-16, 16, 81):
        for qy in np.linspace(-16, 16, 81):
            q = np.array([qx, qy]); Qp = W_loc + np.einsum('kij,j->ki', R(rel_m), q)
            C, r, res = circfit(Qp); p = C - E_loc
            if np.hypot(*p) > 28 or r < 5 or r > 60: continue
            tm = tr_ang(W_loc, Qp, np.repeat(C[None], N, 0))
            out.append((np.sqrt(np.mean(res**2)), np.max(np.abs(res)), tm, qx, qy, p[0], p[1], r))
    return np.array(out)

c1, c2 = loop1(), loop2()
for name, c in (('LOOP1', c1), ('LOOP2', c2)):
    print(name, 'pareto (max residual mm vs min transmission deg):')
    for treq in (25, 30, 35, 40, 45, 50):
        m = c[c[:, 2] >= treq]
        if len(m) == 0: print('  t>=%d none' % treq); continue
        i = np.argmin(m[:, 1]); print('  t>=%d' % treq, np.round(m[i], 2))
np.save('c1.npy', c1); np.save('c2.npy', c2)
