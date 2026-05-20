import numpy as np


def penalidad(x, A, b, signos):
    total = 0.0
    for i in range(len(b)):
        v = float(A[i] @ x)
        if signos[i] == '<=' and v > b[i]:
            total += v - b[i]
        elif signos[i] == '>=' and v < b[i]:
            total += b[i] - v
        elif signos[i] == '=':
            total += abs(v - b[i])
    return total


def fitness(x, c, A, b, signos, es_max, M=1e6):
    z = float(c @ x)
    pen = penalidad(x, A, b, signos)
    return (z - pen * M) if es_max else -(z + pen * M)


def resolver(c, A, b, signos, es_max, ub, HMS=20, HMCR=0.85, PAR=0.35, BW=0.05, NI=5000, seed=42):
    c = np.asarray(c, dtype=float)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    ub = np.asarray(ub, dtype=float)
    lb = np.zeros(len(c))
    n = len(c)
    M = 1e6

    np.random.seed(seed)

    HM = np.random.uniform(lb, ub, (HMS, n))
    fit = np.array([fitness(HM[k], c, A, b, signos, es_max, M) for k in range(HMS)])

    mejor_fit = float(np.max(fit))
    mejor_sol = HM[int(np.argmax(fit))].copy()

    mejor_factible = None
    mejor_z_factible = -np.inf if es_max else np.inf

    for k in range(HMS):
        if penalidad(HM[k], A, b, signos) < 1e-6:
            z_k = float(c @ HM[k])
            if (es_max and z_k > mejor_z_factible) or (not es_max and z_k < mejor_z_factible):
                mejor_z_factible = z_k
                mejor_factible = HM[k].copy()

    historia_fit = [mejor_fit]
    historia_z = [mejor_z_factible if mejor_factible is not None else None]
    todas = list(HM.copy())

    for it in range(NI):
        nueva = np.empty(n)
        for j in range(n):
            if np.random.rand() < HMCR:
                nueva[j] = HM[np.random.randint(HMS), j]
                if np.random.rand() < PAR:
                    nueva[j] += np.random.uniform(-BW, BW) * (ub[j] - lb[j])
            else:
                nueva[j] = np.random.uniform(lb[j], ub[j])
            nueva[j] = float(np.clip(nueva[j], lb[j], ub[j]))

        fit_nueva = fitness(nueva, c, A, b, signos, es_max, M)
        idx_peor = int(np.argmin(fit))

        if fit_nueva > fit[idx_peor]:
            HM[idx_peor] = nueva.copy()
            fit[idx_peor] = fit_nueva
            if fit_nueva > mejor_fit:
                mejor_fit = fit_nueva
                mejor_sol = nueva.copy()

        if penalidad(nueva, A, b, signos) < 1e-6:
            z_n = float(c @ nueva)
            if (es_max and z_n > mejor_z_factible) or (not es_max and z_n < mejor_z_factible):
                mejor_z_factible = z_n
                mejor_factible = nueva.copy()

        todas.append(nueva.copy())
        historia_fit.append(mejor_fit)
        historia_z.append(mejor_z_factible if mejor_factible is not None else None)

    sol_final = mejor_factible if mejor_factible is not None else mejor_sol
    pen_final = penalidad(sol_final, A, b, signos)
    z_final = float(c @ sol_final)

    verificacion = []
    for i in range(len(b)):
        lhs = float(A[i] @ sol_final)
        ok = (
            (signos[i] == '<=' and lhs <= b[i] + 1e-4) or
            (signos[i] == '>=' and lhs >= b[i] - 1e-4) or
            (signos[i] == '=' and abs(lhs - b[i]) <= 1e-4)
        )
        verificacion.append({'restriccion': f'R{i+1}', 'lhs': lhs, 'signo': signos[i], 'rhs': float(b[i]), 'cumple': ok})

    return {
        'x_opt': sol_final,
        'z_opt': z_final,
        'factible': pen_final < 1e-4,
        'penalidad': pen_final,
        'historia_fit': historia_fit,
        'historia_z': historia_z,
        'memoria_final': HM,
        'todas_evaluaciones': np.array(todas),
        'verificacion': verificacion,
        'parametros': {'HMS': HMS, 'HMCR': HMCR, 'PAR': PAR, 'BW': BW, 'NI': NI, 'seed': seed}
    }
