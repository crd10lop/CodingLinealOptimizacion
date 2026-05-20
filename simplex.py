import numpy as np
import pandas as pd


def init_tablero(c, A, b, signos):
    num_restr, num_vars = A.shape
    num_holguras = num_restr
    total = num_vars + num_holguras

    tablero = np.zeros((num_restr + 1, total + 1))
    tablero[:num_restr, :num_vars] = A
    tablero[:num_restr, num_vars:total] = np.eye(num_restr)
    tablero[:num_restr, -1] = b
    tablero[-1, :num_vars] = -c

    columnas = [f"X{i+1}" for i in range(num_vars)] + [f"S{i+1}" for i in range(num_restr)] + ["RHS"]
    base = [f"S{i+1}" for i in range(num_restr)]
    return tablero, columnas, base


def resolver(c, A, b, signos, es_max):
    c = np.asarray(c, dtype=float)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    if any(s != '<=' for s in signos):
        raise ValueError("El metodo Simplex implementado solo soporta restricciones <=. Use Harmony Search para >= o =.")

    c_calc = c.copy() if es_max else -c.copy()
    tablero, columnas, base = init_tablero(c_calc, A, b, signos)

    historial = [{
        'iteracion': 0,
        'tablero': pd.DataFrame(np.round(tablero.copy(), 4), columns=columnas, index=base + ["Z"]),
        'entra': None, 'sale': None, 'pivote': None
    }]

    iteracion = 0
    max_iter = 200
    while iteracion < max_iter:
        if np.all(tablero[-1, :-1] >= -1e-8):
            break

        col_piv = int(np.argmin(tablero[-1, :-1]))
        rhs = tablero[:-1, -1]
        col_val = tablero[:-1, col_piv]

        ratios = np.divide(rhs, col_val, out=np.full_like(rhs, np.inf), where=col_val > 1e-8)
        if np.all(np.isinf(ratios)):
            raise ValueError("Problema no acotado: Z tiende a infinito.")

        fila_piv = int(np.argmin(ratios))
        pivote_val = float(tablero[fila_piv, col_piv])
        entra = columnas[col_piv]
        sale = base[fila_piv]

        base[fila_piv] = entra
        tablero[fila_piv] = tablero[fila_piv] / pivote_val
        for f in range(tablero.shape[0]):
            if f != fila_piv:
                tablero[f] = tablero[f] - tablero[f, col_piv] * tablero[fila_piv]

        iteracion += 1
        historial.append({
            'iteracion': iteracion,
            'tablero': pd.DataFrame(np.round(tablero.copy(), 4), columns=columnas, index=base + ["Z"]),
            'entra': entra, 'sale': sale, 'pivote': pivote_val
        })

    n_vars = len(c)
    x_opt = np.zeros(n_vars)
    for i in range(n_vars):
        col = tablero[:, i]
        es_basica = np.sum(np.isclose(col, 1, atol=1e-6)) == 1 and np.sum(np.isclose(col, 0, atol=1e-6)) == len(col) - 1
        if es_basica:
            idx = int(np.where(np.isclose(col, 1, atol=1e-6))[0][0])
            x_opt[i] = float(tablero[idx, -1])

    z_opt = float(c @ x_opt)

    return {
        'x_opt': x_opt,
        'z_opt': z_opt,
        'iteraciones': iteracion,
        'historial': historial,
        'tablero_final': tablero,
        'columnas': columnas,
        'base_final': base
    }
