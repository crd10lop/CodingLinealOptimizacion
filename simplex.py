import numpy as np
import pandas as pd


def init_tablero(c, A, b, signos, BIG_M=1e6):
    num_restr, num_vars = A.shape
    n_holguras    = sum(1 for s in signos if s == '<=')
    n_excesos     = sum(1 for s in signos if s == '>=')
    n_artificiales = sum(1 for s in signos if s in ('>=', '='))
    total_cols = num_vars + n_holguras + n_excesos + n_artificiales

    tablero = np.zeros((num_restr + 1, total_cols + 1))
    tablero[:num_restr, :num_vars] = A
    tablero[:num_restr, -1] = b

    columnas = [f"X{i+1}" for i in range(num_vars)]
    base = [None] * num_restr

    col = num_vars
    cont_s, cont_e, cont_a = 1, 1, 1

    for i, signo in enumerate(signos):
        if signo == '<=':
            tablero[i, col] = 1.0
            columnas.append(f"S{cont_s}")
            base[i] = f"S{cont_s}"
            cont_s += 1
            col += 1

        elif signo == '>=':
            tablero[i, col] = -1.0
            columnas.append(f"E{cont_e}")
            cont_e += 1
            col += 1

            tablero[i, col] = 1.0
            columnas.append(f"A{cont_a}")
            base[i] = f"A{cont_a}"
            tablero[-1, col] = BIG_M
            cont_a += 1
            col += 1
        
        elif signo == '=':
            tablero[i, col] = 1.0
            columnas.append(f"A{cont_a}")
            base[i] = f"A{cont_a}"
            tablero[-1, col] = BIG_M
            cont_a += 1
            col += 1

    columnas.append("RHS")
    tablero[-1, :num_vars] = -c
    for i, signo in enumerate(signos):
        if signo in ('>=', '='):
            # La fila Z tiene +M en la col de la artificial; hay que restarle M * fila_i
            idx_art = columnas.index(base[i])
            tablero[-1] -= BIG_M * tablero[i]

    return tablero, columnas, base


def resolver(c, A, b, signos, es_max, BIG_M=1e6):
    c = np.asarray(c, dtype=float)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    # Normalizar filas con RHS negativo (multiplicar por -1 e invertir signo)
    for i in range(len(b)):
        if b[i] < 0:
            A[i] = -A[i]
            b[i] = -b[i]
            signos[i] = '>=' if signos[i] == '<=' else '<=' if signos[i] == '>=' else '='

    c_calc = c.copy() if es_max else -c.copy()
    tablero, columnas, base = init_tablero(c_calc, A, b, signos, BIG_M)

    n_cols_decision = len(c)
    nombres_artificiales = [col for col in columnas if col.startswith('A')]

    historial = [{
        'iteracion': 0,
        'tablero': pd.DataFrame(np.round(tablero.copy(), 4), columns=columnas, index=base + ["Z"]),
        'entra': None, 'sale': None, 'pivote': None, 'ratios': None
    }]

    iteracion = 0
    max_iter = 300

    while iteracion < max_iter:
        # Criterio de optimalidad: todos los coefs de Z >= 0
        if np.all(tablero[-1, :-1] >= -1e-8):
            break

        col_piv = int(np.argmin(tablero[-1, :-1]))
        rhs = tablero[:-1, -1]
        col_val = tablero[:-1, col_piv]

        ratios = np.divide(
            rhs, col_val,
            out=np.full_like(rhs, np.inf),
            where=col_val > 1e-8
        )

        if np.all(np.isinf(ratios)):
            raise ValueError("Problema no acotado: Z tiende a infinito.")

        fila_piv = int(np.argmin(ratios))
        pivote_val = float(tablero[fila_piv, col_piv])
        entra = columnas[col_piv]
        sale = base[fila_piv]

        # Guardar ratios legibles para mostrar en la tabla
        ratios_str = []
        for r in ratios:
            ratios_str.append("∞" if np.isinf(r) else f"{r:.4f}")

        base[fila_piv] = entra
        tablero[fila_piv] = tablero[fila_piv] / pivote_val
        for f in range(tablero.shape[0]):
            if f != fila_piv:
                tablero[f] = tablero[f] - tablero[f, col_piv] * tablero[fila_piv]

        iteracion += 1
        historial.append({
            'iteracion': iteracion,
            'tablero': pd.DataFrame(np.round(tablero.copy(), 4), columns=columnas, index=base + ["Z"]),
            'entra': entra,
            'sale': sale,
            'pivote': pivote_val,
            'col_piv': col_piv,
            'fila_piv': fila_piv,
            'ratios': ratios_str
        })

    # Detectar infactibilidad 
    for i, var_base in enumerate(base):
        if var_base.startswith('A') and abs(tablero[i, -1]) > 1e-4:
            raise ValueError(
                f"Problema infactible: la variable artificial {var_base} "
                f"permanece en la base con valor {tablero[i, -1]:.4f}."
            )

    # Extraer solucion optima
    n_vars = len(c)
    x_opt = np.zeros(n_vars)
    for i in range(n_vars):
        col_data = tablero[:-1, i]
        es_basica = (
            np.sum(np.isclose(col_data, 1, atol=1e-6)) == 1 and
            np.sum(np.isclose(col_data, 0, atol=1e-6)) == len(col_data) - 1
        )
        if es_basica:
            idx = int(np.where(np.isclose(col_data, 1, atol=1e-6))[0][0])
            x_opt[i] = float(tablero[idx, -1])

    # Z real siempre calculado con el c original (no negado)
    z_opt = float(c @ x_opt)

    # Detectar soluciones múltiples
    variables_no_basicas = [columnas[j] for j in range(n_vars) if columnas[j] not in base]
    multiples = any(
        abs(tablero[-1, columnas.index(v)]) < 1e-8
        for v in variables_no_basicas
        if not v.startswith('A')
    )

    return {
        'x_opt': x_opt,
        'z_opt': z_opt,
        'iteraciones': iteracion,
        'historial': historial,
        'tablero_final': tablero,
        'columnas': columnas,
        'base_final': base,
        'soluciones_multiples': multiples
    }
