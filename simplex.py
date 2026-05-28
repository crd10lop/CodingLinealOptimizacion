import numpy as np
import pandas as pd


NOMBRES_VARIABLES = ['X', 'Y', 'Z', 'P', 'Q', 'U', 'V', 'W']
FILA_OBJETIVO = "Z (objetivo)"


def nombre_variable(i):
    if i < len(NOMBRES_VARIABLES):
        return NOMBRES_VARIABLES[i]
    return f"V{i + 1}"


def nombres_variables(n):
    return [nombre_variable(i) for i in range(n)]


def init_tablero(c, A, b, signos, BIG_M=1e6):
    num_restr, num_vars = A.shape
    n_holguras    = sum(1 for s in signos if s == '<=')
    n_excesos     = sum(1 for s in signos if s == '>=')
    n_artificiales = sum(1 for s in signos if s in ('>=', '='))
    total_cols = num_vars + n_holguras + n_excesos + n_artificiales

    tablero = np.zeros((num_restr + 1, total_cols + 1))
    tablero[:num_restr, :num_vars] = A
    tablero[:num_restr, -1] = b

    columnas = [nombre_variable(i) for i in range(num_vars)]
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
            tablero[-1] -= BIG_M * tablero[i]

    return tablero, columnas, base


def resolver(c, A, b, signos, es_max, BIG_M=1e6):
    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    signos = list(signos)

    for i in range(len(b)):
        if b[i] < 0:
            A[i] = -A[i]
            b[i] = -b[i]
            signos[i] = '>=' if signos[i] == '<=' else '<=' if signos[i] == '>=' else '='

    c_calc = c.copy() if es_max else -c.copy()
    tablero, columnas, base = init_tablero(c_calc, A, b, signos, BIG_M)

    base_inicial = list(base)

    historial = [{
        'iteracion': 0,
        'tablero': pd.DataFrame(np.round(tablero.copy(), 4), columns=columnas, index=base + [FILA_OBJETIVO]),
        'entra': None, 'sale': None, 'pivote': None, 'ratios': None
    }]

    iteracion = 0
    max_iter = 300

    while iteracion < max_iter:
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

        ratios_str = ["inf" if np.isinf(r) else f"{r:.4f}" for r in ratios]

        base[fila_piv] = entra
        tablero[fila_piv] = tablero[fila_piv] / pivote_val
        for f in range(tablero.shape[0]):
            if f != fila_piv:
                tablero[f] = tablero[f] - tablero[f, col_piv] * tablero[fila_piv]

        iteracion += 1
        historial.append({
            'iteracion': iteracion,
            'tablero': pd.DataFrame(np.round(tablero.copy(), 4), columns=columnas, index=base + [FILA_OBJETIVO]),
            'entra': entra,
            'sale': sale,
            'pivote': pivote_val,
            'ratios': ratios_str
        })

    for i, var_base in enumerate(base):
        if var_base.startswith('A') and abs(tablero[i, -1]) > 1e-4:
            raise ValueError(
                f"Problema infactible: la variable artificial {var_base} "
                f"permanece en la base con valor {tablero[i, -1]:.4f}."
            )

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

    z_opt = float(c @ x_opt)

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
        'base_inicial': base_inicial,
        'soluciones_multiples': multiples
    }


def analisis_sensibilidad(resultado, c, A, b, signos, es_max):
    tablero = resultado['tablero_final']
    columnas = resultado['columnas']
    base = resultado['base_final']
    base_ini = resultado['base_inicial']

    c = np.asarray(c, dtype=float)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    n_vars = len(c)
    n_restr = A.shape[0]
    idx_var = {nombre_variable(i): i for i in range(n_vars)}

    B_inv = np.zeros((n_restr, n_restr))
    for i, var in enumerate(base_ini):
        if var in columnas:
            col_idx = columnas.index(var)
            B_inv[:, i] = tablero[:-1, col_idx]
        else:
            B_inv[i, i] = 1.0

    x_opt = resultado['x_opt']

    filas_vars = []
    for j in range(n_vars):
        var_name = nombre_variable(j)
        val = x_opt[j]
        costo_red = float(tablero[-1, j])
        coef = float(c[j])

        if var_name not in base:
            if es_max:
                aum = abs(costo_red)
                dis = float('inf')
            else:
                aum = float('inf')
                dis = abs(costo_red)
        else:
            fila_k = base.index(var_name)
            aum = float('inf')
            dis = float('inf')
            for m in range(tablero.shape[1] - 1):
                col_name = columnas[m]
                if col_name in base or col_name.startswith('A'):
                    continue
                a_km = float(tablero[fila_k, m])
                c_red_m = float(tablero[-1, m])
                if a_km > 1e-8:
                    val_r = c_red_m / a_km
                    if val_r < dis:
                        dis = val_r
                elif a_km < -1e-8:
                    val_r = -c_red_m / a_km
                    if val_r < aum:
                        aum = val_r
            if not es_max:
                aum, dis = dis, aum

        filas_vars.append([
            var_name,
            round(float(val), 4),
            round(costo_red if es_max else -costo_red, 4),
            round(coef, 4),
            'Inf' if aum == float('inf') else round(aum, 4),
            'Inf' if dis == float('inf') else round(dis, 4)
        ])

    df_vars = pd.DataFrame(filas_vars, columns=[
        'Variable', 'Valor Final', 'Costo Reducido', 'Coef. Original',
        'Aumento Permisible', 'Disminucion Permisible'
    ])

    c_B = np.array([float(c[idx_var[v]]) if v in idx_var else 0.0 for v in base])
    sombras = c_B @ B_inv
    if not es_max:
        sombras = -sombras

    uso = A @ x_opt
    x_B = tablero[:-1, -1]

    filas_rest = []
    for k in range(n_restr):
        col_k = B_inv[:, k]
        aum_rhs = float('inf')
        dis_rhs = float('inf')
        for i in range(n_restr):
            v = col_k[i]
            xb = x_B[i]
            if v > 1e-8:
                val_r = xb / v
                if val_r < dis_rhs:
                    dis_rhs = val_r
            elif v < -1e-8:
                val_r = -xb / v
                if val_r < aum_rhs:
                    aum_rhs = val_r

        filas_rest.append([
            f'R{k+1}',
            round(float(uso[k]), 4),
            round(float(sombras[k]), 4),
            round(float(b[k]), 4),
            'Inf' if aum_rhs == float('inf') else round(aum_rhs, 4),
            'Inf' if dis_rhs == float('inf') else round(dis_rhs, 4)
        ])

    df_rest = pd.DataFrame(filas_rest, columns=[
        'Restriccion', 'Uso Final', 'Precio Sombra', 'RHS Original',
        'Aumento Permisible', 'Disminucion Permisible'
    ])

    return df_vars, df_rest
