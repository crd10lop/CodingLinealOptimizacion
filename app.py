import time
import numpy as np
import pandas as pd
import streamlit as st

import simplex
import harmony_search
import graficos


def _centrar(styler):
    return styler.set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center'), ('vertical-align', 'middle')]},
        {'selector': 'td', 'props': [('text-align', 'center')]},
    ])


def tabla(df, fmt=None):
    sty = df.style
    if fmt is not None:
        sty = sty.format(fmt)
    sty = sty.hide(axis='index')
    return _centrar(sty)


def tabla_tablero(df, entra=None, sale=None):
    def colorear(_):
        s = pd.DataFrame('', index=df.index, columns=df.columns)
        if entra is not None and entra in df.columns:
            s.loc[:, entra] = 'background-color: #bee3f8'
        if sale is not None and sale in df.index:
            s.loc[sale, :] = 'background-color: #fed7d7'
        if (entra is not None and sale is not None
                and entra in df.columns and sale in df.index):
            s.loc[sale, entra] = 'background-color: #f6e05e; font-weight: 700'
        return s
    sty = df.style.apply(colorear, axis=None).format('{:.4f}')
    return _centrar(sty)


st.set_page_config(
    page_title="Optimizacion Lineal",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { color: #1f3a5f; }
    h2 { color: #2c5282; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.3rem; }
    h3 { color: #2d3748; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 0.5rem 1.2rem; }
    [data-testid="stTable"] { overflow-x: auto; }
    [data-testid="stTable"] table { margin-left: auto; margin-right: auto; }
    [data-testid="stTable"] th, [data-testid="stTable"] td { text-align: center !important; }
</style>
""", unsafe_allow_html=True)

st.title("Optimizacion Lineal: Simplex y Harmony Search")
st.caption("Programacion Lineal | Ingenieria de Sistemas")

with st.sidebar:
    st.header("Configuracion del Problema")

    tipo_opt = st.radio("Tipo de optimizacion", ["Maximizar", "Minimizar"], horizontal=True)
    es_max = tipo_opt == "Maximizar"

    col_n1, col_n2 = st.columns(2)
    with col_n1:
        n_vars = st.number_input("Variables", min_value=1, max_value=8, value=2, step=1)
    with col_n2:
        n_rest = st.number_input("Restricciones", min_value=1, max_value=10, value=3, step=1)

    st.divider()
    st.subheader("Funcion Objetivo")
    nombres_side = simplex.nombres_variables(n_vars)
    st.caption("Z = " + " + ".join([f"c{i+1}·{nombres_side[i]}" for i in range(n_vars)]))

    cols_c = st.columns(n_vars)
    c_vals = []
    defaults_c = [3.0, 5.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for i in range(n_vars):
        with cols_c[i]:
            c_vals.append(st.number_input(f"c{i+1}", value=defaults_c[i], step=1.0, key=f"c_{i}"))

    st.divider()
    st.subheader("Restricciones")
    st.caption("Soporta <=, >= y = (Gran M)")

    defaults_A = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    ]
    defaults_b = [4.0, 12.0, 18.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

    A_vals = []
    signos = []
    b_vals = []

    for i in range(n_rest):
        st.markdown(f"**R{i+1}**")
        cols = st.columns(n_vars + 2)
        fila = []
        for j in range(n_vars):
            with cols[j]:
                default_val = defaults_A[i][j] if i < len(defaults_A) and j < len(defaults_A[i]) else 0.0
                fila.append(st.number_input(
                    f"a{i+1},{j+1}", value=default_val, step=1.0,
                    key=f"a_{i}_{j}", label_visibility="collapsed"
                ))
        with cols[n_vars]:
            signos.append(st.selectbox(
                "sig", ["<=", ">=", "="], key=f"s_{i}", label_visibility="collapsed"
            ))
        with cols[n_vars + 1]:
            b_def = defaults_b[i] if i < len(defaults_b) else 0.0
            b_vals.append(st.number_input(
                f"b{i+1}", value=b_def, step=1.0,
                key=f"b_{i}", label_visibility="collapsed"
            ))
        A_vals.append(fila)

    st.divider()
    st.subheader("Metodos a Ejecutar")
    usar_simplex = st.checkbox("Metodo Simplex (Gran M)", value=True)
    usar_hs = st.checkbox("Harmony Search", value=True)

    if usar_hs:
        st.subheader("Parametros Harmony Search")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            HMS  = st.slider("HMS (tamaño memoria)", 5, 100, 20,
                             help="Cantidad de armonias guardadas en memoria")
            HMCR = st.slider("HMCR", 0.50, 0.99, 0.85, step=0.01,
                             help="Probabilidad de tomar valor de la memoria")
            BW   = st.slider("BW (ancho banda)", 0.001, 0.5, 0.05, step=0.005,
                             help="Magnitud del ajuste como fraccion del rango")
        with col_p2:
            PAR  = st.slider("PAR", 0.01, 0.99, 0.35, step=0.01,
                             help="Probabilidad de ajuste de tono")
            NI   = st.slider("Iteraciones", 100, 20000, 5000, step=100)
            seed = st.number_input("Semilla", value=42, step=1)

        st.subheader("Limites Superiores de Busqueda")
        cols_ub = st.columns(n_vars)
        ub_vals = []
        for i in range(n_vars):
            with cols_ub[i]:
                ub_vals.append(st.number_input(f"ub {nombres_side[i]}", value=20.0, step=1.0, key=f"ub_{i}"))
    else:
        HMS, HMCR, PAR, BW, NI, seed = 20, 0.85, 0.35, 0.05, 5000, 42
        ub_vals = [20.0] * n_vars

    st.divider()
    boton_resolver = st.button("Resolver", type="primary", use_container_width=True)


c_arr  = np.array(c_vals, dtype=float)
A_arr  = np.array(A_vals, dtype=float)
b_arr  = np.array(b_vals, dtype=float)
ub_arr = np.array(ub_vals, dtype=float)


def resolver_problema():
    res = {}

    if usar_simplex:
        try:
            t0 = time.perf_counter()
            res['simplex'] = simplex.resolver(
                c_arr.copy(), A_arr.copy(), b_arr.copy(), list(signos), es_max
            )
            res['simplex']['tiempo'] = time.perf_counter() - t0

            try:
                df_v, df_r = simplex.analisis_sensibilidad(
                    res['simplex'], c_arr.copy(), A_arr.copy(), b_arr.copy(), list(signos), es_max
                )
                res['simplex']['sens_vars'] = df_v
                res['simplex']['sens_rest'] = df_r
            except Exception:
                res['simplex']['sens_vars'] = None
                res['simplex']['sens_rest'] = None

            res['simplex_error'] = None
        except Exception as ex:
            res['simplex'] = None
            res['simplex_error'] = str(ex)

    if usar_hs:
        try:
            t0 = time.perf_counter()
            res['hs'] = harmony_search.resolver(
                c_arr.copy(), A_arr.copy(), b_arr.copy(), list(signos), es_max, ub_arr.copy(),
                HMS=HMS, HMCR=HMCR, PAR=PAR, BW=BW, NI=NI, seed=int(seed)
            )
            res['hs']['tiempo'] = time.perf_counter() - t0
            res['hs_error'] = None
        except Exception as ex:
            res['hs'] = None
            res['hs_error'] = str(ex)

    return res


if boton_resolver:
    with st.spinner("Calculando..."):
        st.session_state.resultados = resolver_problema()
        st.session_state.config = {
            'n_vars': int(n_vars), 'n_rest': int(n_rest), 'es_max': es_max,
            'c': c_arr.copy(), 'A': A_arr.copy(), 'b': b_arr.copy(),
            'signos': list(signos), 'usar_simplex': usar_simplex, 'usar_hs': usar_hs
        }


tab_problema, tab_simplex, tab_hs, tab_comp = st.tabs(
    ["Problema", "Simplex (Gran M)", "Harmony Search", "Comparacion"]
)


with tab_problema:
    st.header("Planteamiento del problema")

    accion = "Maximizar" if es_max else "Minimizar"
    nombres_prob = simplex.nombres_variables(n_vars)
    st.markdown(f"**{accion}** &nbsp; Z = {graficos.expresion_objetivo(c_arr)}")

    st.markdown("**Sujeto a:**")
    for i in range(n_rest):
        partes = []
        for j in range(n_vars):
            coef = A_vals[i][j]
            if coef != 0:
                sep = " + " if coef >= 0 and partes else (" - " if coef < 0 and partes else ("" if coef >= 0 else "-"))
                coef_abs = abs(coef)
                coef_str = "" if coef_abs == 1 else f"{coef_abs:g}·"
                partes.append(f"{sep}{coef_str}{nombres_prob[j]}")
        if not partes:
            partes = ["0"]
        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{''.join(partes)} {signos[i]} {b_vals[i]:g}")
    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{', '.join(nombres_prob)} >= 0")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Metodo Simplex (Gran M)")
        st.markdown("""
        Algoritmo exacto basado en el recorrido de los vertices del poliedro factible.
        Garantiza la solucion optima cuando existe. Soporta restricciones `<=`, `>=` y `=`
        mediante el metodo de la Gran M (variables artificiales). Incluye deteccion de
        soluciones multiples y analisis de sensibilidad.
        """)
    with col_b:
        st.subheader("Harmony Search")
        st.markdown("""
        Metaheuristica inspirada en la improvisacion musical. No garantiza el optimo
        global pero se adapta a cualquier tipo de restriccion mediante penalizacion.
        Util como referencia comparativa y para verificar la solucion del Simplex.
        """)


with tab_simplex:
    st.header("Metodo Simplex (Gran M)")

    if not st.session_state.get('resultados'):
        st.info("Configure el problema en el panel lateral y presione **Resolver**.")
    elif not st.session_state.config.get('usar_simplex'):
        st.warning("El metodo Simplex no esta seleccionado.")
    elif st.session_state.resultados.get('simplex_error'):
        st.error(f"No se pudo resolver con Simplex: {st.session_state.resultados['simplex_error']}")
        cfg = st.session_state.config
        if cfg['n_vars'] == 2:
            st.subheader("Region factible 2D (sin optimo)")
            try:
                fig2d = graficos.grafico_region_2d(cfg['c'], cfg['A'], cfg['b'], cfg['signos'])
                st.plotly_chart(fig2d, use_container_width=True)
            except Exception:
                st.warning("Tampoco se pudo generar la grafica 2D para este problema.")
        st.info("No se puede generar la grafica 3D para este caso.")
    else:
        res_s = st.session_state.resultados['simplex']
        cfg   = st.session_state.config

        col1, col2, col3 = st.columns(3)
        col1.metric("Z optima (maximo)" if cfg['es_max'] else "Z optima (minimo)", f"{res_s['z_opt']:.4f}")
        col2.metric("Iteraciones", res_s['iteraciones'])
        col3.metric("Tiempo", f"{res_s['tiempo']*1000:.2f} ms")

        if res_s.get('soluciones_multiples'):
            st.warning(
                "Se detectaron multiples soluciones optimas. El problema tiene infinitas "
                "combinaciones que producen el mismo valor Z. La solucion mostrada es un "
                "vertice optimo; existen otros vertices con el mismo Z."
            )

        nombres_s = simplex.nombres_variables(cfg['n_vars'])

        extremos = graficos.extremos_objetivo(cfg['c'], cfg['A'], cfg['b'], cfg['signos'])
        if extremos is not None:
            st.subheader("Valor maximo y minimo en la region factible")
            punto_max = ", ".join(f"{nombres_s[i]}={extremos['x_max'][i]:.2f}" for i in range(cfg['n_vars']))
            punto_min = ", ".join(f"{nombres_s[i]}={extremos['x_min'][i]:.2f}" for i in range(cfg['n_vars']))
            cmax, cmin = st.columns(2)
            cmax.metric("Z maximo", f"{extremos['z_max']:.4f}")
            cmax.caption(f"en ({punto_max})")
            cmin.metric("Z minimo", f"{extremos['z_min']:.4f}")
            cmin.caption(f"en ({punto_min})")

        df_x = pd.DataFrame({
            'Variable': nombres_s,
            'Valor Optimo': np.round(res_s['x_opt'], 4)
        })
        st.subheader("Valores optimos")
        st.table(tabla(df_x))

        st.subheader("Evolucion tabular")
        st.caption(
            "Columna azul: variable que entra · Fila roja: variable que sale · "
            "Celda amarilla: elemento pivote."
        )
        historial = res_s['historial']
        for idx, paso in enumerate(historial):
            entra_col = None
            sale_row = None
            siguiente = historial[idx + 1] if idx + 1 < len(historial) else None

            if paso['iteracion'] == 0:
                st.markdown("**Tablero inicial**")
            else:
                st.markdown(f"**Iteracion {paso['iteracion']}**")

            if siguiente is not None:
                entra_col = siguiente['entra']
                sale_row = siguiente['sale']
                st.caption(
                    f"Entra: {entra_col}  ·  Sale: {sale_row}  ·  "
                    f"Pivote: {siguiente['pivote']:.4f}"
                )
                if siguiente.get('ratios'):
                    with st.expander("Ver razones (regla de la razon minima)"):
                        df_ratios = pd.DataFrame({
                            'Variable base': paso['tablero'].index[:-1].tolist(),
                            'Razon b/a': siguiente['ratios']
                        })
                        st.table(tabla(df_ratios))
            else:
                st.caption("Tablero optimo: no hay coeficientes negativos en la fila Z.")

            st.table(tabla_tablero(paso['tablero'], entra_col, sale_row))

        if res_s.get('sens_vars') is not None:
            st.subheader("Analisis de Sensibilidad")
            st.caption("Rangos dentro de los cuales los parametros pueden variar sin cambiar la base optima.")

            col_sv, col_sr = st.columns(2)
            with col_sv:
                st.markdown("**Reporte de Variables (Coeficientes de Z)**")
                st.table(tabla(res_s['sens_vars']))
            with col_sr:
                st.markdown("**Reporte de Restricciones (RHS)**")
                st.table(tabla(res_s['sens_rest']))

        if cfg['n_vars'] == 2:
            st.subheader("Region factible 2D")
            try:
                fig2d = graficos.grafico_region_2d(
                    cfg['c'], cfg['A'], cfg['b'], cfg['signos'],
                    res_s['x_opt'], res_s['z_opt'], cfg['es_max']
                )
                st.plotly_chart(fig2d, use_container_width=True)
            except Exception:
                st.warning("No se pudo generar la grafica 2D para este problema.")

            st.subheader("Funcion objetivo en 3D")
            fig3d = graficos.grafico_objetivo_3d(
                cfg['c'], cfg['A'], cfg['b'], cfg['signos'],
                res_s['x_opt'], res_s['z_opt']
            )
            if fig3d is not None:
                st.plotly_chart(fig3d, use_container_width=True)
            else:
                st.info("No se puede generar la grafica 3D para este problema.")

        elif cfg['n_vars'] == 3:
            st.subheader("Region factible 3D (poliedro)")
            fig3d = graficos.grafico_polytope_3d(
                cfg['c'], cfg['A'], cfg['b'], cfg['signos'],
                res_s['x_opt'], res_s['z_opt']
            )
            if fig3d is not None:
                st.plotly_chart(fig3d, use_container_width=True)
            else:
                st.warning(
                    "No se puede generar la grafica 3D: no se encontraron vertices "
                    "factibles o el problema no esta acotado."
                )
        else:
            st.info(
                f"La visualizacion grafica esta disponible para 2 o 3 variables. "
                f"Este problema tiene {cfg['n_vars']} variables."
            )


with tab_hs:
    st.header("Harmony Search")

    if not st.session_state.get('resultados'):
        st.info("Configure el problema en el panel lateral y presione **Resolver**.")
    elif not st.session_state.config.get('usar_hs'):
        st.warning("Harmony Search no esta seleccionado.")
    elif st.session_state.resultados.get('hs_error'):
        st.error(f"Error: {st.session_state.resultados['hs_error']}")
    else:
        res_h = st.session_state.resultados['hs']
        cfg   = st.session_state.config

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Z encontrada", f"{res_h['z_opt']:.4f}")
        col2.metric("Iteraciones", res_h['parametros']['NI'])
        col3.metric("Tiempo", f"{res_h['tiempo']*1000:.2f} ms")
        col4.metric("Factible", "Si" if res_h['factible'] else "No")

        if not res_h['factible']:
            st.warning(
                f"Violacion acumulada de restricciones: {res_h['penalidad']:.6f}. "
                "Intente aumentar las iteraciones o ajustar los parametros."
            )

        df_x = pd.DataFrame({
            'Variable': simplex.nombres_variables(cfg['n_vars']),
            'Valor': np.round(res_h['x_opt'], 4)
        })
        st.subheader("Valores encontrados")
        st.table(tabla(df_x))

        st.subheader("Verificacion de restricciones")
        df_v = pd.DataFrame(res_h['verificacion'])
        df_v['lhs'] = df_v['lhs'].round(4)
        df_v['Estado'] = df_v['cumple'].map({True: 'OK', False: 'VIOLADA'})
        df_v = df_v[['restriccion', 'lhs', 'signo', 'rhs', 'Estado']]
        df_v.columns = ['Restriccion', 'Lado Izquierdo', 'Signo', 'Lado Derecho', 'Estado']
        st.table(tabla(df_v))

        st.subheader("Convergencia")
        try:
            fig_conv = graficos.grafico_convergencia(res_h['historia_fit'], res_h['historia_z'])
            st.plotly_chart(fig_conv, use_container_width=True)
        except Exception:
            st.warning("No se pudo generar la grafica de convergencia.")

        st.subheader("Exploracion del espacio de busqueda")
        fig_exp = graficos.grafico_exploracion_3d(
            res_h['todas_evaluaciones'], cfg['c'],
            res_h['x_opt'], res_h['z_opt'], cfg['n_vars']
        )
        if fig_exp is not None:
            st.plotly_chart(fig_exp, use_container_width=True)
        else:
            st.info("No se puede generar la grafica 3D para este problema.")

        with st.expander("Parametros utilizados"):
            st.json(res_h['parametros'])


with tab_comp:
    st.header("Comparacion entre metodos")

    if not st.session_state.get('resultados'):
        st.info("Configure el problema en el panel lateral y presione **Resolver**.")
    else:
        cfg = st.session_state.config
        res = st.session_state.resultados

        if cfg['usar_simplex'] and cfg['usar_hs'] and res.get('simplex') and res.get('hs'):
            res_s = res['simplex']
            res_h = res['hs']

            df_comp = pd.DataFrame({
                'Metrica': ['Z optima', 'Tiempo (ms)', 'Iteraciones', 'Solucion factible'],
                'Simplex': [
                    f"{res_s['z_opt']:.4f}",
                    f"{res_s['tiempo']*1000:.2f}",
                    res_s['iteraciones'],
                    "Si"
                ],
                'Harmony Search': [
                    f"{res_h['z_opt']:.4f}",
                    f"{res_h['tiempo']*1000:.2f}",
                    res_h['parametros']['NI'],
                    "Si" if res_h['factible'] else "No"
                ]
            })
            st.table(tabla(df_comp))

            diff = abs(res_s['z_opt'] - res_h['z_opt'])
            rel  = diff / abs(res_s['z_opt']) * 100 if res_s['z_opt'] != 0 else 0.0
            st.metric("|Z_simplex - Z_HS|", f"{diff:.4f}", f"{rel:.2f}% de diferencia")

            nombres_c = simplex.nombres_variables(cfg['n_vars'])
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Solucion Simplex**")
                st.table(tabla(pd.DataFrame({
                    'Variable': nombres_c,
                    'Valor': np.round(res_s['x_opt'], 4)
                })))
            with col_b:
                st.markdown("**Solucion Harmony Search**")
                st.table(tabla(pd.DataFrame({
                    'Variable': nombres_c,
                    'Valor': np.round(res_h['x_opt'], 4)
                })))

            if cfg['n_vars'] == 2:
                st.subheader("Exploracion HS sobre la region factible")
                fig_comp = graficos.grafico_comparacion(
                    res_s, res_h, cfg['c'], cfg['A'], cfg['b'],
                    cfg['signos'], cfg['n_vars'], cfg['es_max']
                )
                if fig_comp is not None:
                    st.plotly_chart(fig_comp, use_container_width=True)
                else:
                    st.info("No se pudo generar la grafica de comparacion para este problema.")
        else:
            st.info("Active ambos metodos en el panel lateral y resuelva el problema para comparar.")

st.divider()
st.caption("Programacion Lineal - Optimizacion: Simplex y Harmony Search")
