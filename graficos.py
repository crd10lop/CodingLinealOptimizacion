import numpy as np
import plotly.graph_objects as go
from itertools import combinations


def _es_factible(x, A, b, signos, tol=1e-6):
    for i in range(len(b)):
        v = float(A[i] @ x)
        if signos[i] == '<=' and v > b[i] + tol:
            return False
        if signos[i] == '>=' and v < b[i] - tol:
            return False
        if signos[i] == '=' and abs(v - b[i]) > tol:
            return False
    if np.any(x < -tol):
        return False
    return True


def vertices_factibles(A, b, signos, n_vars):
    A_ext = np.vstack([A, np.eye(n_vars)])
    b_ext = np.hstack([b, np.zeros(n_vars)])
    vertices = []

    for combo in combinations(range(len(b_ext)), n_vars):
        M = A_ext[list(combo)]
        rhs = b_ext[list(combo)]
        try:
            p = np.linalg.solve(M, rhs)
            if _es_factible(p, A, b, signos):
                vertices.append(p)
        except np.linalg.LinAlgError:
            continue

    if not vertices:
        return np.array([])

    vertices = np.array(vertices)
    unicos = []
    for v in vertices:
        if not any(np.allclose(v, u, atol=1e-4) for u in unicos):
            unicos.append(v)
    return np.array(unicos)


def grafico_region_2d(c, A, b, signos, x_opt, z_opt, es_max):
    vertices = vertices_factibles(A, b, signos, 2)
    fig = go.Figure()

    if len(vertices) > 0:
        max_v = float(np.max(vertices)) * 1.4
        max_x = max(max_v, 10.0)
    else:
        max_x = 20.0
    max_y = max_x

    x_vals = np.linspace(0, max_x, 200)
    colores = ['#1f77b4', '#2ca02c', '#9467bd', '#17becf', '#bcbd22', '#e377c2']
    for i in range(A.shape[0]):
        a1, a2 = A[i]
        color = colores[i % len(colores)]
        if a2 != 0:
            y_vals = (b[i] - a1 * x_vals) / a2
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode='lines', name=f'R{i+1}',
                line=dict(color=color, dash='dash', width=2)
            ))
        elif a1 != 0:
            xv = b[i] / a1
            fig.add_trace(go.Scatter(
                x=[xv, xv], y=[0, max_y],
                mode='lines', name=f'R{i+1}',
                line=dict(color=color, dash='dash', width=2)
            ))

    if len(vertices) >= 3:
        cx, cy = vertices.mean(axis=0)
        angulos = np.arctan2(vertices[:, 1] - cy, vertices[:, 0] - cx)
        orden = np.argsort(angulos)
        sv = vertices[orden]
        fig.add_trace(go.Scatter(
            x=np.append(sv[:, 0], sv[0, 0]),
            y=np.append(sv[:, 1], sv[0, 1]),
            fill='toself',
            mode='lines+markers',
            line=dict(color='goldenrod', width=2),
            fillcolor='rgba(255, 215, 0, 0.25)',
            marker=dict(size=8, color='goldenrod'),
            name='Region Factible'
        ))

    if c[1] != 0:
        y_iso = (z_opt - c[0] * x_vals) / c[1]
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_iso, mode='lines',
            name=f'Z = {z_opt:.2f}',
            line=dict(color='crimson', width=3)
        ))
    elif c[0] != 0:
        xv = z_opt / c[0]
        fig.add_trace(go.Scatter(
            x=[xv, xv], y=[0, max_y], mode='lines',
            name=f'Z = {z_opt:.2f}',
            line=dict(color='crimson', width=3)
        ))

    fig.add_trace(go.Scatter(
        x=[x_opt[0]], y=[x_opt[1]],
        mode='markers',
        marker=dict(size=18, color='crimson', symbol='diamond', line=dict(color='black', width=2)),
        name=f'Optimo ({x_opt[0]:.2f}, {x_opt[1]:.2f})'
    ))

    tipo = 'Maximo' if es_max else 'Minimo'
    fig.update_layout(
        xaxis=dict(title='X1', range=[0, max_x], gridcolor='lightgray'),
        yaxis=dict(title='X2', range=[0, max_y], gridcolor='lightgray'),
        title=f'Region Factible 2D | Z {tipo} = {z_opt:.2f}',
        plot_bgcolor='white',
        height=600,
        showlegend=True
    )
    return fig


def grafico_objetivo_3d(c, A, b, signos, x_opt, z_opt):
    vertices = vertices_factibles(A, b, signos, 2)
    if len(vertices) == 0:
        return None

    max_v = float(np.max(vertices)) * 1.3
    max_x = max(max_v, 10.0)
    max_y = max_x

    x_grid = np.linspace(0, max_x, 40)
    y_grid = np.linspace(0, max_y, 40)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z_plano = c[0] * X + c[1] * Y

    Z_factible = Z_plano.copy()
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            p = np.array([X[i, j], Y[i, j]])
            if not _es_factible(p, A, b, signos):
                Z_factible[i, j] = np.nan

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z_plano,
        opacity=0.25, colorscale='Greys', showscale=False,
        name='Plano Z'
    ))
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z_factible,
        opacity=0.9, colorscale='Viridis',
        colorbar=dict(title='Z'),
        name='Region Factible'
    ))

    z_v = vertices @ c
    fig.add_trace(go.Scatter3d(
        x=vertices[:, 0], y=vertices[:, 1], z=z_v,
        mode='markers',
        marker=dict(size=6, color='black'),
        name='Vertices'
    ))

    fig.add_trace(go.Scatter3d(
        x=[x_opt[0]], y=[x_opt[1]], z=[z_opt],
        mode='markers',
        marker=dict(size=12, color='crimson', symbol='diamond', line=dict(color='black', width=2)),
        name=f'Optimo Z={z_opt:.2f}'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X1',
            yaxis_title='X2',
            zaxis_title='Z',
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.2))
        ),
        title='Funcion Objetivo y Region Factible en 3D',
        height=650
    )
    return fig


def grafico_polytope_3d(c, A, b, signos, x_opt, z_opt):
    vertices = vertices_factibles(A, b, signos, 3)
    if len(vertices) == 0:
        return None

    fig = go.Figure()
    z_vals = vertices @ c

    if len(vertices) >= 4:
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(vertices)
            fig.add_trace(go.Mesh3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                i=hull.simplices[:, 0], j=hull.simplices[:, 1], k=hull.simplices[:, 2],
                opacity=0.35, color='lightblue',
                flatshading=True,
                name='Region Factible'
            ))
        except Exception:
            pass

    fig.add_trace(go.Scatter3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        mode='markers',
        marker=dict(size=6, color=z_vals, colorscale='Viridis',
                    showscale=True, colorbar=dict(title='Z'),
                    line=dict(color='black', width=1)),
        name='Vertices'
    ))

    fig.add_trace(go.Scatter3d(
        x=[x_opt[0]], y=[x_opt[1]], z=[x_opt[2]],
        mode='markers',
        marker=dict(size=12, color='crimson', symbol='diamond', line=dict(color='black', width=2)),
        name=f'Optimo Z={z_opt:.2f}'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X1',
            yaxis_title='X2',
            zaxis_title='X3',
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.2))
        ),
        title='Region Factible 3D',
        height=650
    )
    return fig


def grafico_convergencia(historia_fit, historia_z):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=historia_fit, mode='lines',
        line=dict(color='steelblue', width=1.5),
        name='Fitness (con penalidad)'
    ))

    z_clean = [z for z in historia_z if z is not None]
    primera_factible = next((i for i, z in enumerate(historia_z) if z is not None), None)
    if primera_factible is not None:
        fig.add_trace(go.Scatter(
            x=list(range(primera_factible, len(historia_z))),
            y=z_clean, mode='lines',
            line=dict(color='crimson', width=2),
            name='Mejor Z factible'
        ))

    fig.update_layout(
        xaxis_title='Improvisacion',
        yaxis_title='Valor',
        title='Convergencia del Harmony Search',
        plot_bgcolor='white',
        xaxis=dict(gridcolor='lightgray'),
        yaxis=dict(gridcolor='lightgray'),
        height=450
    )
    return fig


def grafico_exploracion_3d(todas_eval, c, x_opt, z_opt, n_vars):
    fig = go.Figure()

    if n_vars == 2:
        z_vals = todas_eval @ c
        iteraciones = np.arange(len(todas_eval))
        fig.add_trace(go.Scatter3d(
            x=todas_eval[:, 0], y=todas_eval[:, 1], z=z_vals,
            mode='markers',
            marker=dict(size=3, color=iteraciones, colorscale='Plasma',
                        showscale=True, colorbar=dict(title='Iteracion'),
                        opacity=0.6),
            name='Evaluaciones'
        ))
        fig.add_trace(go.Scatter3d(
            x=[x_opt[0]], y=[x_opt[1]], z=[z_opt],
            mode='markers',
            marker=dict(size=12, color='crimson', symbol='diamond', line=dict(color='black', width=2)),
            name=f'Optimo Z={z_opt:.2f}'
        ))
        fig.update_layout(scene=dict(xaxis_title='X1', yaxis_title='X2', zaxis_title='Z'))

    elif n_vars >= 3:
        iteraciones = np.arange(len(todas_eval))
        z_vals = todas_eval @ c
        fig.add_trace(go.Scatter3d(
            x=todas_eval[:, 0], y=todas_eval[:, 1], z=todas_eval[:, 2],
            mode='markers',
            marker=dict(size=3, color=z_vals, colorscale='Viridis',
                        showscale=True, colorbar=dict(title='Z'),
                        opacity=0.5),
            name='Evaluaciones'
        ))
        fig.add_trace(go.Scatter3d(
            x=[x_opt[0]], y=[x_opt[1]], z=[x_opt[2]],
            mode='markers',
            marker=dict(size=12, color='crimson', symbol='diamond', line=dict(color='black', width=2)),
            name=f'Optimo Z={z_opt:.2f}'
        ))
        fig.update_layout(scene=dict(xaxis_title='X1', yaxis_title='X2', zaxis_title='X3'))

    else:
        return None

    fig.update_layout(
        title='Exploracion del Espacio de Busqueda - Harmony Search',
        height=650,
        scene=dict(camera=dict(eye=dict(x=1.6, y=-1.6, z=1.2)))
    )
    return fig


def grafico_comparacion(simplex_res, hs_res, c, A, b, signos, n_vars, es_max):
    if n_vars != 2:
        return None

    fig = grafico_region_2d(c, A, b, signos, simplex_res['x_opt'], simplex_res['z_opt'], es_max)

    todas = hs_res['todas_evaluaciones']
    iteraciones = np.arange(len(todas))
    fig.add_trace(go.Scatter(
        x=todas[:, 0], y=todas[:, 1],
        mode='markers',
        marker=dict(size=4, color=iteraciones, colorscale='Plasma',
                    showscale=True, colorbar=dict(title='Iteracion HS', x=1.15),
                    opacity=0.4),
        name='Exploracion HS'
    ))

    fig.add_trace(go.Scatter(
        x=[hs_res['x_opt'][0]], y=[hs_res['x_opt'][1]],
        mode='markers',
        marker=dict(size=15, color='purple', symbol='star', line=dict(color='black', width=2)),
        name=f'HS Optimo Z={hs_res["z_opt"]:.2f}'
    ))

    fig.update_layout(title='Comparacion Simplex vs Harmony Search')
    return fig
