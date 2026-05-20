# Optimizacion Lineal: Simplex y Harmony Search

Aplicacion web interactiva para resolver problemas de programacion lineal usando dos enfoques: el metodo exacto **Simplex** y la metaheuristica **Harmony Search**. Incluye visualizaciones 2D y 3D con Plotly.

## Estructura del proyecto

```
.
├── app.py                       Interfaz Streamlit
├── simplex.py                   Algoritmo Simplex
├── harmony_search.py            Algoritmo Harmony Search
├── graficos.py                  Visualizaciones Plotly
├── requirements.txt             Dependencias
├── .streamlit/config.toml       Tema de la app
└── Programación_lineal.ipynb    Notebook original (referencia)
```

## Despliegue en Streamlit Community Cloud (gratuito)

Como la app esta pensada para verse sin instalar nada localmente:

1. Ingresar a https://share.streamlit.io e iniciar sesion con la cuenta de GitHub.
2. Click en **New app**.
3. Seleccionar el repositorio `crd10lop/CodingLinealOptimizacion`.
4. Branch: `main` (despues de hacer merge del PR).
5. Main file path: `app.py`.
6. Click en **Deploy**.

Streamlit Cloud instalara las dependencias listadas en `requirements.txt` y publicara la app en una URL del tipo `https://<nombre>.streamlit.app`.

## Como usar la app

1. En el panel lateral se configura el problema: tipo de optimizacion, numero de variables y restricciones.
2. Se ingresan los coeficientes de la funcion objetivo y de cada restriccion.
3. Se eligen los metodos a ejecutar (Simplex, Harmony Search o ambos).
4. Si se usa Harmony Search, se ajustan los parametros del algoritmo (HMS, HMCR, PAR, BW, NI).
5. Se presiona **Resolver** y los resultados aparecen en las pestañas correspondientes.

## Notas tecnicas

- El metodo Simplex implementado soporta restricciones de tipo `<=`. Para `>=` o `=` se debe usar Harmony Search, que las maneja mediante penalizacion.
- Las visualizaciones graficas se generan automaticamente para problemas de 2 o 3 variables. Para mas variables solo se muestran los resultados numericos.
- Harmony Search es estocastico, por lo que cada ejecucion puede dar resultados ligeramente diferentes. La semilla permite reproducir corridas especificas.
