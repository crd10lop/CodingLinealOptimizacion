# Optimizacion Lineal: Simplex y Harmony Search

Aplicacion web interactiva para resolver problemas de programacion lineal usando dos enfoques: el metodo exacto **Simplex** y la metaheuristica **Harmony Search**. Incluye visualizaciones 2D y 3D con Plotly.

## Equipo

| Nombre | Correo |
|---|---|
| Juan Pablo Uribe González | juan.uribe4@udea.edu.co |
| Cristian David Diez López | cristian.diez@udea.edu.co |
| Zareth Mariana Vega Sánchez | zarethmariana.vega@udea.edu.co |

## Curso

**Asignatura:** Optimización  
**Docente:** Ronald Akerman Ortiz García  
**Departamento de Ingeniería de Sistemas**  
**Universidad de Antioquia — Semestre 2026-1**

## Contexto

Este proyecto hace parte de la actividad de programación lineal del curso de Optimización. El equipo tuvo asignado el algoritmo **Harmony Search** como método metaheurístico a estudiar e implementar. A partir de un notebook desarrollado en Google Colab, se construyó una aplicación web interactiva con Streamlit que permite comparar el método exacto Simplex (Gran M) con Harmony Search, incluyendo análisis de sensibilidad y visualizaciones 2D y 3D generadas con Plotly.


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

Como la app esta pensada para verse sin instalar nada localmente, se publico con el siguiente enlace: 
*`https://codinglinealoptimizacion.streamlit.app/`.*


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
