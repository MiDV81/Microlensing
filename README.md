# Simulación de Microlentes Gravitacionales y Clasificación con Redes Neuronales

## Descripción del Proyecto

Este proyecto es un marco integral para la simulación, visualización y clasificación con aprendizaje automático de eventos de microlentes gravitacionales. Combina simulaciones de física teórica con técnicas modernas de aprendizaje profundo para analizar y clasificar eventos de microlentes, enfocándose particularmente en la detección de exoplanetas a través de sistemas de lentes binarias.

## Tabla de Contenidos

- [Características](#características)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
- [Uso](#uso)
- [Descripción de Módulos](#descripción-de-módulos)
- [Ejemplos](#ejemplos)
- [Requisitos](#requisitos)

## Características

### 🔭 **Simulaciones Físicas**
- **Microlentes Simple**: Simulación completa de eventos de lente puntual
- **Microlentes Binaria**: Sistemas complejos de lentes binarias con curvas críticas y cáusticas
- **Curvas Críticas y Cáusticas**: Cálculo y visualización automatizados
- **Generación de Curvas de Luz**: Curvas de magnificación realistas con parámetros personalizables

### 📊 **Visualización y Gráficos**
- **Gráficos Estáticos**: Visualizaciones de alta calidad de trayectorias y magnificación
- **GIFs Animados**: Visualización dinámica del movimiento de la fuente y formación de imágenes
- **Gráficos Interactivos**: Exploración basada en controles deslizantes de sistemas de lentes binarias
- **Diseños en Cuadrícula**: Visualización comparativa de múltiples configuraciones de lentes
- **Temas Personalizables**: Soporte para diferentes colores de fondo y estilos

### 🤖 **Aprendizaje Automático**
- **Clasificación con Redes Neuronales**: Modelos de aprendizaje profundo para clasificación de tipos de eventos
- **Pipeline de Datos**: Preprocesamiento y aumento de datos automatizado
- **Evaluación de Modelos**: Análisis y visualización integral del rendimiento
- **Integración de Datos Reales**: Soporte para datos reales de surveys de microlentes

### 📈 **Análisis de Datos**
- **Base de Datos de Exoplanetas**: Integración con conjuntos de datos de exoplanetas confirmados
- **Análisis Estadístico**: Estudios de frecuencia de eventos y eficiencia de detección
- **Simulación de Ruido**: Modelado realista de ruido observacional
- **Métodos de Interpolación**: Múltiples técnicas de interpolación de curvas

## Estructura del Proyecto

```
GitHub/
├── Functions/                 # Módulos de funcionalidad central
│   ├── SimulationFunctions.py    # Simulaciones físicas y estructuras de datos
│   ├── PlotFunctions.py          # Utilidades de visualización y gráficos
│   ├── NNFunctions.py            # Utilidades de redes neuronales
│   ├── NNWorkflows.py            # Gestión de flujos de trabajo de ML
│   └── NNClass.py               # Clases personalizadas de redes neuronales
│
├── Simulacion/               # Scripts de simulación y ejemplos
│   ├── SingleLense.py           # Simulación de lente simple
│   ├── SingleLenseAnimated.py   # Gráficos animados de lente simple
│   ├── Static_BinaryLense.py    # Gráficos estáticos de lente binaria
│   ├── BinaryLenseAnimated.py   # Gráficos animados de lente binaria
│   ├── Topology_BinaryLense.py  # Análisis de topología de cáusticas
│   ├── CriticalCausticCurves.py # Visualización de curvas críticas
│   ├── PWPPlot.py              # Gráficos de calidad de publicación
│   └── Images/                 # Gráficos y animaciones generados
│
├── Redes/                    # Entrenamiento y análisis de redes neuronales
│   ├── 1X_*.py                 # Carga y preprocesamiento de datos (10s)
│   ├── 2X_*.py                 # Simulación y generación de datos (20s)
│   ├── 3X_*.py                 # Construcción y entrenamiento de modelos (30s)
│   ├── 4X_*.py                 # Aplicaciones e interfaces (40s)
│   ├── 8X_*.py                 # Evaluación y comparación de modelos (80s)
│   ├── 9X_*.py                 # Pruebas y análisis (90s)
│   └── MicrolensingData/       # Conjuntos de datos de entrenamiento
│
├── ExoplanetsDatabase/       # Datos y análisis de exoplanetas
│   ├── Exoplanets.ipynb         # Análisis en Jupyter notebook
│   ├── Exoplanets_Confirmed_db_1504.csv  # Catálogo de exoplanetas
│   └── *.pdf                   # Reportes y gráficos generados
```

## Instalación

### Prerrequisitos
- Python 3.8+

### Configuración

1. **Clonar el repositorio:**
   ```bash
   git clone <repository-url>
   cd GitHub
   ```

2. **Instalar dependencias:**
   ```bash
   pip install numpy matplotlib scipy tensorflow pandas jupyter pillow
   ```

## Uso

### Ejemplos de Inicio Rápido

#### 1. Simulación de Lente Simple
```python
from Functions.SimulationFunctions import SingleLens_Data
from Functions.PlotFunctions import plot_single_lens

# Crear parámetros de la lente
lens = SingleLens_Data(
    t_E=10.0,     # Tiempo de Einstein (días)
    u_0=0.3,      # Parámetro de impacto
    num_points=1000
)

# Generar gráfico estático
plot_single_lens(lens, plot_type='both', save_path='single_lens.png')
```

#### 2. Animación de Lente Binaria
```python
from Simulacion.BinaryLenseAnimated import plot_binary_lens_animated_gif
from Functions.SimulationFunctions import BinaryLens_Data

# Crear sistema binario
binary = BinaryLens_Data(
    m_t=1.0, q=0.1, z1=0.5,
    start_point=(-2, -1), end_point=(2, 1),
    num_points=500
)

# Generar GIF animado
plot_binary_lens_animated_gif(
    binary_data=binary,
    save_path='binary_animation.gif',
    fps=15, duration_seconds=8.0
)
```

#### 3. Entrenamiento de Red Neuronal
```python
from Functions.NNWorkflows import ModelBuilder

config = {
    "sequence_length": 100,
    "batch_size": 32,
    "epochs": 50,
    "test_fraction": 0.2
}

ModelBuilder(
    model_configuration=config,
    load_filename="combined_lightcurves.pkl",
    model_filename="classifier.keras"
)
```

### Ejecutar Scripts de Simulación

Ejecutar cualquier script desde la carpeta GitHub:

```bash
python Simulacion/SingleLense.py          # Generar gráficos de lente simple
python Simulacion/Topology_BinaryLense.py # Crear cuadrícula de topología de cáusticas
python Simulacion/PWPPlot.py              # Generar gráficos de publicación
```

### Flujo de Trabajo de Redes Neuronales

El directorio `Redes/` contiene scripts numerados para el pipeline completo de ML:

```bash
python Redes/21_SingleLensesSimulations.py  # Generar datos de lente simple
python Redes/22_BinaryLensesSimulations.py  # Generar datos de lente binaria
python Redes/31_ModelBuilder.py             # Entrenar red neuronal
python Redes/32_ModelChecker.py             # Evaluar modelo
```

## Descripción de Módulos

### Functions/SimulationFunctions.py
Módulo central de simulación física que contiene:
- `SingleLens_Data`: Dataclass para parámetros de lente simple
- `BinaryLens_Data`: Dataclass para sistemas de lentes binarias
- `trajectory_single_lens()`: Calcular trayectorias de lente simple
- `calculate_caustics_and_critical_curves_binary_lense()`: Computar curvas críticas y cáusticas

### Functions/PlotFunctions.py
Utilidades integrales de visualización:
- `plot_single_lens()`: Visualización estática de lente simple
- `plot_binary_lens_trajectory_static()`: Gráficos estáticos de lente binaria
- `plot_binary_lens_trajectory_interactive()`: Explorador interactivo de lente binaria
- `plot_binary_lens_caustics_grid()`: Comparación en cuadrícula de topologías de cáusticas

### Functions/NNWorkflows.py
Gestión de pipeline de aprendizaje automático:
- `ModelBuilder()`: Flujo de trabajo automatizado de entrenamiento de modelos
- `ModelChecker()`: Validación y prueba de modelos
- `ModelUser()`: Utilidades de inferencia y predicción

## Características Clave en Detalle

### 🎨 **Capacidades de Visualización**
- **Temas de Fondo**: Soporte para fondos blancos, oscuros y personalizados (#E8E8E8)
- **Animación**: Generación suave de GIFs con velocidades de fotogramas personalizables
- **Elementos Interactivos**: Widgets deslizantes de matplotlib para exploración de parámetros
- **Calidad de Publicación**: Gráficos de alta resolución listos para publicaciones académicas

### 🔬 **Precisión Física**
- **Anillo de Einstein**: Implementación adecuada de la teoría de lentes gravitacionales
- **Ecuaciones de Lente Binaria**: Resolución precisa de polinomios de 5º orden para posiciones de imágenes
- **Clasificación de Cáusticas**: Detección automática de topologías cerradas, intermedias y anchas
- **Cálculo de Magnificación**: Computación precisa incluyendo efectos de fuente finita

### 🧠 **Pipeline de Aprendizaje Automático**
- **Aumento de Datos**: Simulación de ruido e interpolación de curvas
- **Arquitectura de Modelos**: Redes híbridas LSTM/CNN personalizables
- **Métricas de Rendimiento**: Evaluación integral con matrices de confusión
- **Integración de Datos Reales**: Soporte para datos de surveys OGLE, MOA y otros

## Ejemplos y Resultados

### Visualizaciones Generadas
- **Trayectorias de Lente Simple**: Trayectorias de fuente e imágenes con anillo de Einstein
- **Animaciones de Lente Binaria**: Eventos dinámicos de cruce de cáusticas
- **Cuadrículas de Topología**: Estructuras comparativas de cáusticas
- **Curvas de Luz**: Magnificación vs. tiempo con múltiples componentes

### Resultados de Aprendizaje Automático
- **Precisión de Clasificación**: Predicción de tipos de eventos (Simple/Binaria/Planetaria)
- **Eficiencia de Detección**: Análisis estadístico de tasas de detección
- **Estimación de Parámetros**: Extracción automatizada de parámetros de lente

## Requisitos

```
numpy
matplotlib
scipy
tensorflow
pandas
jupyter
pillow
```

## Licencia

Este proyecto se desarrolla como parte de un Trabajo de Fin de Grado (TFG) en la Universidad de Oviedo.

---

*Este proyecto representa un enfoque integral para la simulación y análisis de microlentes gravitacionales, combinando física teórica con técnicas computacionales modernas para la detección y caracterización de exoplanetas.*
