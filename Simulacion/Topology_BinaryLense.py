import sys
import os

# Add the Functions directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Functions.SimulationFunctions import BinaryLens_Data
from Functions.PlotFunctions import plot_binary_lens_caustics_grid
import numpy as np

# Crear el directorio si no existe
output_dir = "./Simulacion/Images/BinaryLense"
os.makedirs(output_dir, exist_ok=True)


# Caso 1: Topología Cerrada (d pequeño)
binary_system_1 = BinaryLens_Data(
    m_t=1.0,    # Masa total normalizada
    m_d=0.0,    # Masas iguales (q=1)
    z1=0.25     # Separación pequeña (d=0.5)
)

# Caso 2: Topología Intermedia (d medio)
binary_system_2 = BinaryLens_Data(
    m_t=1.0,    # Masa total normalizada
    m_d=0.0,    # Masas iguales (q=1)
    z1=0.625    # Separación media (d=1.25)
)

# Caso 3: Topología Ancha (d grande)
binary_system_3 = BinaryLens_Data(
    m_t=1.0,    # Masa total normalizada
    m_d=0.0,    # Masas iguales (q=1)
    z1=1.25     # Separación grande (d=2.5)
)

# Etiquetas de topología
topology_labels = ['Cerrada', 'Intermedia', 'Ancha']

# Crear el plot de grilla
plot_binary_lens_caustics_grid(
    binary_systems=[binary_system_1, binary_system_2, binary_system_3],
    topology_labels=topology_labels,
    num_points=1000,
    save_path=os.path.join(output_dir, 'binary_lens_topology_comparison.pdf')
)