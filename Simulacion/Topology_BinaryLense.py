import sys
import os
import numpy as np

# Add the Functions directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Functions.SimulationFunctions import BinaryLens_Data
from Functions.PlotFunctions import plot_binary_lens_caustics_grid

# Crear el directorio si no existe
output_dir = "./Simulacion/Images/BinaryLense"
os.makedirs(output_dir, exist_ok=True)

# Crear 4 sistemas con diferentes configuraciones de q y d
# q: logarithmically spaced from 0.001 to 0.01
# d: linearly spaced from 1.0 to 3.0

num_systems = 4
# q_values = np.logspace(np.log10(0.01), np.log10(0.1), num_systems)  # [0.001, 0.002, 0.005, 0.01]
# d_values = np.linspace(0.5, 2.0, num_systems)  # [1.0, 1.67, 2.33, 3.0]
q_values = [1.0, 1.0, 1.0]
d_values = [0.5, 1.2, 2.5]
# Crear los sistemas binarios usando el parámetro d
binary_systems = []
topology_labels = []

for i in range(num_systems):
    # Crear sistema usando d (separation distance) en lugar de z1
    system = BinaryLens_Data(
        m_t=1.0,           # Masa total constante
        q=q_values[i],     # Mass ratio variable
        d=d_values[i]      # Separation distance variable (z1 will be calculated as d/2)
    )
    binary_systems.append(system)
    

print(f"Creados {len(binary_systems)} sistemas:")
for i, system in enumerate(binary_systems):
    print(f"Sistema {i+1}: q={system.q:.3f}, d={d_values[i]:.1f}, z1={system.z1:.2f}")

# Crear el plot de grilla con 1 fila (4x1 grid)
plot_binary_lens_caustics_grid(
    binary_systems=binary_systems,
    num_rows=1,  # 1 fila, 4 columnas
    num_points=1000,
    auto_limits=True,  # Límites automáticos basados en z1 de cada sistema
    save_path=os.path.join(output_dir, 'topologies_presentation.pdf')
)
