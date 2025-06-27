import sys
import os
import numpy as np

# Add the Functions directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Functions.SimulationFunctions import BinaryLens_Data, trajectory_binary_lens
from Functions.PlotFunctions import plot_binary_lens_trajectory_static

# Crear el directorio si no existe
output_dir = "./Simulacion/Images/BinaryLense"
os.makedirs(output_dir, exist_ok=True)

print("=== Análisis Estático de Trayectoria de Lentes Binarias ===")

# Definir parámetros del sistema
m_t = 1.0      # Masa total
q = 0.1        # Relación de masas
z1 = 0.5       # Posición de la primera lente

# Definir trayectoria
start_point = (-2, -1)    # Punto inicial (x, y)
end_point = (2, 1)        # Punto final (x, y)
num_points = 2_000          # Número de puntos en la trayectoria

binary_system = BinaryLens_Data(
    m_t=m_t, 
    q=q, 
    z1=z1,
    start_point=start_point,
    end_point=end_point,
    num_points=num_points
)

binary_system.images_magnification_calculation()
print(f"{binary_system=}")

plot_binary_lens_trajectory_static(
    binary_system,
    show_caustics=True,
    special_indices = [800, 995, 1030, 1170, 1293, 1400],
    plot_type='both',
    save_path=os.path.join(output_dir, 'binary_lens_trajectory_static_complete_v2.pdf')
)

