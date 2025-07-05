import sys
import os

# Add the Functions directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Functions.SimulationFunctions import BinaryLens_Data
from Functions.PlotFunctions import plot_binary_lens_trajectory_interactive
import numpy as np

# Crear el directorio si no existe
output_dir = "./Simulacion/Images/BinaryLense"
os.makedirs(output_dir, exist_ok=True)

# Definir parámetros del sistema
m_t = 1.0      # Masa total
q = 1        # Relación de masas
z1 = 0.0       # Posición de la primera lente

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
print(f"{binary_system=}")
binary_system.images_magnification_calculation()

plot_binary_lens_trajectory_interactive(
    binary_system, 
    plot_type='both',
    save_path=os.path.join(output_dir, 'binary_lens_trajectory_interactive.pdf')
)