import sys
import os

# Add the Functions directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Functions.SimulationFunctions import BinaryLens_Data, lens_equation_binary_lense, magnification_binary_lense
from Functions.PlotFunctions import plot_binary_lens_trajectory_interactive
import numpy as np

# Crear el directorio si no existe
output_dir = "./Simulacion/Images/BinaryLense"
os.makedirs(output_dir, exist_ok=True)

# Definir parámetros del sistema
m_t = 1.0      # Masa total
m_d = 0.3      # Diferencia de masa
z1 = 2.0       # Posición de la primera lente

# Definir trayectoria
start_point = (-3, -1)    # Punto inicial (x, y)
end_point = (3, 1)        # Punto final (x, y)
num_points = 500          # Número de puntos en la trayectoria

binary_system = BinaryLens_Data(m_t=m_t, m_d=m_d, z1=z1, 
                                start_point=start_point, end_point=end_point, num_points=num_points)
binary_system.images_magnification_calculation()

plot_binary_lens_trajectory_interactive(
    binary_system, 
    plot_type='both',
    save_path=os.path.join(output_dir, 'binary_lens_trajectory_interactive.pdf')
)