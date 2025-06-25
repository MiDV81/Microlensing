import sys
import os

# Add the Functions directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Functions.SimulationFunctions import SingleLens_Data
from Functions.PlotFunctions import plot_single_lens
import numpy as np

# Crear el directorio si no existe
output_dir = "./Simulacion/Images/SingleLense"
os.makedirs(output_dir, exist_ok=True)

# Definir parámetros de la lente
lens_params = SingleLens_Data(
    t_E=10.0,    # Tiempo de Einstein en días
    u_0=0.5,     # Parámetro de impacto mínimo
    num_points=1000
)

plot_single_lens(
    lens_data=lens_params, 
    plot_type='trajectory', 
    save_path=os.path.join(output_dir, 'SingleLense_trajectory.pdf')
)
plot_single_lens(
    lens_data=lens_params, 
    plot_type='magnification', 
    save_path=os.path.join(output_dir, 'SingleLense_magnification.pdf')
)
