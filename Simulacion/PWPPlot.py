import sys
import os
import numpy as np
from typing import Optional

# Add the Functions directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Functions.SimulationFunctions import BinaryLens_Data, calculate_caustics_and_critical_curves_binary_lense
# from Functions.PlotFunctions import plot_binary_lens_trajectory_static

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




def plot_binary_lens_trajectory_static(binary_data: BinaryLens_Data, 
                                      plot_type: str = 'both',
                                      special_indices: Optional[list] = None,
                                      show_caustics: bool = True,
                                      num_caustic_points: int = 1000,
                                      save_path: Optional[str] = None):
    """
    Plotea un análisis estático completo de la trayectoria de lentes binarias con caústicas en modo oscuro.

    Args:
        binary_data (BinaryLens_Data): Datos del sistema con trayectoria calculada
        plot_type (str): Tipo de plot - 'trajectory', 'magnification', o 'both' (default: 'both')
        special_indices (list): Ignorado, se muestran todas las imágenes en gris
        show_caustics (bool): Si mostrar curvas críticas y caústicas
        num_caustic_points (int): Número de puntos para calcular caústicas
        save_path (str): Ruta para guardar la imagen de la trayectoria (como PNG)
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    print(f"=== Plot Estático Binary Lens - Tipo: {plot_type} ===")

    if len(binary_data.zeta) == 0 or len(binary_data.image_positions) == 0:
        raise ValueError("No hay datos de trayectoria. Ejecutar trajectory_binary_lens primero.")

    num_points = len(binary_data.zeta)

    if show_caustics and plot_type in ['trajectory', 'both']:
        if not hasattr(binary_data, 'critical_points') or binary_data.critical_points.size == 0:
            binary_data = calculate_caustics_and_critical_curves_binary_lense(binary_data, num_caustic_points)

    plt.style.use('dark_background')

    # === TRAJECTORY PLOT ===
    if plot_type in ['trajectory', 'both']:
        fig, plot_ax = plt.subplots(1, 1, figsize=(10, 8))
        fig.patch.set_facecolor('#262626')
        plot_ax.set_facecolor('#262626')

        if show_caustics and hasattr(binary_data, 'critical_points') and len(binary_data.critical_points) > 0:
            plot_ax.scatter(binary_data.critical_points.real, binary_data.critical_points.imag, 
                            s=1, c='cyan', alpha=0.3)
            plot_ax.scatter(binary_data.caustic_points.real, binary_data.caustic_points.imag, 
                            s=1, c='magenta', alpha=0.3)

        lens1_size = 0.15 * binary_data.m1
        lens2_size = 0.15 * binary_data.m2
        lens1_circle = Circle((binary_data.z1, 0), lens1_size, 
                              facecolor='white', edgecolor='red', fill=True, zorder=10)
        lens2_circle = Circle((binary_data.z2, 0), lens2_size, 
                              facecolor='white', edgecolor='red', fill=True, zorder=10)
        plot_ax.add_patch(lens1_circle)
        plot_ax.add_patch(lens2_circle)

        source_real = [z.real for z in binary_data.zeta]
        source_imag = [z.imag for z in binary_data.zeta]
        plot_ax.plot(source_real, source_imag, 'lime', alpha=0.3, linestyle='--', linewidth=2)

        # Mostrar todas las imágenes como puntos grises pequeños
        for img_list in binary_data.image_positions:
            for sol in img_list:
                plot_ax.scatter(sol.real, sol.imag, c='black', s=1, alpha=0.6, zorder=2)

        plot_ax.set_xlim(-2, 2)
        plot_ax.set_ylim(-1.25, 1.25)
        plot_ax.set_aspect('equal')
        plot_ax.set_xticks([])
        plot_ax.set_yticks([])
        for spine in plot_ax.spines.values():
            spine.set_visible(False)
        # plot_ax.set_xlabel(r'$\mathrm{Re}(z)$ $[\theta_\mathrm{E}]$', fontsize=12, fontweight='bold')
        # plot_ax.set_ylabel(r'$\mathrm{Im}(z)$ $[\theta_\mathrm{E}]$', fontsize=12, fontweight='bold')

        if save_path:
            png_path = save_path.replace('.pdf', '.png')
            plt.savefig(png_path, bbox_inches='tight', dpi=300)
            print(f"Trayectoria guardada en: {png_path}")

        plt.show()

    # === MAGNIFICATION PLOT ===
    if plot_type in ['magnification', 'both']:
        fig, mag_ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.patch.set_facecolor('#262626')
        mag_ax.set_facecolor('#262626')

        t_normalized = np.linspace(0, 1, num_points)
        mag_ax.plot(t_normalized, binary_data.magnification, color='cyan', linewidth=1.5, alpha=0.9)

        mag_ax.set_xlim(0, 1)
        # mag_ax.set_xlabel(r'Posición Normalizada en la Trayectoria', fontsize=12, fontweight='bold')
        # mag_ax.set_ylabel(r'Magnificación Total', fontsize=12, fontweight='bold')

        for spine in mag_ax.spines.values():
            spine.set_visible(False)
        mag_ax.set_xticks([])
        mag_ax.set_yticks([])

        mag_range = np.max(binary_data.magnification) / np.min(binary_data.magnification)
        if mag_range > 100:
            mag_ax.set_yscale('log')

        plt.tight_layout()

        if save_path:
            png_path = save_path.replace('.pdf', '.png')
            plt.savefig(png_path, bbox_inches='tight', dpi=300)
            print(f"Trayectoria guardada en: {png_path}")

        plt.show()

    print("=== Plot estático completado ===")

# Ruta base sin extensión (se usará `.png` para trayectoria)
output_base_path = "./Simulacion/Images/BinaryLense/binary_lens_trajectory_static_dark.png"

# Llamar a la función con ambos plots y ruta base para guardado
plot_binary_lens_trajectory_static(
    binary_data=binary_system,
    plot_type='trajectory',
    show_caustics=True,
    save_path=output_base_path
)

# Para guardar también el gráfico de magnificación, puedes repetir con solo magnification:
plot_binary_lens_trajectory_static(
    binary_data=binary_system,
    plot_type='magnification',
    show_caustics=False,
    save_path=output_base_path.replace(".png", "_mag.png")  # o simplemente da otro nombre
)