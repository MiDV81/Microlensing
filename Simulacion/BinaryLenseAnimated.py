import sys
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import numpy as np

# Add the Functions directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Functions.SimulationFunctions import BinaryLens_Data, calculate_caustics_and_critical_curves_binary_lense

def plot_binary_lens_animated_gif(binary_data: BinaryLens_Data, save_path: str, 
                                 fps: int = 10, duration_seconds: float = 6.0,
                                 trail_length: int = 20, figsize: tuple = (12, 16),
                                 show_caustics: bool = True, num_caustic_points: int = 1000):
    """
    Crea un GIF animado mostrando la trayectoria de lentes binarias con magnificación.
    
    Args:
        binary_data (BinaryLens_Data): Datos del sistema de lentes binarias
        save_path (str): Ruta para guardar el GIF
        fps (int): Frames por segundo del GIF
        duration_seconds (float): Duración total del GIF en segundos
        trail_length (int): Longitud de la estela de puntos anteriores
        figsize (tuple): Tamaño de la figura (ancho, alto)
        show_caustics (bool): Si mostrar curvas críticas y caústicas
        num_caustic_points (int): Número de puntos para calcular caústicas
    """
    
    print(f"Generando GIF animado para Binary Lens: {save_path}")
    
    # Verificar que hay datos de trayectoria
    if len(binary_data.zeta) == 0 or len(binary_data.image_positions) == 0:
        print("ERROR: No hay datos de trayectoria calculados")
        raise ValueError("No hay datos de trayectoria. Ejecutar images_magnification_calculation primero.")
    
    # Calcular caústicas si se requieren y no están calculadas
    if show_caustics:
        if not hasattr(binary_data, 'critical_points') or binary_data.critical_points.size == 0:
            print("Calculando curvas críticas y caústicas...")
            binary_data = calculate_caustics_and_critical_curves_binary_lense(binary_data, num_caustic_points)
    
    num_points = len(binary_data.zeta)
    
    # Configurar parámetros de animación
    total_frames = int(fps * duration_seconds)
    step = max(1, num_points // total_frames)
    frame_indices = range(0, num_points, step)
    
    print(f"Total frames: {total_frames}, Step: {step}, Puntos de datos: {num_points}")
    
    # Crear figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[1, 0.6])
    fig.patch.set_facecolor('#E8E8E8')
    
    # === CONFIGURAR SUBPLOT DE TRAYECTORIAS (ARRIBA) ===
    ax1.set_facecolor('#E8E8E8')
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-1.25, 1.25)
    ax1.set_aspect('equal')
    ax1.set_ylabel(r'$\mathbf{Im(z)} [\theta_\mathrm{E}]$', fontsize=12, fontweight='bold', color='black')
    ax1.set_title(r'Trayectorias de la Fuente e Imágenes - Lentes Binarias', 
                  fontsize=14, pad=15, fontweight="bold", color='black')
    ax1.tick_params(labelsize=10, colors='black')
    ax1.grid(True, alpha=0.3, color='gray')
    
    # Elementos estáticos en el subplot de trayectorias
    # Plot caústicas y curvas críticas si están disponibles
    if show_caustics and hasattr(binary_data, 'critical_points') and len(binary_data.critical_points) > 0:
        ax1.scatter(binary_data.critical_points.real, binary_data.critical_points.imag, 
                   s=1, c='cyan', alpha=0.4, label='Curvas Críticas')
        ax1.scatter(binary_data.caustic_points.real, binary_data.caustic_points.imag, 
                   s=1, c='magenta', alpha=0.4, label='Caústicas')
    
    # Lentes
    lens1_size = 0.15 * binary_data.m1
    lens2_size = 0.15 * binary_data.m2
    lens1_circle = Circle((binary_data.z1, 0), lens1_size, 
                         facecolor='orange', edgecolor='black', fill=True, zorder=10, label='Lentes')
    lens2_circle = Circle((binary_data.z2, 0), lens2_size, 
                         facecolor='orange', edgecolor='black', fill=True, zorder=10)
    ax1.add_patch(lens1_circle)
    ax1.add_patch(lens2_circle)
    
    # Trayectoria completa de la fuente (línea de fondo)
    source_real = [z.real for z in binary_data.zeta]
    source_imag = [z.imag for z in binary_data.zeta]
    ax1.plot(source_real, source_imag, color='gray', alpha=0.4, 
            linewidth=4, linestyle='--', label='Trayectoria de la Fuente')
    
    # Leyenda
    ax1.legend(fontsize=10, loc='upper right', framealpha=0.9, facecolor='#E8E8E8', edgecolor='black')
    
    # === CONFIGURAR SUBPLOT DE MAGNIFICACIÓN (ABAJO) ===
    ax2.set_facecolor('#E8E8E8')
    
    # Crear array de tiempo normalizado para el eje x
    t_normalized = np.linspace(0, 1, num_points)
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, max(binary_data.magnification) * 1.1)  # Y desde 0
    ax2.set_xlabel(r'Posición Normalizada en la Trayectoria', fontsize=12, fontweight='bold', color='black')
    ax2.set_ylabel(r'Magnificación Total', fontsize=12, fontweight='bold', color='black')
    ax2.tick_params(labelsize=10, colors='black')
    ax2.grid(True, alpha=0.3, color='gray')
    
    # Configurar escala logarítmica si hay gran rango
    mag_range = np.max(binary_data.magnification) / np.min(binary_data.magnification)
    if mag_range > 100:
        ax2.set_yscale('log')
        ax2.set_ylim(min(binary_data.magnification) * 0.9, max(binary_data.magnification) * 1.1)
    
    # Curva de magnificación completa (línea de fondo)
    ax2.plot(t_normalized, binary_data.magnification, color='gray', alpha=0.4, 
             linewidth=4, linestyle='-', label='Magnificación Total')
    
    ax2.legend(fontsize=10, loc='upper right', framealpha=0.9, facecolor='#E8E8E8', edgecolor='black')
    
    # === ELEMENTOS ANIMADOS ===
    # Punto actual de la fuente
    source_current, = ax1.plot([], [], '^', color='blue', markersize=10, zorder=15, 
                               markeredgecolor='blue', markeredgewidth=2, label='Fuente Actual')
    
    # Puntos actuales de las imágenes (se actualizarán dinámicamente)
    images_current = ax1.scatter([], [], c='black', marker='o', s=50, zorder=14, 
                                edgecolors='black', linewidth=4, label='Imágenes Actuales')
    
    # Estela de la fuente
    source_trail, = ax1.plot([], [], '^', color='gray', alpha=0.6, markersize=6, zorder=12)
    
    # Línea vertical en el plot de magnificación
    time_line = ax2.axvline(x=0, color='purple', linewidth=1, alpha=0.8, zorder=10)
    
    # Punto actual en el plot de magnificación
    mag_current_point, = ax2.plot([], [], 'o', color='blue', markersize=8, zorder=15,
                                 markeredgecolor='blue', markeredgewidth=2)
    
    # Texto con información del tiempo actual
    time_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, fontsize=12, 
                         color='blue', fontweight='bold', verticalalignment='top',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor='#E8E8E8', alpha=0.8, edgecolor='black'))
    
    def animate(frame_num):
        if frame_num >= len(frame_indices):
            return []
        
        idx = frame_indices[frame_num]
        t_current = idx / (num_points - 1)  # Tiempo normalizado entre 0 y 1
        
        # Actualizar punto actual de la fuente
        current_source = binary_data.zeta[idx]
        source_current.set_data([current_source.real], [current_source.imag])
        
        # Actualizar imágenes actuales
        current_images = binary_data.image_positions[idx]
        if len(current_images) > 0:
            images_x = [img.real for img in current_images]
            images_y = [img.imag for img in current_images]
            images_current.set_offsets(np.column_stack([images_x, images_y]))
        else:
            images_current.set_offsets(np.empty((0, 2)))
        
        # Actualizar estela de la fuente (trail_length puntos anteriores)
        start_trail = max(0, idx - trail_length)
        trail_indices = range(start_trail, idx)
        
        if len(trail_indices) > 0:
            trail_sources = binary_data.zeta[trail_indices]
            trail_x = [z.real for z in trail_sources]
            trail_y = [z.imag for z in trail_sources]
            source_trail.set_data(trail_x, trail_y)
        else:
            source_trail.set_data([], [])
        
        # Actualizar línea vertical en magnificación
        time_line.set_xdata([t_current, t_current])
        
        # Actualizar punto en plot de magnificación
        mag_current_point.set_data([t_current], [binary_data.magnification[idx]])
        
        # Actualizar texto con información actual
        num_images = len(current_images) if len(current_images) > 0 else 0
        time_text.set_text(f'Posición: {t_current:.3f}\nImágenes: {num_images}\nMag: {binary_data.magnification[idx]:.2f}')
        
        return [source_current, images_current, source_trail, time_line, mag_current_point, time_text]
    
    # Crear animación
    anim = animation.FuncAnimation(fig, animate, frames=len(frame_indices), 
                                  interval=int(1000/fps), blit=True, repeat=True)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar GIF
    print(f"Guardando GIF animado en: {save_path}")
    try:
        anim.save(save_path, writer='pillow', fps=fps, bitrate=1800, dpi=100)
        print(f"GIF guardado exitosamente: {save_path}")
    except Exception as e:
        print(f"Error al guardar GIF: {e}")
        # Intentar guardar como MP4 si GIF falla
        mp4_path = save_path.replace('.gif', '.mp4')
        try:
            anim.save(mp4_path, writer='ffmpeg', fps=fps, bitrate=1800, dpi=100)
            print(f"Guardado como MP4 en su lugar: {mp4_path}")
        except Exception as e2:
            print(f"Error al guardar MP4: {e2}")
    
    plt.close(fig)
    return anim

# === EJECUCIÓN PRINCIPAL ===
if __name__ == "__main__":
    # Crear el directorio si no existe
    output_dir = "./Simulacion/Images/BinaryLense"
    os.makedirs(output_dir, exist_ok=True)

    print("=== Generación de GIF Animado - Lentes Binarias ===")

    # Definir parámetros del sistema (mismos que Static_BinaryLense.py)
    m_t = 1.0      # Masa total
    q = 0.1        # Relación de masas
    z1 = 0.5       # Posición de la primera lente

    # Definir trayectoria
    start_point = (-2, -1)    # Punto inicial (x, y)
    end_point = (2, 1)        # Punto final (x, y)
    num_points = 1000          # Reducido para GIF más fluido

    binary_system = BinaryLens_Data(
        m_t=m_t, 
        q=q, 
        z1=z1,
        start_point=start_point,
        end_point=end_point,
        num_points=num_points
    )

    print(f"Parámetros del sistema binario:")
    print(f"  - Masa total (m_t): {m_t}")
    print(f"  - Relación de masas (q): {q}")
    print(f"  - Posición primera lente (z1): {z1}")
    print(f"  - Trayectoria: {start_point} → {end_point}")
    print(f"  - Número de puntos: {num_points}")

    # Calcular trayectoria y magnificaciones
    binary_system.images_magnification_calculation()
    print(f"Sistema calculado: {binary_system}")

    # Generar GIF animado
    gif_path = os.path.join(output_dir, 'BinaryLense_animated_gray.gif')

    plot_binary_lens_animated_gif(
        binary_data=binary_system,
        save_path=gif_path,
        fps=12,  # Frames por segundo
        duration_seconds=8.0,  # Duración total del GIF
        trail_length=25,  # Longitud de la estela de puntos
        figsize=(12, 16),  # Tamaño de la figura
        show_caustics=True,  # Mostrar caústicas y curvas críticas
        num_caustic_points=1000
    )

    print("=== GIF Animado Completado ===")
    print(f"Archivo guardado en: {gif_path}")
