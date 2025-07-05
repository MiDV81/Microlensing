import sys
import os
import matplotlib.pyplot as plt
# Add the Functions directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Functions.SimulationFunctions import SingleLens_Data, trajectory_single_lens
import numpy as np
def plot_single_lens_animated_gif(lens_data: SingleLens_Data, save_path: str, 
                                 fps: int = 10, duration_seconds: float = 5.0,
                                 trail_length: int = 20, figsize: tuple = (12, 16)):
    """
    Crea un GIF animado mostrando la trayectoria de lente simple con magnificación.
    
    Args:
        lens_data (SingleLens_Data): Parámetros de la lente
        save_path (str): Ruta para guardar el GIF
        fps (int): Frames por segundo del GIF
        duration_seconds (float): Duración total del GIF en segundos
        trail_length (int): Longitud de la estela de puntos anteriores
        figsize (tuple): Tamaño de la figura (ancho, alto)
    """
    import matplotlib.animation as animation
    from matplotlib.patches import Circle
    
    print(f"Generando GIF animado para Single Lens: {save_path}")
    
    # Calcular todos los datos si no están calculados
    if lens_data.t.size == 0 or len(lens_data.image_pos) == 0:
        lens_data = trajectory_single_lens(lens_data)
    
    # Configurar parámetros de animación
    total_frames = int(fps * duration_seconds)
    step = max(1, len(lens_data.t) // total_frames)
    frame_indices = range(0, len(lens_data.t), step)
    
    print(f"Total frames: {total_frames}, Step: {step}, Puntos de datos: {len(lens_data.t)}")
    
    # Crear figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[1, 0.6])
    fig.patch.set_facecolor('#E8E8E8')

    # === CONFIGURAR SUBPLOT DE TRAYECTORIAS (ARRIBA) ===
    ax1.set_facecolor('#E8E8E8')
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-1, 2)
    ax1.set_aspect('equal')
    ax1.set_ylabel(r'$\mathbf{y (\theta_\mathrm{E})}$', fontsize=12, fontweight='bold', color='black')
    ax1.set_title(r'Trayectorias de la Fuente e Imágenes - Lente Simple', 
                  fontsize=14, pad=15, fontweight="bold", color='black')
    ax1.tick_params(labelsize=10, colors='black')
    ax1.grid(True, alpha=0.3, color='gray')
    
    # Elementos estáticos en el subplot de trayectorias
    # Anillo de Einstein
    einstein_ring = Circle((0, 0), 1, fill=False, color='blue', linestyle='--', 
                          label='Anillo de Einstein', alpha=0.7, linewidth=5)
    ax1.add_patch(einstein_ring)
    
    # Lente en el centro
    ax1.plot(0, 0, 'o', color='orange', markeredgecolor='black', 
             markeredgewidth=2, markersize=12, label='Lente', zorder=10)
    
    # Trayectorias completas (líneas de fondo)
    ax1.plot(lens_data.source_pos[:,0], lens_data.source_pos[:,1], 
             color='gray', alpha=0.4, linewidth=4, linestyle='-', label='Trayectoria Fuente')
    ax1.plot(lens_data.image_pos['image1'][:,0], lens_data.image_pos['image1'][:,1], 
             color='green', alpha=0.4, linewidth=4, linestyle='-', label='Trayectoria Imagen Mayor')
    ax1.plot(lens_data.image_pos['image2'][:,0], lens_data.image_pos['image2'][:,1], 
             color='red', alpha=0.4, linewidth=4, linestyle='-', label='Trayectoria Imagen Menor')
    
    # Leyenda
    ax1.legend(fontsize=10, loc='upper right', framealpha=0.9, facecolor='#E8E8E8', edgecolor='black')
    
    # === CONFIGURAR SUBPLOT DE MAGNIFICACIÓN (ABAJO) ===
    ax2.set_facecolor('#E8E8E8')
    ax2.set_xlim(lens_data.t[0], lens_data.t[-1])
    ax2.set_ylim(0, max(lens_data.magnification) * 1.1)  # Y desde 0
    ax2.set_xlabel(r'Tiempo (días)', fontsize=12, fontweight='bold', color='black')
    ax2.set_ylabel(r'Magnificación', fontsize=12, fontweight='bold', color='black')
    ax2.tick_params(labelsize=10, colors='black')
    ax2.grid(True, alpha=0.3, color='gray')
    
    # Curvas de magnificación completas (líneas de fondo)
    ax2.plot(lens_data.t, lens_data.magnification, color='gray', alpha=0.4, 
             linewidth=4, linestyle='-', label='Magnificación Total')
    ax2.plot(lens_data.t, lens_data.magnification_partial['image1'], color='green', alpha=0.4, 
             linewidth=4, linestyle='-', label='Magnificación Mayor')
    ax2.plot(lens_data.t, lens_data.magnification_partial['image2'], color='red', alpha=0.4, 
             linewidth=4, linestyle='-', label='Magnificación Menor')
    
    ax2.legend(fontsize=10, loc='upper right', framealpha=0.9, facecolor='#E8E8E8', edgecolor='black')
    
    # === ELEMENTOS ANIMADOS ===
    # Puntos actuales (destacados)
    source_current, = ax1.plot([], [], 'o', color='black', markersize=8, zorder=15, 
                               markeredgecolor='blue', markeredgewidth=2)
    image1_current, = ax1.plot([], [], 'o', color='green', markersize=8, zorder=15,
                               markeredgecolor='black', markeredgewidth=1)
    image2_current, = ax1.plot([], [], 'o', color='red', markersize=8, zorder=15,
                               markeredgecolor='black', markeredgewidth=1)
    
    # Estelas de puntos anteriores
    source_trail, = ax1.plot([], [], 'o', color='gray', alpha=0.6, markersize=4, zorder=12)
    image1_trail, = ax1.plot([], [], 'o', color='green', alpha=0.6, markersize=4, zorder=12)
    image2_trail, = ax1.plot([], [], 'o', color='red', alpha=0.6, markersize=4, zorder=12)
    
    # Línea vertical en el plot de magnificación
    time_line = ax2.axvline(x=lens_data.t[0], color='purple', linewidth=5, alpha=0.8, zorder=10)
    
    # Puntos actuales en el plot de magnificación
    mag_total_point, = ax2.plot([], [], 'o', color='black', markersize=8, zorder=15,
                                markeredgecolor='blue', markeredgewidth=2)
    mag_image1_point, = ax2.plot([], [], 'o', color='green', markersize=8, zorder=15,
                                 markeredgecolor='black', markeredgewidth=1)
    mag_image2_point, = ax2.plot([], [], 'o', color='red', markersize=8, zorder=15,
                                 markeredgecolor='black', markeredgewidth=1)
    
    # Texto con información del tiempo actual (sin índice)
    time_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, fontsize=12, 
                         color='blue', fontweight='bold', verticalalignment='top',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor='#E8E8E8', alpha=0.8, edgecolor='black'))
    
    def animate(frame_num):
        if frame_num >= len(frame_indices):
            return []
        
        idx = frame_indices[frame_num]
        current_time = lens_data.t[idx]
        
        # Actualizar puntos actuales en trayectorias
        source_current.set_data([lens_data.source_pos[idx, 0]], [lens_data.source_pos[idx, 1]])
        image1_current.set_data([lens_data.image_pos['image1'][idx, 0]], [lens_data.image_pos['image1'][idx, 1]])
        image2_current.set_data([lens_data.image_pos['image2'][idx, 0]], [lens_data.image_pos['image2'][idx, 1]])
        
        # Actualizar estelas (trail_length puntos anteriores)
        start_trail = max(0, idx - trail_length)
        trail_indices = range(start_trail, idx)
        
        if len(trail_indices) > 0:
            source_trail.set_data(lens_data.source_pos[trail_indices, 0], 
                                  lens_data.source_pos[trail_indices, 1])
            image1_trail.set_data(lens_data.image_pos['image1'][trail_indices, 0], 
                                  lens_data.image_pos['image1'][trail_indices, 1])
            image2_trail.set_data(lens_data.image_pos['image2'][trail_indices, 0], 
                                  lens_data.image_pos['image2'][trail_indices, 1])
        
        # Actualizar línea vertical en magnificación
        time_line.set_xdata([current_time, current_time])
        
        # Actualizar puntos en plot de magnificación
        mag_total_point.set_data([current_time], [lens_data.magnification[idx]])
        mag_image1_point.set_data([current_time], [lens_data.magnification_partial['image1'][idx]])
        mag_image2_point.set_data([current_time], [lens_data.magnification_partial['image2'][idx]])
        
        # Actualizar texto con tiempo actual (sin índice)
        time_text.set_text(f'Tiempo: {current_time:.2f} días')
        
        return [source_current, image1_current, image2_current, source_trail, image1_trail, 
                image2_trail, time_line, mag_total_point, mag_image1_point, mag_image2_point, time_text]
    
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
# Crear el directorio si no existe
output_dir = "./Simulacion/Images/SingleLense"
os.makedirs(output_dir, exist_ok=True)

print("=== Generación de GIF Animado - Lente Simple ===")

# Definir parámetros de la lente
lens_params = SingleLens_Data(
    t_E=10.0,    # Tiempo de Einstein en días
    u_0=0.3,     # Parámetro de impacto mínimo (más pequeño para mayor magnificación)
    num_points=500  # Reducido para GIF más fluido
)

print(f"Parámetros de la lente:")
print(f"  - Tiempo de Einstein (t_E): {lens_params.t_E} días")
print(f"  - Parámetro de impacto (u_0): {lens_params.u_0}")
print(f"  - Número de puntos: {lens_params.num_points}")

# Generar GIF animado
gif_path = os.path.join(output_dir, 'SingleLense_animated_gray.gif')

plot_single_lens_animated_gif(
    lens_data=lens_params,
    save_path=gif_path,
    fps=15,  # Frames por segundo
    duration_seconds=8.0,  # Duración total del GIF
    trail_length=30,  # Longitud de la estela de puntos
    figsize=(12, 16)  # Tamaño de la figura
)

print("=== GIF Animado Completado ===")
print(f"Archivo guardado en: {gif_path}")
