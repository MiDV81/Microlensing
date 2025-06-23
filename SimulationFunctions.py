"""
Codigo que contiene las funciones usadas a lo largo de la simulacion de las curvas de luz, de las imagenes generadas y de las curvas críticas/caústicas de un sistema de lentes gravitacionales.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Union, Optional
from dataclasses import dataclass

@dataclass
class SingleLens_Data:
    """
    Dataclass para almacenar todos los datos de una lente gravitacional simple.
    
    Attributes:
        # Parámetros de entrada
        t_E (float): Tiempo de Einstein en días
        u_0 (float): Parámetro de impacto mínimo
        t_0 (float): Tiempo de máximo acercamiento
        num_points (int): Número de puntos para el cálculo
        
        # Arrays calculados
        t (np.ndarray): Array de tiempos
        u (np.ndarray): Distancia fuente-lente en el plano de la lente
        source_pos (np.ndarray): Posiciones de la fuente [x, y]
        
        # Diccionarios para posiciones de imágenes
        image_pos (Dict[str, np.ndarray]): Posiciones de las imágenes
            - 'image1': posiciones de la imagen positiva
            - 'image2': posiciones de la imagen negativa
        
        # Magnificaciones
        magnification (np.ndarray): Magnificación total
        magnification_partial (Dict[str, np.ndarray]): Magnificaciones individuales
            - 'image1': magnificación de la imagen positiva
            - 'image2': magnificación de la imagen negativa
    """
    # Parámetros de entrada
    t_E: float
    u_0: float
    t_0: float = 0.0
    num_points: int = 1_000
    
    # Arrays calculados (inicializados como None)
    t: np.ndarray = np.ndarray([])
    u: np.ndarray = np.ndarray([])
    source_pos: np.ndarray = np.ndarray([])

    # Diccionarios para datos múltiples
    image_pos: Dict[str, np.ndarray] = {}
    magnification: np.ndarray = np.ndarray([])
    magnification_partial: Dict[str, np.ndarray] = {}


def magnification_single_lens(lens_data: SingleLens_Data) -> SingleLens_Data:
    """
    Calcula la magnificación de una lente simple y actualiza el dataclass.
    
    Args:
        lens_data (SingleLens_Data): Parámetros de entrada de la lente
    
    Returns:
        SingleLens_Data: Dataclass con los datos de magnificación calculados
    """
    # Crear array de tiempos
    t_ini: float = lens_data.t_0 - 1.5 * lens_data.t_E
    t_fin: float = lens_data.t_0 + 1.5 * lens_data.t_E
    lens_data.t = np.linspace(t_ini, t_fin, lens_data.num_points)

    # Calcular distancia fuente-lente en el plano de la lente
    lens_data.u = np.sqrt(lens_data.u_0**2 + ((lens_data.t - lens_data.t_0)/lens_data.t_E)**2)
    
    # Calcular magnificaciones
    lens_data.magnification = (lens_data.u**2 + 2)/(lens_data.u * np.sqrt(lens_data.u**2 + 4))
    lens_data.magnification_partial['image1'] = ((lens_data.u**2 + 2)/(lens_data.u * np.sqrt(lens_data.u**2 + 4)) + 1)/2
    lens_data.magnification_partial['image2'] = ((lens_data.u**2 + 2)/(lens_data.u * np.sqrt(lens_data.u**2 + 4)) - 1)/2
    
    return lens_data

def trajectory_single_lens_trajectories(lens_data: SingleLens_Data) -> SingleLens_Data:
    """
    Calcula las trayectorias de la fuente e imágenes para una lente simple.
    
    Args:
        lens_data (SingleLens_Data): Parámetros de entrada de la lente
    
    Returns:
        SingleLens_Data: Dataclass con todos los datos calculados
    """
    # Primero calcular las magnificaciones
    lens_data = magnification_single_lens(lens_data)
    
    # Crear vector de posición de la fuente
    u0_vec = np.full_like(lens_data.t, lens_data.u_0)
    lens_data.source_pos = np.column_stack(((lens_data.t - lens_data.t_0)/lens_data.t_E, u0_vec))
    
    # Calcular posiciones de las imágenes
    y1 = lens_data.u + np.sqrt(lens_data.u**2 + 4)/2
    y2 = lens_data.u - np.sqrt(lens_data.u**2 + 4)/2
    
    lens_data.image_pos['image1'] = lens_data.source_pos * (y1/lens_data.u)[:, np.newaxis]
    lens_data.image_pos['image2'] = lens_data.source_pos * (y2/lens_data.u)[:, np.newaxis]
    
    return lens_data

def plot_single_lens_complete(lens_data: SingleLens_Data, save_path=None):
    """
    Plotea tanto las trayectorias como las magnificaciones para una lente simple.
    
    Args:
        lens_data (SingleLens_Data): Parámetros de la lente
        save_path (str): Ruta para guardar la imagen (opcional)
    """
    # Calcular todos los datos si no están calculados
    if lens_data.t is None or lens_data.image_pos == {}:
        lens_data = trajectory_single_lens_trajectories(lens_data)
    
    # Crear figura con dos subplots verticales
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16), height_ratios=[1, 0.6], sharex=True)
    
    # Configurar puntos de color para la trayectoria
    step = 100
    num_color_points = len(lens_data.source_pos) // step
    cmap = plt.cm.plasma
    colors = [cmap(i/num_color_points) for i in range(num_color_points)]
    
    # Plot superior (Trayectorias)
    ax1.plot(lens_data.source_pos[:,0], lens_data.source_pos[:,1], 'g-', label='Fuente', alpha=0.8, markersize=1)
    ax1.plot(lens_data.image_pos['image1'][:,0], lens_data.image_pos['image1'][:,1], 'r-', label='Imagen 1', alpha=0.8, markersize=1)
    ax1.plot(lens_data.image_pos['image2'][:,0], lens_data.image_pos['image2'][:,1], 'm-', label='Imagen 2', alpha=0.8, markersize=1)
    
    # Plotear puntos con gradiente de color
    for i in range(0, len(lens_data.source_pos), step):
        idx = i // step
        if idx < len(colors):
            ax1.plot(lens_data.source_pos[i,0], lens_data.source_pos[i,1], 'o', color=colors[idx], markersize=5)
            ax1.plot(lens_data.image_pos['image1'][i,0], lens_data.image_pos['image1'][i,1], 's', color=colors[idx], markersize=5)
            ax1.plot(lens_data.image_pos['image2'][i,0], lens_data.image_pos['image2'][i,1], '^', color=colors[idx], markersize=5)
    
    # Agregar anillo de Einstein
    circle = plt.Circle((0, 0), 1, fill=False, color='red', linestyle='--', label='Anillo de Einstein')
    ax1.add_patch(circle)
    
    # Lente en el centro
    ax1.plot(0, 0, 'k.', markersize=10, label='Lente', zorder=5)
    
    # Configurar plot superior
    ax1.legend(fontsize=10, loc='upper right')
    ax1.axis('equal')
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_ylabel(r'$\mathbf{y (\theta_\mathrm{E})}$', fontsize=12, fontweight='bold')
    ax1.set_title(r'Trayectorias de la Fuente e Imágenes', fontsize=14, pad=15, fontweight="bold")
    ax1.tick_params(labelsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot inferior (Magnificación)
    ax2.plot(lens_data.t, lens_data.magnification, 'g-', label='Magnificación Total', linewidth=2)
    ax2.plot(lens_data.t, lens_data.magnification_partial['image1'], 'r-', label='Magnificación +', linewidth=1.5)
    ax2.plot(lens_data.t, lens_data.magnification_partial['image2'], 'k-', label='Magnificación -', linewidth=1.5)
    ax2.set_xlabel(r'Tiempo (días)', fontsize=12, fontweight='bold')
    ax2.set_ylabel(r'Magnificación', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=10)
    
    # Ajustar espaciado entre subplots
    plt.subplots_adjust(hspace=0.1)
    
    # Guardar si se especifica ruta
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

# Ejemplo de uso:
lens_params = SingleLens_Data(
    t_E=10,
    u_0=0.5,
    t_0=0
)

# Calcular solo las trayectorias
lens_data = trajectory_single_lens_trajectories(lens_params)

# Acceder a los datos calculados
print(f"Magnificación máxima: {np.max(lens_data.magnification)}")

# O plotear todo junto
plot_single_lens_complete(lens_params, save_path='MuLens/Images/single_lens_complete.pdf')