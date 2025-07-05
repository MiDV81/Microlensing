import sys
import os
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Functions.SimulationFunctions import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def plot_single_lens(lens_data: SingleLens_Data, plot_type: str = 'both', save_path: Optional[str] = None):
    """
    Plotea las trayectorias y/o magnificaciones para una lente simple.
    
    Args:
        lens_data (SingleLens_Data): Parámetros de la lente
        plot_type (str): Tipo de plot - 'trajectory', 'magnification', o 'both' (default: 'both')
        save_path (str): Ruta para guardar la imagen (opcional)
    """
    # Validar tipo de plot
    valid_types = ['trajectory', 'magnification', 'both']
    if plot_type not in valid_types:
        raise ValueError(f"plot_type debe ser uno de: {valid_types}")
    print(f"Generando plot para Single Lense de tipo: {plot_type}")
    # Calcular todos los datos si no están calculados
    if lens_data.t.size == 0 or len(lens_data.image_pos) == 0:
        lens_data = trajectory_single_lens(lens_data)
    
    # Inicializar ax1 y ax2 como None
    ax1 = None
    ax2 = None
    # Configurar figura según el tipo de plot
    if plot_type == 'both':
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16), height_ratios=[1, 0.6])
        plot_trajectory = True
        plot_magnification = True
    elif plot_type == 'trajectory':
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))
        plot_trajectory = True
        plot_magnification = False
    else:  # magnification
        fig, ax2 = plt.subplots(1, 1, figsize=(12, 8))
        plot_trajectory = False
        plot_magnification = True
    
    # Plot de trayectorias
    if plot_trajectory and ax1 is not None:
        # Configurar puntos de color para la trayectoria
        step = 100
        num_color_points = len(lens_data.source_pos) // step
        cmap = plt.get_cmap('plasma')
        colors = [cmap(i/num_color_points) for i in range(num_color_points)]
        
        # Plot de líneas
        ax1.plot(lens_data.source_pos[:,0], lens_data.source_pos[:,1], 'k-', label='Fuente', alpha=0.8, markersize=1)
        ax1.plot(lens_data.image_pos['image1'][:,0], lens_data.image_pos['image1'][:,1], 'g-', label='Imagen Mayor', alpha=0.8, markersize=1)
        ax1.plot(lens_data.image_pos['image2'][:,0], lens_data.image_pos['image2'][:,1], 'r-', label='Imagen Menor', alpha=0.8, markersize=1)
        
        # Plotear puntos con gradiente de color
        for i in range(0, len(lens_data.source_pos), step):
            idx = i // step
            if idx < len(colors):
                ax1.plot(lens_data.source_pos[i,0], lens_data.source_pos[i,1], 'o', color=colors[idx], markersize=5)
                ax1.plot(lens_data.image_pos['image1'][i,0], lens_data.image_pos['image1'][i,1], 's', color=colors[idx], markersize=5)
                ax1.plot(lens_data.image_pos['image2'][i,0], lens_data.image_pos['image2'][i,1], '^', color=colors[idx], markersize=5)
        # Agregar anillo de Einstein
        circle = Circle((0, 0), 1, fill=False, color='blue', linestyle='--', label='Anillo de Einstein')
        ax1.add_patch(circle)
        ax1.add_patch(circle)
        
        # Lente en el centro 
        ax1.plot(0, 0, 'o', color='white', markeredgecolor='black', markeredgewidth=2, markersize=10, label='Lente', zorder=5)
        
        # Configurar plot de trayectorias
        ax1.legend(fontsize=12, loc='lower right')
        ax1.axis('equal')
        ax1.set_xlim(-2, 2)
        ax1.set_ylim(-1, 2)  
        ax1.set_ylabel(r'$\mathbf{y (\theta_\mathrm{E})}$', fontsize=12, fontweight='bold')
        ax1.set_title(r'Trayectorias de la Fuente e Imágenes', fontsize=14, pad=15, fontweight="bold")
        ax1.tick_params(labelsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Solo añadir xlabel si no hay plot de magnificación debajo
        if not plot_magnification:
            ax1.set_xlabel(r'$\mathbf{x (\theta_\mathrm{E})}$', fontsize=12, fontweight='bold')
    
    # Plot de magnificación
    if plot_magnification and ax2 is not None:
        ax2.plot(lens_data.t, lens_data.magnification, 'k-', label='Magnificación Total', linewidth=2)
        ax2.plot(lens_data.t, lens_data.magnification_partial['image1'], 'g--', label='Magnificación Mayor', linewidth=1.5)
        ax2.plot(lens_data.t, lens_data.magnification_partial['image2'], 'r--', label='Magnificación Menor', linewidth=1.5)
        ax2.set_xlabel(r'Tiempo (días)', fontsize=12, fontweight='bold')
        ax2.set_ylabel(r'Magnificación', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=10)
        
        # Solo añadir título si es el único plot
        if not plot_trajectory:
            ax2.set_title(r'Curva de Luz - Magnificación vs Tiempo', fontsize=14, pad=15, fontweight="bold")
    
    # Ajustar espaciado entre subplots si hay ambos
    if plot_type == 'both':
        plt.subplots_adjust(hspace=0.1)
    
    # Guardar si se especifica ruta
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot guardado en: {save_path}")
        plt.close()
    else:
        plt.show()

def plot_binary_lens_trajectory_interactive(binary_data: BinaryLens_Data, plot_type: str = 'both', save_path: Optional[str] = None):
    """
    Plotea la trayectoria de lentes binarias con slider interactivo.
    
    Args:
        binary_data (BinaryLens_Data): Datos del sistema con trayectoria calculada
        plot_type (str): Tipo de plot - 'trajectory', 'magnification', o 'both'
        save_path (str): Ruta para guardar la imagen (opcional)
    """
    from matplotlib.widgets import Slider
    
    # Validar tipo de plot
    valid_types = ['trajectory', 'magnification', 'both']
    if plot_type not in valid_types:
        raise ValueError(f"plot_type debe ser uno de: {valid_types}")
    
    print(f"=== Plot Interactivo Binary Lens ===")
    print(f"Tipo de plot seleccionado: {plot_type}")
    
    # Verificar que hay datos de trayectoria
    if len(binary_data.zeta) == 0 or len(binary_data.image_positions) == 0:
        print("ERROR: No hay datos de trayectoria calculados")
        raise ValueError("No hay datos de trayectoria. Ejecutar trajectory_binary_lens primero.")
    
    num_points = len(binary_data.zeta)
    print(f"Sistema: m1={binary_data.m1:.2f}, m2={binary_data.m2:.2f}, z1={binary_data.z1}, z2={binary_data.z2}")
        
    if plot_type == 'both':
        print("Creando layout con trayectorias y magnificación".ljust(50), end="\r")
        fig = plt.figure(figsize=(15, 12))
        # Subplot para trayectorias (arriba)
        ax1 = plt.subplot2grid((10, 1), (0, 0), rowspan=6)
        # Subplot para magnificación (medio)
        ax2 = plt.subplot2grid((10, 1), (6, 0), rowspan=3)
        # Slider (abajo)
        slider_ax = plt.subplot2grid((10, 1), (9, 0))
        plot_trajectory = True
        plot_magnification = True
    elif plot_type == 'trajectory':
        print("Creando layout solo con trayectorias".ljust(50), end="\r")
        fig = plt.figure(figsize=(12, 10))
        ax1 = plt.subplot2grid((10, 1), (0, 0), rowspan=9)
        slider_ax = plt.subplot2grid((10, 1), (9, 0))
        ax2 = None  # No hay subplot de magnificación
        plot_trajectory = True
        plot_magnification = False
    else:  # magnification
        print("Creando layout solo con magnificación".ljust(50), end="\r")
        fig = plt.figure(figsize=(12, 8))
        ax2 = plt.subplot2grid((10, 1), (0, 0), rowspan=9)
        slider_ax = plt.subplot2grid((10, 1), (9, 0))
        ax1 = None  # No hay subplot de trayectorias
        plot_trajectory = False
        plot_magnification = True
    
    # Inicializar elementos interactivos como None
    source_scatter = None
    images_scatter = None

    if plot_trajectory and ax1 is not None:
        print("Configurando plot de trayectorias...".ljust(50), end="\r")
        # Plot elementos estáticos
        if binary_data.z1 is not None:
            ax1.plot(binary_data.z1, 0, 'k*', markersize=15, label=f'Lente 1 (m={binary_data.m1:.2f})')
        else:
            raise ValueError("binary_data.z1 must not be None.")
        ax1.plot(binary_data.z2, 0, 'k*', markersize=15, label=f'Lente 2 (m={binary_data.m2:.2f})')
        
        # Trayectoria completa de la fuente
        source_real = [z.real for z in binary_data.zeta]
        source_imag = [z.imag for z in binary_data.zeta]
        ax1.plot(source_real, source_imag, 'b--', alpha=0.3, label='Trayectoria de la Fuente')
        
        # Calcular y plotear caústicas y curvas críticas si no están calculadas
        if not hasattr(binary_data, 'critical_points') or not hasattr(binary_data, 'caustic_points') or \
        binary_data.critical_points.size == 0 or binary_data.caustic_points.size == 0:
            print("Calculando curvas críticas y caústicas...".ljust(50), end="\r")
            binary_data = calculate_caustics_and_critical_curves_binary_lense(binary_data, 1000)
        
        # Plot curvas críticas y caústicas
        if len(binary_data.critical_points) > 0:
            ax1.scatter(binary_data.critical_points.real, binary_data.critical_points.imag, 
                    s=1, c='blue', alpha=0.6, label='Curvas Críticas')
            print(f"Curvas críticas añadidas: {len(binary_data.critical_points)} puntos")
        
        if len(binary_data.caustic_points) > 0:
            ax1.scatter(binary_data.caustic_points.real, binary_data.caustic_points.imag, 
                    s=1, c='red', alpha=0.6, label='Caústicas')
            print(f"Caústicas añadidas: {len(binary_data.caustic_points)} puntos")
        
        # Elementos interactivos (inicialmente vacíos)
        source_scatter = ax1.scatter([], [], c='g', marker='^', s=100, label='Fuente Actual', zorder=5)
        images_scatter = ax1.scatter([], [], c='orange', marker='o', s=50, label='Imágenes', zorder=4)
        
        # Configurar plot
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        ax1.set_xlim(-4, 4)
        ax1.set_ylim(-4, 4)
        ax1.set_xlabel(r'$\mathbf{Re(z)}$ $(\theta_\mathrm{E})$', fontsize=12, fontweight='bold')
        ax1.set_ylabel(r'$\mathbf{Im(z)}$ $(\theta_\mathrm{E})$', fontsize=12, fontweight='bold')
        ax1.set_title('Trayectoria de la Fuente e Imágenes', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        print("Configuración del plot de trayectorias completada".ljust(50), end="\r")
    else:
        # Si no hay trayectoria, crear objetos vacíos para evitar errores
        class DummyScatter:
            def set_offsets(self, val): pass
        source_scatter = DummyScatter()
        images_scatter = DummyScatter()
    
    # Inicializar mag_line como None
    mag_line = None

    # Plot de magnificación
    if plot_magnification and ax2 is not None:
        print("Configurando plot de magnificación...".ljust(50), end="\r")
        # Crear array de índices para el eje x
        indices = np.arange(len(binary_data.magnification))
        
        # Plot de magnificación completa
        ax2.plot(indices, binary_data.magnification, 'k-', label='Magnificación Total', linewidth=2)
        # Línea vertical para posición actual (inicialmente invisible)
        mag_line = ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Posición Actual')
        
        # Configurar plot
        ax2.set_xlabel('Índice de Posición', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Magnificación', fontsize=12, fontweight='bold')
        ax2.set_title('Curva de Magnificación', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
    
    # Crear slider
    print("Creando slider interactivo...".ljust(50), end="\r")
    slider = Slider(slider_ax, 'Posición', 0, num_points-1, valinit=0, valstep=1)
    
    def update(val):
        """Función para actualizar el plot cuando cambia el slider"""
        idx = int(slider.val)
        
        if plot_trajectory:
            # Actualizar posición de la fuente
            current_source = binary_data.zeta[idx]
            source_scatter.set_offsets([[current_source.real, current_source.imag]])
            
            # Actualizar posiciones de las imágenes
            current_images = binary_data.image_positions[idx]
            if len(current_images) > 0:
                image_coords = [[img.real, img.imag] for img in current_images]
                images_scatter.set_offsets(image_coords)
            else:
                images_scatter.set_offsets([])
        
        if plot_magnification and mag_line is not None:
            # Actualizar línea vertical en plot de magnificación
            mag_line.set_xdata([idx, idx])
        
        fig.canvas.draw_idle()
    
    # Conectar slider con función de actualización
    slider.on_changed(update)

    # Inicializar plot
    print("Inicializando plot...".ljust(50), end="\r")
    update(0)
    
    plt.tight_layout()
    
    # Guardar si se especifica ruta (solo guarda el estado inicial)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot interactivo guardado en: {save_path}".ljust(50))
    
    print("=== Plot interactivo listo para usar ===")
    plt.show()
    
def plot_binary_lens_caustics_grid(binary_systems: list[BinaryLens_Data], topology_labels: Optional[list] = None, 
                                  num_points: int = 1000, num_rows: int = 1,
                                  auto_limits: bool = True, save_path: Optional[str] = None):
    """
    Plotea curvas críticas y caústicas para múltiples sistemas de lentes binarias en una grilla.
    
    Args:
        binary_systems (list): Lista de objetos BinaryLens_Data
        topology_labels (list): Lista de etiquetas de topología (opcional). Si se proporciona y no está vacía, 
                               se mostrarán como etiquetas en la parte inferior. Siempre se mostrará q y d como título.
        num_points (int): Número de puntos para calcular las curvas
        num_rows (int): Número de filas en la grilla (default: 1)
        auto_limits (bool): Si True, ajusta los límites de los ejes a ±4*z1 para cada sistema (default: True)
        save_path (str): Ruta para guardar la imagen (opcional)
    """
    # Validar que se proporcionan sistemas
    if len(binary_systems) == 0:
        raise ValueError("Se requiere al menos un sistema de lentes binarias")
    
    # Validar que el número de sistemas es múltiplo del número de filas
    num_systems = len(binary_systems)
    if num_systems % num_rows != 0:
        print(f"Warning: El número de sistemas ({num_systems}) no es múltiplo del número de filas ({num_rows})")
    
    # Calcular número de columnas
    num_cols = num_systems // num_rows
    if num_systems % num_rows != 0:
        num_cols += 1  # Añadir una columna extra si hay resto
    
    print(f"Calculando y ploteando curvas críticas y caústicas para {num_systems} sistemas binarios en grilla {num_rows}x{num_cols}...")
    
    # Calcular curvas críticas y caústicas para cada sistema si no están calculadas
    for i, binary_data in enumerate(binary_systems):
        if not hasattr(binary_data, 'critical_points') or not hasattr(binary_data, 'caustic_points') or \
           binary_data.critical_points.size == 0 or binary_data.caustic_points.size == 0:
            print(f"Sistema {i+1}: m_t={binary_data.m_t}, m_d={binary_data.m_d}, z1={binary_data.z1}")
            binary_systems[i] = calculate_caustics_and_critical_curves_binary_lense(binary_data, num_points)
    
    # Crear figura con GridSpec para control preciso del layout
    from matplotlib.gridspec import GridSpec
    fig_width = 5 * num_cols
    fig_height = 4 * num_rows  # Reduced height per row
    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor('#E8E8E8')  # Set figure background
    
    # Ajustar espaciado - subplots tocándose horizontalmente, más espacio vertical si hay múltiples filas o etiquetas
    if num_rows > 1:
        gs = GridSpec(num_rows, num_cols, wspace=-0.3, hspace=0.15)  # No horizontal spacing, more vertical spacing for multiple rows
    elif topology_labels and len(topology_labels) > 0:
        gs = GridSpec(num_rows, num_cols, wspace=-0.3, hspace=0.1)  # No horizontal spacing, slightly more vertical spacing for bottom labels
    else:
        gs = GridSpec(num_rows, num_cols, wspace=-0.3, hspace=0.02)  # No horizontal spacing, minimal vertical spacing

    # Iterar sobre todos los sistemas binarios
    for i, binary_data in enumerate(binary_systems):
        # Obtener etiqueta de topología si está disponible
        topology = topology_labels[i] if topology_labels and i < len(topology_labels) else None
        
        # Calcular posición en la grilla
        row = i // num_cols
        col = i % num_cols
        
                # Crear subplot individual
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor('#E8E8E8')  # Set axes background
        
        # Añadir círculos negros para representar las posiciones de las lentes (PRIMERO)
        if binary_data.m1 is None or binary_data.m2 is None or binary_data.z1 is None or binary_data.z2 is None:
            raise ValueError("Lens masses m1, m2 and positions z1, z2 must not be None.")
        circle1 = Circle((float(binary_data.z1), 0), 0.1 * np.sqrt(float(binary_data.m1)), 
                        color='black', fill=True)
        circle2 = Circle((float(binary_data.z2), 0), 0.1 * np.sqrt(float(binary_data.m2)), 
                        color='black', fill=True)
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        
        # Plotear curvas críticas en azul (DESPUÉS DE LAS LENTES)
        if len(binary_data.critical_points) > 0:
            ax.scatter(binary_data.critical_points.real, binary_data.critical_points.imag, 
                      s=1, c='b', alpha=0.5, label='Critical Curves')
        
        # Plotear caústicas en rojo (DESPUÉS DE LAS LENTES)
        if len(binary_data.caustic_points) > 0:
            ax.scatter(binary_data.caustic_points.real, binary_data.caustic_points.imag, 
                      s=1, c='r', alpha=0.5, label='Caustics')
        
        # Añadir grilla sutil de fondo
        ax.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.9)
        
        # Establecer límites de los ejes
        if auto_limits:
            # Límites basados en z1 para cada sistema individual
            limit = 2 * abs(binary_data.z1)
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
        else:
            # Límites fijos para todos los subplots
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
        ax.set_aspect('equal')  # Mantener aspecto cuadrado
        
        # Ocultar etiquetas de los ejes para un aspecto más limpio
        ax.tick_params(labelbottom=False, labelleft=False)
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # Remover todos los bordes del subplot
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Calcular parámetros físicos del sistema
        q = binary_data.q  # Use the calculated q from the class
        d = 2 * abs(binary_data.z1)  # Separación total entre lentes
        
        # Añadir etiqueta de parámetros como título (siempre en la parte superior)
        title_str = f"q={q:.3f}, d={d:.1f}"  
        ax.set_title(title_str, fontsize=12, weight="bold")
        
        # Añadir etiqueta de topología en la parte inferior si está disponible
        if topology:
            from matplotlib.patches import Rectangle
            text_x, text_y = 0.5, -0.08  # Posición en la parte inferior
            bbox_props = dict(boxstyle="round,pad=0.3", facecolor='#E8E8E8', alpha=0.8, edgecolor='none')
            ax.text(text_x, text_y, topology, 
                    ha='center', va='bottom', transform=ax.transAxes, 
                    fontsize=12, color='black', weight="bold",
                    bbox=bbox_props)
    # Guardar la figura si se especifica una ruta
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot de curvas críticas y caústicas guardado en: {save_path}")
        plt.close()
    else:
        plt.show()

def plot_binary_lens_trajectory_static(binary_data: BinaryLens_Data, 
                                      plot_type: str = 'both',
                                      special_indices: Optional[list] = None,
                                      show_caustics: bool = True,
                                      num_caustic_points: int = 1000,
                                      save_path: Optional[str] = None):
    """
    Plotea un análisis estático completo de la trayectoria de lentes binarias con caústicas.
    
    Args:
        binary_data (BinaryLens_Data): Datos del sistema con trayectoria calculada
        plot_type (str): Tipo de plot - 'trajectory', 'magnification', o 'both' (default: 'both')
        special_indices (list): Índices específicos a destacar con colores únicos
        show_caustics (bool): Si mostrar curvas críticas y caústicas
        num_caustic_points (int): Número de puntos para calcular caústicas
        save_path (str): Ruta para guardar la imagen
    """
    # Validar tipo de plot
    valid_types = ['trajectory', 'magnification', 'both']
    if plot_type not in valid_types:
        raise ValueError(f"plot_type debe ser uno de: {valid_types}")
    
    print(f"=== Plot Estático Binary Lens - Tipo: {plot_type} ===")
    
    # Verificar que hay datos de trayectoria
    if len(binary_data.zeta) == 0 or len(binary_data.image_positions) == 0:
        print("ERROR: No hay datos de trayectoria calculados")
        raise ValueError("No hay datos de trayectoria. Ejecutar trajectory_binary_lens primero.")
    
    num_points = len(binary_data.zeta)
    
    # Calcular caústicas si se requieren y no están calculadas
    if show_caustics and plot_type in ['trajectory', 'both']:
        print("Calculando curvas críticas y caústicas...".ljust(50), end="\r")
        if not hasattr(binary_data, 'critical_points') or binary_data.critical_points.size == 0:
            binary_data = calculate_caustics_and_critical_curves_binary_lense(binary_data, num_caustic_points)
    
    # Configurar índices especiales si no se proporcionan
    if special_indices is None:
        special_indices = []
        # Seleccionar automáticamente puntos interesantes basado en picos de magnificación
        mags = np.array(binary_data.magnification)
        # Encontrar picos locales de magnificación
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(mags, height=np.percentile(mags, 80))
        
        # Añadir picos encontrados
        if len(peaks) > 0:
            special_indices.extend(peaks[:3].tolist())  # Máximo 3 picos
        
        # Completar con puntos distribuidos normalmente
        remaining_points = 5 - len(special_indices)
        if remaining_points > 0:
            step = num_points // remaining_points
            normal_indices = [i * step for i in range(remaining_points) if i * step < num_points]
            special_indices.extend(normal_indices)
        
        # Asegurar que no hay duplicados y ordenar
        special_indices = sorted(list(set(special_indices)))
        
        # Limitar a máximo 5 puntos
        special_indices = special_indices[:5]
    
    print(f"Índices especiales destacados: {special_indices}")
    
    # Configurar figura según el tipo de plot
    if plot_type == 'both':
        fig, (plot_ax, mag_ax) = plt.subplots(2, 1, figsize=(10, 12), 
                                             gridspec_kw={'height_ratios': [1.5, 1]})
        plot_trajectory = True
        plot_magnification = True
    elif plot_type == 'trajectory':
        fig, plot_ax = plt.subplots(1, 1, figsize=(10, 8))
        mag_ax = None
        plot_trajectory = True
        plot_magnification = False
    else:  # magnification
        fig, mag_ax = plt.subplots(1, 1, figsize=(10, 6))
        plot_ax = None
        plot_trajectory = False
        plot_magnification = True
    
   # === SUBPLOT DE TRAYECTORIAS ===
    if plot_trajectory and plot_ax is not None:
        print("Configurando plot de trayectorias...".ljust(50), end="\r")
        
        # Plot caústicas y curvas críticas como scatter points (unchanged)
        if show_caustics and hasattr(binary_data, 'critical_points') and len(binary_data.critical_points) > 0:
            plot_ax.scatter(binary_data.critical_points.real, binary_data.critical_points.imag, 
                           s=1, c='b', alpha=0.3)
            plot_ax.scatter(binary_data.caustic_points.real, binary_data.caustic_points.imag, 
                           s=1, c='r', alpha=0.3)
            
            # Add legend entries as lines
            plot_ax.plot([], [], 'b-', alpha=0.6, linewidth=2, label='Curvas Críticas')
            plot_ax.plot([], [], 'r-', alpha=0.6, linewidth=2, label='Caústicas')

        if binary_data.m1 is None or binary_data.m2 is None:
            raise ValueError("Lens masses m1 and m2 must not be None.")  
        lens1_size = 0.15 * binary_data.m1
        lens2_size = 0.15 * binary_data.m2
        if binary_data.z1 is None or binary_data.z2 is None:
            raise ValueError("Lens positions z1 and z2 must not be None.")
        lens1_circle = Circle((binary_data.z1, 0), lens1_size, 
                             facecolor='k', edgecolor='red', fill=True, zorder=10)
        lens2_circle = Circle((binary_data.z2, 0), lens2_size, 
                             facecolor='k', edgecolor='red', fill=True, zorder=10)
        plot_ax.add_patch(lens1_circle)
        plot_ax.add_patch(lens2_circle)
        # Raise error if any lens mass is None
              
        # Add lens labels to legend
        plot_ax.plot([], [], 'ko', markersize=8, label=f'Lentes')
        
        # Plot trayectoria completa de la fuente
        source_real = [z.real for z in binary_data.zeta]
        source_imag = [z.imag for z in binary_data.zeta]
        plot_ax.plot(source_real, source_imag, 'g--', alpha=0.3, 
                    label='Trayectoria de la Fuente', linewidth=2)
        
        # Add legend entries for source and images
        plot_ax.plot([], [], '^', color='white', markeredgecolor='black', 
                    markersize=10, markeredgewidth=1, label='Fuente en diferentes puntos')
        plot_ax.plot([], [], 'o', color='white', markeredgecolor='black', 
                    markersize=8, markeredgewidth=1, label='Imágenes producidas')
        
        
        # Configurar colormap para índices especiales
        viridis = plt.cm.get_cmap('viridis', len(special_indices))
        colors = [viridis(i) for i in range(len(special_indices))]
        
        # Plot todas las imágenes en el rango de índices especiales como puntos grises pequeños
        if special_indices:
            min_idx = min(special_indices)
            max_idx = max(special_indices)
            
            for sol_idx in range(min_idx, max_idx + 1):
                if sol_idx not in special_indices and sol_idx < len(binary_data.image_positions):
                    for sol in binary_data.image_positions[sol_idx]:
                        plot_ax.scatter(sol.real, sol.imag, c='gray', s=1, alpha=0.6, zorder=2)
        
        # Plot posiciones especiales con colores únicos (KEEP ORIGINAL COLORS)
        for i, idx in enumerate(special_indices):
            if idx < len(binary_data.zeta):
                color = colors[i]
                
                # Plot fuente en posición especial (keep original colored version)
                plot_ax.scatter(binary_data.zeta[idx].real, binary_data.zeta[idx].imag, 
                              c=[color], marker='^', s=120, zorder=5, 
                              edgecolors='black', linewidth=1)
                
                # Plot imágenes en posición especial (keep original colored version)
                for sol in binary_data.image_positions[idx]:
                    plot_ax.scatter(sol.real, sol.imag, c=[color], marker='o', 
                                  s=60, zorder=5, edgecolors='black', linewidth=0.5)
        
        # Configurar subplot de trayectorias con límites consistentes
        plot_ax.set_xlim(-2, 2)
        plot_ax.set_ylim(-1.25, 1.25)
        plot_ax.grid(True, alpha=0.3)
        plot_ax.set_aspect('equal')
        plot_ax.set_xlabel(r'$\mathbf{Re(z)}$ $(\theta_\mathrm{E})$', fontsize=12, fontweight='bold')
        plot_ax.set_ylabel(r'$\mathbf{Im(z)}$ $(\theta_\mathrm{E})$', fontsize=12, fontweight='bold')
        
        # Solo añadir título si es el único plot o si es el de arriba
        if plot_type == 'trajectory':
            plot_ax.set_title('Trayectoria con Curvas Críticas y Caústicas', 
                             fontsize=14, pad=15, fontweight="bold")
        
        # Legend in top-left corner
        plot_ax.legend(fontsize=10, loc='upper left')
    
    # === SUBPLOT DE MAGNIFICACIÓN ===
    if plot_magnification and mag_ax is not None:
        print("Configurando plot de magnificación...".ljust(50), end="\r")
        
        # Plot curva de magnificación completa
        t_normalized = np.linspace(0, 1, num_points)
        mag_ax.plot(t_normalized, binary_data.magnification, 'b-', 
                   linewidth=1.5, alpha=0.8, label='Magnificación Total')
        
        # Add legend entry for trajectory points (white circle with black border)
        mag_ax.plot([], [], 'o', color='white', markeredgecolor='black', 
                   markersize=8, markeredgewidth=1, label='Punto de la trayectoria')
        
        # Configurar colormap para índices especiales (mismos colores que arriba)
        if special_indices:
            viridis = plt.cm.get_cmap('viridis', len(special_indices))
            colors = [viridis(i) for i in range(len(special_indices))]
            
            # Destacar puntos especiales en la curva de magnificación (keep original colors, no labels)
            for i, idx in enumerate(special_indices):
                if idx < len(binary_data.magnification):
                    t_val = idx / (num_points - 1)
                    mag_ax.scatter(t_val, binary_data.magnification[idx], 
                                  c=[colors[i]], s=80, zorder=5, 
                                  edgecolors='black', linewidth=1)
        
        # Configurar subplot de magnificación
        mag_ax.set_xlim(0, 1)
        mag_ax.grid(True, alpha=0.3)
        mag_ax.set_xlabel(r'Posición Normalizada en la Trayectoria', fontsize=12, fontweight='bold')
        mag_ax.set_ylabel(r'Magnificación Total', fontsize=12, fontweight='bold')
        
        # Solo añadir título si es el único plot
        if plot_type == 'magnification':
            mag_ax.set_title('Curva de Magnificación', fontsize=14, pad=15, fontweight="bold")
            
        mag_ax.legend(fontsize=10, loc='upper left')
        
        # Configurar escala logarítmica si hay gran rango de magnificaciones
        mag_range = np.max(binary_data.magnification) / np.min(binary_data.magnification)
        if mag_range > 100:
            mag_ax.set_yscale('log')

    # Ajustar layout para asegurar mismo ancho
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot estático guardado en: {save_path}")
    
    print("=== Plot estático completado ===")
    plt.show()
