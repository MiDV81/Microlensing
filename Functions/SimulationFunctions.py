"""
Codigo que contiene las funciones usadas a lo largo de la simulacion de las curvas de luz, de las imagenes generadas y de las curvas críticas/caústicas de un sistema de lentes gravitacionales.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import Dict, Union, Optional
from dataclasses import dataclass, field

############################################################################################
"""Single Lens Functions and Data Class"""
############################################################################################
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
    image_pos: Dict[str, np.ndarray] = field(default_factory=dict)
    magnification: np.ndarray = np.ndarray([])
    magnification_partial: Dict[str, np.ndarray] = field(default_factory=dict)


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

def trajectory_single_lens(lens_data: SingleLens_Data) -> SingleLens_Data:
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

#############################################################################################
"""Binary Lens Functions and Data Class"""
#############################################################################################

@dataclass
class BinaryLens_Data:
    """
    Dataclass para almacenar todos los datos de un sistema de lentes binarias.
    
    Attributes:
        # Parámetros de entrada
        m_t (float): Masa total del sistema
        m_d (float): Diferencia de masa entre lentes
        z1 (float): Posición de la primera lente (segunda lente en -z1)
        
        # Masas individuales calculadas
        m1 (float): Masa de la primera lente
        m2 (float): Masa de la segunda lente
        z2 (float): Posición de la segunda lente
        
        # Datos de la fuente
        zeta (complex): Posición de la fuente en el plano complejo
        
        # Soluciones de la ecuación de lentes
        image_positions (np.ndarray): Posiciones de las imágenes
        magnifications (np.ndarray): Magnificaciones de cada imagen
        total_magnification (float): Magnificación total
    """
    # Parámetros de entrada
    m_t: float
    m_d: float
    z1: float
    
    # Masas y posiciones calculadas
    m1: float = field(init=False)
    m2: float = field(init=False)
    z2: float = field(init=False)
    
    # Datos de la fuente
    zeta: complex = 0.0 + 0.0j
    
    # Soluciones
    image_positions: np.ndarray = field(default_factory=lambda: np.array([]))
    magnifications: np.ndarray = field(default_factory=lambda: np.array([]))
    total_magnification: float = 0.0
    
    def __post_init__(self):
        """Calcular masas individuales y posición de la segunda lente"""
        self.m1 = (self.m_t + self.m_d) / 2
        self.m2 = (self.m_t - self.m_d) / 2
        self.z2 = -self.z1

def coefficients_lens_equation_binary_lense(binary_data: BinaryLens_Data) -> np.ndarray:
    """
    Calcula los coeficientes del polinomio de la ecuación de lentes binarias.
    
    Args:
        binary_data (BinaryLens_Data): Datos del sistema de lentes binarias
    
    Returns:
        np.ndarray: Array de coeficientes [c5, c4, c3, c2, c1, c0]
    """
    zeta = binary_data.zeta
    z_1 = binary_data.z1
    m_t = binary_data.m_t
    m_d = binary_data.m_d
    
    # Conjugado complejo de zeta
    zeta_conj = np.conjugate(zeta)
    
    # Calcular coeficientes
    c5 = z_1**2 - zeta_conj**2
    
    c4 = zeta*zeta_conj**2 - zeta*z_1**2 - m_t*zeta_conj + m_d*z_1
    
    c3 = 2*m_t*zeta*zeta_conj + 2*zeta_conj**2*z_1**2 - 2*m_d*zeta_conj*z_1 - 2*z_1**4
    
    c2 = zeta*(-2*zeta_conj**2*z_1**2 + 2*m_d*zeta_conj*z_1 + m_t**2 + 2*z_1**4) \
         - m_t*m_d*z_1 - 2*m_d*z_1**3
    
    c1 = -2*m_t*zeta*zeta_conj*z_1**2 + 2*m_t*m_d*zeta*z_1 \
         - zeta_conj**2*z_1**4 + 2*m_d*zeta_conj*z_1**3 \
         - m_d**2*z_1**2 + z_1**6 - m_t**2*z_1**2
    
    c0 = (zeta_conj**2*z_1**4 - 2*m_d*zeta_conj*z_1**3 + m_d**2*z_1**2 - z_1**6)*zeta \
         + (m_t*zeta_conj*z_1**4 - m_t*m_d*z_1**3 + m_d*z_1**5)
    
    return np.array([c5, c4, c3, c2, c1, c0], dtype=np.complex128)

def solve_lens_equation_binary_lense(coefficients: np.ndarray) -> np.ndarray:
    """
    Resuelve el polinomio de la ecuación de lentes usando numpy.roots().
    
    Args:
        coefficients (np.ndarray): Coeficientes del polinomio en orden descendente
    
    Returns:
        np.ndarray: Array de raíces complejas
    """
    coeffs = np.array(coefficients, dtype=np.complex128)    
    roots = np.roots(coeffs)    
    return roots

def verify_lens_equation_binary_lense(z: complex, binary_data: BinaryLens_Data, tolerance: float = 1e-10) -> bool:
    """
    Verifica si una solución satisface la ecuación de lentes binarias.
    
    Args:
        z (complex): Posición de la imagen a verificar
        binary_data (BinaryLens_Data): Datos del sistema de lentes
        tolerance (float): Tolerancia máxima permitida
    
    Returns:
        bool: True si la ecuación se satisface dentro de la tolerancia
    """
    z_conj = np.conjugate(z)
    zeta = binary_data.zeta
    z1 = binary_data.z1
    z2 = binary_data.z2
    m1 = binary_data.m1
    m2 = binary_data.m2
    
    # Lado derecho de la ecuación de lentes
    right_side = z - m1/(z_conj - z1) - m2/(z_conj - z2)
    
    # Verificar si la ecuación se satisface dentro de la tolerancia
    return abs(zeta - right_side) < tolerance

def solve_binary_lens_equation(binary_data: BinaryLens_Data) -> BinaryLens_Data:
    """
    Resuelve la ecuación de lentes binarias para encontrar todas las posiciones de imágenes válidas.
    
    Args:
        binary_data (BinaryLens_Data): Datos del sistema de lentes binarias
    
    Returns:
        BinaryLens_Data: Datos actualizados con las posiciones de imágenes encontradas
    """
    # Calcular coeficientes del polinomio
    coefficients = coefficients_lens_equation_binary_lense(binary_data)

    # Obtener todas las soluciones potenciales
    all_solutions = solve_lens_equation_binary_lense(coefficients)

    # Filtrar soluciones válidas
    valid_solutions = []
    for z in all_solutions:
        if verify_lens_equation_binary_lense(z, binary_data):
            valid_solutions.append(z)
    
    # Actualizar el dataclass con las soluciones válidas
    binary_data.image_positions = np.array(valid_solutions)
    
    return binary_data

def calculate_binary_magnification(z: complex, binary_data: BinaryLens_Data) -> float:
    """
    Calcula la magnificación para una posición de imagen dada.
    
    Args:
        z (complex): Posición de la imagen
        binary_data (BinaryLens_Data): Datos del sistema de lentes
    
    Returns:
        float: Magnificación de la imagen
    """
    z_conj = np.conjugate(z)
    z1 = binary_data.z1
    m1 = binary_data.m1
    m2 = binary_data.m2
    
    # Calcular A = m1/(z_conj - z1)^2 + m2/(z_conj + z1)^2
    A = m1/((z_conj - z1)**2) + m2/((z_conj + z1)**2)
    
    # Calcular magnificación = 1/|1-|A|^2|
    return 1/abs(1 - abs(A)**2)

def calculate_total_binary_magnification(binary_data: BinaryLens_Data) -> BinaryLens_Data:
    """
    Calcula la magnificación total y las magnificaciones individuales.
    
    Args:
        binary_data (BinaryLens_Data): Datos del sistema con posiciones de imágenes
    
    Returns:
        BinaryLens_Data: Datos actualizados con magnificaciones calculadas
    """
    if len(binary_data.image_positions) == 0:
        binary_data.magnifications = np.array([])
        binary_data.total_magnification = 0.0
        return binary_data
    
    # Calcular magnificaciones individuales
    individual_mags = []
    for z in binary_data.image_positions:
        mag = calculate_binary_magnification(z, binary_data)
        individual_mags.append(mag)
    
    # Actualizar el dataclass
    binary_data.magnifications = np.array(individual_mags)
    binary_data.total_magnification = sum(individual_mags)
    
    return binary_data

# Ejemplo de uso:
def solve_complete_binary_system(m_t: float, m_d: float, z1: float, zeta: complex) -> BinaryLens_Data:
    """
    Función de conveniencia para resolver completamente un sistema de lentes binarias.
    
    Args:
        m_t (float): Masa total
        m_d (float): Diferencia de masa
        z1 (float): Posición de la primera lente
        zeta (complex): Posición de la fuente
    
    Returns:
        BinaryLens_Data: Sistema completamente resuelto
    """
    # Crear el dataclass
    binary_system = BinaryLens_Data(m_t=m_t, m_d=m_d, z1=z1, zeta=zeta)
    
    # Resolver la ecuación de lentes
    binary_system = solve_binary_lens_equation(binary_system)
    
    # Calcular magnificaciones
    binary_system = calculate_total_binary_magnification(binary_system)
    
    return binary_system
# First, functions that work with the equations of the binary lens system
def solve_critical_curve(phi, z1, m1, m2, tolerance=1e-10):
    """
    Solve the critical curve polynomial equation for a given phi
    
    Parameters:
    phi : float
        Angle in radians
    z1 : float
        Position of first lens (second lens at -z1)
    m1, m2 : float
        Masses of the lenses
    tolerance : float
        Tolerance for filtering solutions
    
    Returns:
    array : Complex array of valid solutions
    """
    exp_iphi = np.exp(-1j * phi)
    
    c4 = exp_iphi
    c3 = 0
    c2 = -2 * z1**2 * exp_iphi - (m1 + m2)
    c1 = 2 * z1 * (m2 - m1)
    c0 = -z1**2 * (m1 + m2) + z1**4 * exp_iphi
    
    coeffs = [c4, c3, c2, c1, c0]
    
    solutions = np.roots(coeffs)
    valid_sols = solutions[np.abs(np.polyval(coeffs, solutions)) < tolerance]
    
    return valid_sols

def map_to_source_plane(z, z1, m1, m2):
    """
    Map points from lens plane to source plane using lens equation
    """
    z2 = -z1
    z_conj = np.conjugate(z)
    return z - m1/(z_conj - z1) - m2/(z_conj - z2)

def calculate_caustics(num_points=1000, z1=2.0, m1=0.5, m2=0.5):
    """
    Calculate caustic curves for binary lens system
    """
    phi_values = np.linspace(0, 2*np.pi, num_points)
    critical_points = []
    caustic_points = []
    
    for phi in phi_values:
        # Get critical curve points
        crit_sols = solve_critical_curve(phi, z1, m1, m2)
        critical_points.extend(crit_sols)
        
        # Map to source plane to get caustics
        caustic_sols = [map_to_source_plane(z, z1, m1, m2) for z in crit_sols]
        caustic_points.extend(caustic_sols)
    
    return np.array(critical_points), np.array(caustic_points)

def solve_complex_polynomial(coefficients):
    """
    Solve a complex polynomial equation using numpy.roots()
    
    Parameters:
    coefficients : array_like of complex numbers
        Polynomial coefficients in descending order
        For (a+bi)z^5 + (c+di)z^4 + ... + (k+li), input [(a+bi), (c+di), ..., (k+li)]
    
    Returns:
    array : Complex array of roots
    """
    # Convert coefficients to complex numbers if they aren't already
    coeffs = np.array(coefficients, dtype=np.complex128)
    
    # Solve the polynomial
    roots = np.roots(coeffs)
    
    return roots


def calculate_coefficients(zeta, z_1, m_t, m_d):
    """
    Calculate coefficients for the binary lens polynomial equation
    
    Parameters:
    zeta : complex
        Source position in complex plane
    z_1 : float
        Position of first lens (real number)
    m_t : float
        Total mass of the system
    m_d : float
        Mass difference between lenses
    
    Returns:
    array : Complex array of coefficients [c5, c4, c3, c2, c1, c0]
    """
    # Complex conjugate of zeta
    zeta_conj = np.conjugate(zeta)
    
    # Calculate coefficients
    c5 = z_1**2 - zeta_conj**2
    
    c4 = zeta*zeta_conj**2 - zeta*z_1**2 - m_t*zeta_conj + m_d*z_1
    
    c3 = 2*m_t*zeta*zeta_conj + 2*zeta_conj**2*z_1**2 - 2*m_d*zeta_conj*z_1 - 2*z_1**4
    
    c2 = zeta*(-2*zeta_conj**2*z_1**2 + 2*m_d*zeta_conj*z_1 + m_t**2 + 2*z_1**4) \
         - m_t*m_d*z_1 - 2*m_d*z_1**3
    
    c1 = -2*m_t*zeta*zeta_conj*z_1**2 + 2*m_t*m_d*zeta*z_1 \
         - zeta_conj**2*z_1**4 + 2*m_d*zeta_conj*z_1**3 \
         - m_d**2*z_1**2 + z_1**6 - m_t**2*z_1**2
    
    c0 = (zeta_conj**2*z_1**4 - 2*m_d*zeta_conj*z_1**3 + m_d**2*z_1**2 - z_1**6)*zeta \
         + (m_t*zeta_conj*z_1**4 - m_t*m_d*z_1**3 + m_d*z_1**5)
    
    return np.array([c5, c4, c3, c2, c1, c0], dtype=np.complex128)

def check_lens_equation(z, zeta, z1, z2, m1, m2, tolerance=1e-10):
    """
    Check if a solution satisfies the binary lens equation
    
    Parameters:
    z : complex
        Image position to check
    zeta : complex
        Source position
    z1, z2 : float
        Lens positions
    m1, m2 : float
        Lens masses
    tolerance : float
        Maximum allowed difference between left and right sides
    
    Returns:
    bool : True if equation is satisfied within tolerance
    """
    z_conj = np.conjugate(z)
    # Right side of lens equation
    right_side = z - m1/(z_conj - z1) - m2/(z_conj - z2)
    # Check if equation is satisfied within tolerance
    return abs(zeta - right_side) < tolerance

# Example usage:
def find_image_positions(zeta, z_1, m_t, m_d):
    """
    Find image positions by solving the complex polynomial and filtering valid solutions
    """
    # Calculate masses from total mass and mass difference
    m1 = (m_t + m_d)/2
    m2 = (m_t - m_d)/2
    z_m2 = -z_1  # Second lens position
    
    # Get all potential solutions
    coeffs = calculate_coefficients(zeta, z_1, m_t, m_d)
    solutions = solve_complex_polynomial(coeffs)
    
    # Filter valid solutions
    valid_solutions = [z for z in solutions if check_lens_equation(z, zeta, z_1, z_m2, m1, m2)]

    return np.array(valid_solutions)

def calculate_magnification(z, z1, m1, m2):
    """
    Calculate magnification for a given image position
    
    Parameters:
    z : complex
        Image position
    z1 : float
        First lens position (second lens at -z1)
    m1, m2 : float
        Masses of the lenses
    
    Returns:
    float : Magnification of the image
    """
    z_conj = np.conjugate(z)
    # Calculate A = m1/(z_conj - z1)^2 + m2/(z_conj + z1)^2
    A = m1/((z_conj - z1)**2) + m2/((z_conj + z1)**2)
    # Calculate magnification = 1/|1-|A|^2|
    return 1/abs(1 - abs(A)**2)

def calculate_total_magnification(image_positions, z1, m1, m2):
    """
    Calculate total magnification for all images
    
    Parameters:
    image_positions : array of complex
        Array of image positions
    z1 : float
        First lens position
    m1, m2 : float
        Masses of the lenses
    
    Returns:
    float : Total magnification
    array of float : Individual magnifications
    """
    individual_mags = [calculate_magnification(z, z1, m1, m2) for z in image_positions]
    return sum(individual_mags), individual_mags