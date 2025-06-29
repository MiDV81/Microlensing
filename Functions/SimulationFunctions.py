"""
Codigo que contiene las funciones usadas a lo largo de la simulacion de las curvas de luz, de las imagenes generadas y de las curvas críticas/caústicas de un sistema de lentes gravitacionales.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import Dict, Union, Optional
from numpy.typing import NDArray
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

#############################################################################################
"""Binary Lens Functions and Data Class"""
#############################################################################################

@dataclass
class BinaryLens_Data:
    """
    Dataclass para almacenar todos los datos de un sistema de lentes binarias.
    
    Accepts any two of: m_t, m_d, m1, m2, q and calculates the rest automatically.
    Relationships: q = m2/m1, m_t = m1 + m2, m_d = m1 - m2
    
    Position can be defined either by z1 OR d:
    - z1 (float): Position of first lens (second lens at -z1)
    - d (float): Separation distance between lenses (z1 = d/2)
    
    Attributes:
        # Position parameters (provide either z1 OR d, not both)
        z1 (float, optional): Posición de la primera lente (segunda lente en -z1)
        d (float, optional): Separación entre lentes (z1 = d/2)
        
        # Parámetros de entrada (any two of these five)
        m_t (float, optional): Masa total del sistema
        m_d (float, optional): Diferencia de masa entre lentes  
        m1 (float, optional): Masa de la primera lente
        m2 (float, optional): Masa de la segunda lente
        q (float, optional): Ratio de masas q = m2/m1
        
        # Parámetros de trayectoria (opcionales)
        start_point (tuple): Punto inicial de la trayectoria (x, y)
        end_point (tuple): Punto final de la trayectoria (x, y)
        num_points (int): Número de puntos en la trayectoria
        
        # Datos de la fuente
        zeta (NDArray): Posiciones de la fuente en el plano complejo
        
        # Soluciones de la ecuación de lentes
        image_positions (np.ndarray): Posiciones de las imágenes
        magnification (np.ndarray): Magnificación total
        
        # Curvas críticas y caústicas
        critical_points (np.ndarray): Puntos de las curvas críticas
        caustic_points (np.ndarray): Puntos de las caústicas
    """
    # Position parameters (provide either z1 OR d, not both)
    z1: Optional[float] = None  # Position of first lens
    d: Optional[float] = None   # Separation distance between lenses
    
    # Mass parameters (any two of these five should be provided)
    m_t: Optional[float] = None  # Total mass
    m_d: Optional[float] = None  # Mass difference
    m1: Optional[float] = None   # Mass of first lens
    m2: Optional[float] = None   # Mass of second lens
    q: Optional[float] = None    # Mass ratio q = m2/m1
    
    # Parámetros de trayectoria (opcionales)
    start_point: Optional[tuple] = None
    end_point: Optional[tuple] = None
    num_points: int = 1000
    
    # Posición de la segunda lente (calculated)
    z2: float = field(init=False)
    
    # Datos de la fuente
    zeta: NDArray[np.complex128] = field(default_factory=lambda: np.array([]))
    
    # Soluciones
    image_positions: NDArray[np.complex128] = field(default_factory=lambda: np.array([]))
    magnification: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Curvas críticas y caústicas
    critical_points: NDArray[np.complex128] = field(default_factory=lambda: np.array([]))
    caustic_points: NDArray[np.complex128] = field(default_factory=lambda: np.array([]))
    
    def __post_init__(self):
        """Calculate all parameters from provided inputs and generate trajectory if needed"""
        self._calculate_position_parameters()
        self._calculate_mass_parameters()
        if self.z1 is None:
            raise ValueError("z1 must be set before assigning z2. Check position parameters.")
        self.z2 = -self.z1
        
        # Generar trayectoria si zeta está vacío pero se proporcionaron puntos de trayectoria
        if len(self.zeta) == 0 and self.start_point is not None and self.end_point is not None:
            # print(f"Generando trayectoria desde {self.start_point} hasta {self.end_point} con {self.num_points} puntos...")
            self._generate_trajectory()
    
    def __repr__(self) -> str:
        """Custom string representation for BinaryLens_Data"""
        return (f"BinaryLens_Data: "
                f"Mass Parameters: m_t={self.m_t:.3f}, m_d={self.m_d:.3f}, m1={self.m1:.3f}, m2={self.m2:.3f}, q={self.q:.3f} | "
                f"Positions: z1={self.z1:.3f}, z2={self.z2:.3f}, d={self.d:.3f} | "
                f"Source pos: len(zeta)={len(self.zeta)} | "
                f"Image pos: len(image_positions)={len(self.image_positions)} | "
                f"Magnification: len(magnification)={len(self.magnification)} | "
                f"Critical curves: len(critical_points)={len(self.critical_points)} | "
                f"Caustics: len(caustic_points)={len(self.caustic_points)}")
    
    def _calculate_position_parameters(self):
        """Calculate z1 and d from the provided position parameter"""
        # Count position parameters provided
        position_params = sum([
            self.z1 is not None,
            self.d is not None
        ])
        
        if position_params == 0:
            raise ValueError("Either z1 or d must be provided")
        elif position_params > 1:
            raise ValueError("Provide either z1 OR d, not both")
        
        # Calculate missing parameter
        if self.z1 is not None:
            # z1 provided, calculate d
            self.d = 2 * abs(self.z1)
            # print(f"Position parameters: z1={self.z1:.3f} provided, calculated d={self.d:.3f}")
        elif self.d is not None:
            # d provided, calculate z1
            if self.d <= 0:
                raise ValueError(f"Separation distance d must be positive: d={self.d}")
            self.z1 = self.d / 2
            # print(f"Position parameters: d={self.d:.3f} provided, calculated z1={self.z1:.3f}")
    
    def _calculate_mass_parameters(self):
        """Calculate all mass parameters from any two provided"""
        # Count how many parameters are provided
        provided_params = sum([
            self.m_t is not None,
            self.m_d is not None, 
            self.m1 is not None,
            self.m2 is not None,
            self.q is not None
        ])
        
        if provided_params < 2:
            raise ValueError("At least two mass parameters must be provided from: m_t, m_d, m1, m2, q")
        elif provided_params > 2:
            print(f"Warning: {provided_params} mass parameters provided. Using first two found for calculation.")
        
        # Case 1: m1 and m2 provided
        if self.m1 is not None and self.m2 is not None:
            self.m_t = self.m1 + self.m2
            self.m_d = self.m1 - self.m2
            self.q = self.m2 / self.m1
            
        # Case 2: m_t and m_d provided
        elif self.m_t is not None and self.m_d is not None:
            self.m1 = (self.m_t + self.m_d) / 2
            self.m2 = (self.m_t - self.m_d) / 2
            self.q = self.m2 / self.m1
            
        # Case 3: m_t and q provided
        elif self.m_t is not None and self.q is not None:
            self.m1 = self.m_t / (1 + self.q)
            self.m2 = self.q * self.m1
            self.m_d = self.m1 - self.m2
            
        # Case 4: m_d and q provided
        elif self.m_d is not None and self.q is not None:
            self.m1 = self.m_d / (1 - self.q)
            self.m2 = self.q * self.m1
            self.m_t = self.m1 + self.m2
            
        # Case 5: m1 and m_t provided
        elif self.m1 is not None and self.m_t is not None:
            self.m2 = self.m_t - self.m1
            self.m_d = self.m1 - self.m2
            self.q = self.m2 / self.m1
            
        # Case 6: m1 and m_d provided
        elif self.m1 is not None and self.m_d is not None:
            self.m2 = self.m1 - self.m_d
            self.m_t = self.m1 + self.m2
            self.q = self.m2 / self.m1
            
        # Case 7: m1 and q provided
        elif self.m1 is not None and self.q is not None:
            self.m2 = self.q * self.m1
            self.m_t = self.m1 + self.m2
            self.m_d = self.m1 - self.m2
            
        # Case 8: m2 and m_t provided
        elif self.m2 is not None and self.m_t is not None:
            self.m1 = self.m_t - self.m2
            self.m_d = self.m1 - self.m2
            self.q = self.m2 / self.m1
            
        # Case 9: m2 and m_d provided
        elif self.m2 is not None and self.m_d is not None:
            self.m1 = self.m2 + self.m_d
            self.m_t = self.m1 + self.m2
            self.q = self.m2 / self.m1
            
        # Case 10: m2 and q provided
        elif self.m2 is not None and self.q is not None:
            self.m1 = self.m2 / self.q
            self.m_t = self.m1 + self.m2
            self.m_d = self.m1 - self.m2
            
        else:
            raise ValueError("Invalid combination of mass parameters provided")
        
        # Validation checks
        if self.m1 <= 0 or self.m2 <= 0:
            raise ValueError(f"Masses must be positive: m1={self.m1:.3f}, m2={self.m2:.3f}")
        
        if self.q <= 0:
            raise ValueError(f"Mass ratio q must be positive: q={self.q:.3f}")
        
        # print(f"Mass parameters calculated: m1={self.m1:.3f}, m2={self.m2:.3f}, m_t={self.m_t:.3f}, m_d={self.m_d:.3f}, q={self.q:.3f}")
    
    def _generate_trajectory(self):
        """Genera la trayectoria entre start_point y end_point"""
        if self.start_point is None or self.end_point is None:
            raise ValueError("start_point y end_point deben estar definidos para generar trayectoria")
        
        # Crear array de posiciones de fuente a lo largo de la línea
        x = np.linspace(self.start_point[0], self.end_point[0], self.num_points)
        y = np.linspace(self.start_point[1], self.end_point[1], self.num_points)
        
        # Convertir a números complejos
        self.zeta = np.array([complex(x[i], y[i]) for i in range(self.num_points)])
    
    def images_magnification_calculation(self):
        """Calcula la ecuación de lentes y magnificaciones para toda la trayectoria"""
        if len(self.zeta) == 0:
            raise ValueError("No hay datos de trayectoria. Definir zeta o usar start_point/end_point.")
        
        # print("Calculando posiciones y magnificaciones de imágenes para toda la trayectoria...")
        lens_equation_binary_lense(self)        
        magnification_binary_lense(self)        
        return self

def coefficients_lens_equation_binary_point(zeta: complex, binary_data: BinaryLens_Data) -> np.ndarray:
    """
    Calcula los coeficientes del polinomio de la ecuación de lentes binarias para una sola posición de fuente.
    
    Args:
        zeta (complex): Posición de la fuente en el plano complejo
        binary_data (BinaryLens_Data): Datos del sistema de lentes binarias
    
    Returns:
        np.ndarray: Array de coeficientes [c5, c4, c3, c2, c1, c0]
    """
    z_1 = binary_data.z1
    m_t = binary_data.m_t
    m_d = binary_data.m_d

    # Check that z_1, m_t and m_d are not None
    if z_1 is None or m_t is None or m_d is None:
        raise ValueError("z1, m_t, and m_d must not be None in coefficients_lens_equation_binary_point")

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

def solve_lens_equation_binary_point(coefficients: np.ndarray) -> np.ndarray:
    """
    Resuelve el polinomio de la ecuación de lentes usando numpy.roots() para una sola fuente.
    
    Args:
        coefficients (np.ndarray): Coeficientes del polinomio en orden descendente
    
    Returns:
        np.ndarray: Array de raíces complejas
    """
    coeffs = np.array(coefficients, dtype=np.complex128)    
    roots = np.roots(coeffs)    
    return roots

def verify_lens_equation_binary_point(z: complex, zeta: complex, binary_data: BinaryLens_Data, tolerance: float = 1e-10) -> bool:
    """
    Verifica si una solución satisface la ecuación de lentes binarias para una sola fuente.
    
    Args:
        z (complex): Posición de la imagen a verificar
        zeta (complex): Posición de la fuente
        binary_data (BinaryLens_Data): Datos del sistema de lentes
        tolerance (float): Tolerancia máxima permitida
    
    Returns:
        bool: True si la ecuación se satisface dentro de la tolerancia
    """
    z_conj = np.conjugate(z)
    z1 = binary_data.z1
    z2 = binary_data.z2
    m1 = binary_data.m1
    m2 = binary_data.m2
    
    # Lado derecho de la ecuación de lentes
    right_side = z - m1/(z_conj - z1) - m2/(z_conj - z2)
    
    # Verificar si la ecuación se satisface dentro de la tolerancia
    return abs(zeta - right_side) < tolerance

def lens_equation_binary_point(zeta: complex, binary_data: BinaryLens_Data) -> np.ndarray:
    """
    Resuelve la ecuación de lentes binarias para una sola posición de fuente.
    
    Args:
        zeta (complex): Posición de la fuente
        binary_data (BinaryLens_Data): Datos del sistema de lentes binarias
    
    Returns:
        np.ndarray: Array con las posiciones de imágenes válidas para esta fuente
    """
    # Calcular coeficientes del polinomio
    coefficients = coefficients_lens_equation_binary_point(zeta, binary_data)

    # Obtener todas las soluciones potenciales
    all_solutions = solve_lens_equation_binary_point(coefficients)

    # Filtrar soluciones válidas
    valid_solutions = []
    for z in all_solutions:
        if verify_lens_equation_binary_point(z, zeta, binary_data):
            valid_solutions.append(z)
    
    return np.array(valid_solutions)

def magnification_binary_point(z: complex, binary_data: BinaryLens_Data) -> float:
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

def lens_equation_binary_lense(binary_data: BinaryLens_Data) -> BinaryLens_Data:
    """
    Resuelve la ecuación de lentes binarias para todas las posiciones de fuente.
    
    Args:
        binary_data (BinaryLens_Data): Datos del sistema de lentes binarias
    
    Returns:
        BinaryLens_Data: Datos actualizados con las posiciones de imágenes encontradas
    """
       
    all_solutions = []
    for zeta in binary_data.zeta:
        valid_solutions = lens_equation_binary_point(zeta, binary_data)
        all_solutions.append(valid_solutions)
        
    binary_data.image_positions = np.array(all_solutions, dtype=object)
    
    return binary_data

def magnification_binary_lense(binary_data: BinaryLens_Data) -> BinaryLens_Data:
    """
    Calcula las magnificaciones totales para el sistema de lentes binarias.
    
    Args:
        binary_data (BinaryLens_Data): Datos del sistema con posiciones de imágenes
    
    Returns:
        BinaryLens_Data: Datos actualizados con magnificaciones calculadas
    """

    total_mags = []
        
    for i, image_positions in enumerate(binary_data.image_positions):
        total_mag = 0
        for z in image_positions:
            mag = magnification_binary_point(z, binary_data)
            total_mag += mag
            
        total_mags.append(total_mag)
        
    binary_data.magnification = np.array(total_mags)
    
    return binary_data

def solve_critical_curve_binary_lense(phi: float, binary_data: BinaryLens_Data, tolerance: float = 1e-10) -> np.ndarray:
    """
    Resuelve la ecuación polinómica de la curva crítica para un ángulo phi dado.
    
    Args:
        phi (float): Ángulo en radianes
        binary_data (BinaryLens_Data): Datos del sistema de lentes binarias
        tolerance (float): Tolerancia para filtrar soluciones
    
    Returns:
        np.ndarray: Array de soluciones complejas válidas
    """
    z1 = binary_data.z1
    m1 = binary_data.m1
    m2 = binary_data.m2

    # Ensure z1, m1 and m2 are not None
    if z1 is None or m1 is None or m2 is None:
        raise ValueError("z1, m1, and m2 must be set (not None) in solve_critical_curve_binary_lense")

    # Calcular coeficientes
    exp_iphi = np.exp(-1j * phi)
    
    c4 = exp_iphi
    c3 = 0
    c2 = -2 * z1**2 * exp_iphi - (m1 + m2)
    c1 = 2 * z1 * (m2 - m1)
    c0 = -z1**2 * (m1 + m2) + z1**4 * exp_iphi
    
    coeffs = [c4, c3, c2, c1, c0]
    
    # Resolver polinomio
    solutions = np.roots(coeffs)
    
    # Filtrar soluciones válidas (remover artefactos numéricos)
    valid_sols = solutions[np.abs(np.polyval(coeffs, solutions)) < tolerance]
    
    return valid_sols

def map_to_source_plane_binary_lense(z: complex, binary_data: BinaryLens_Data) -> complex:
    """
    Mapea puntos del plano de la lente al plano de la fuente usando la ecuación de lentes.
    
    Args:
        z (complex): Punto en el plano de la lente
        binary_data (BinaryLens_Data): Datos del sistema de lentes binarias
    
    Returns:
        complex: Punto correspondiente en el plano de la fuente
    """
    z1 = binary_data.z1
    z2 = binary_data.z2
    m1 = binary_data.m1
    m2 = binary_data.m2
    
    z_conj = np.conjugate(z)
    return z - m1/(z_conj - z1) - m2/(z_conj - z2)

def calculate_caustics_and_critical_curves_binary_lense(binary_data: BinaryLens_Data, num_points: int = 1000) -> BinaryLens_Data:
    """
    Calcula las curvas críticas y caústicas para un sistema de lentes binarias y las añade al dataclass.
    
    Args:
        binary_data (BinaryLens_Data): Datos del sistema de lentes binarias
        num_points (int): Número de puntos para calcular las curvas
    
    Returns:
        BinaryLens_Data: Datos actualizados con curvas críticas y caústicas
    """
    phi_values = np.linspace(0, 2*np.pi, num_points)
    critical_points = []
    caustic_points = []
    
    for phi in phi_values:
        # Obtener puntos de la curva crítica
        crit_sols = solve_critical_curve_binary_lense(phi, binary_data)
        critical_points.extend(crit_sols)
        
        # Mapear al plano de la fuente para obtener caústicas
        caustic_sols = [map_to_source_plane_binary_lense(z, binary_data) for z in crit_sols]
        caustic_points.extend(caustic_sols)
    
    # Añadir las curvas al dataclass
    binary_data.critical_points = np.array(critical_points)
    binary_data.caustic_points = np.array(caustic_points)
    
    return binary_data

def trajectory_binary_lens(binary_data: BinaryLens_Data, start_point: tuple, end_point: tuple, num_points: int = 1000) -> BinaryLens_Data:
    """
    Calcula la trayectoria de la fuente y las posiciones de imágenes para un sistema de lentes binarias.
    
    Args:
        binary_data (BinaryLens_Data): Datos del sistema de lentes binarias
        start_point (tuple): Punto inicial (x, y) de la trayectoria
        end_point (tuple): Punto final (x, y) de la trayectoria
        num_points (int): Número de puntos en la trayectoria
    
    Returns:
        BinaryLens_Data: Datos actualizados con trayectoria y posiciones de imágenes
    """
    # Crear array de posiciones de fuente a lo largo de la línea
    x = np.linspace(start_point[0], end_point[0], num_points)
    y = np.linspace(start_point[1], end_point[1], num_points)
    
    # Convertir a números complejos
    zeta_array = np.array([complex(x[i], y[i]) for i in range(num_points)])
    binary_data.zeta = zeta_array
    
    # Resolver la ecuación de lentes para cada posición
    binary_data = lens_equation_binary_lense(binary_data)
    
    # Calcular magnificaciones
    binary_data = magnification_binary_lense(binary_data)
    
    return binary_data

