import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys 
import json
import warnings
from typing import Tuple, Union, List, Dict, Any, Optional
from pathlib import Path
from scipy.signal import find_peaks
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Functions.NNClass import PersonalizedSequential
from Functions.SimulationFunctions import (
    SingleLens_Data, 
    BinaryLens_Data, 
    trajectory_single_lens,
    trajectory_binary_lens
)

ROOT_DIR = Path().absolute() / "Redes"
print("Using ROOT_DIR:", ROOT_DIR)
# ------------------------------------------------------
# Functions for parsing the Real Curve Data
# ------------------------------------------------------

def parse_params(path):
    """
    Return a dict of key: value pairs from params.dat.
    For entries with two values, saves second as err_{key}.
    Example:
        "Tmax 2455644.193 0.008" becomes:
        {'Tmax': 2455644.193, 'err_Tmax': 0.008}
    """
    out = {}
    with open(path, "r", encoding="ascii", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            # Split all parts by whitespace
            parts = line.split()
            
            # Handle key-value pairs with different formats
            if len(parts) >= 2:
                key = parts[0]
                try:
                    val = float(parts[1])
                    out[key] = val
                    
                    # If there's a third value, store it as error
                    if len(parts) >= 3:
                        try:
                            err = float(parts[2])
                            out[f'err_{key}'] = err
                        except ValueError:
                            pass
                except ValueError:
                    out[key] = parts[1]  # keep as string if not numeric
    return out

def parse_phot(path):
    """
    Return dict of arrays for the five phot.dat columns,
    coercing any non-numeric tokens to NaN and padding
    short rows with NaN.
    """
    cols = [[] for _ in range(5)]
    with open(path, "r", encoding="ascii", errors="ignore") as fh:
        for line in fh:
            tok = line.strip().split()
            if not tok:
                continue

            cleaned = []
            for t in tok[:5]:                 # keep only first 5 tokens
                try:
                    cleaned.append(float(t))
                except ValueError:
                    cleaned.append(np.nan)    # letter or flag → NaN

            # pad if fewer than 5 numbers present
            cleaned += [np.nan] * (5 - len(cleaned))

            for i, val in enumerate(cleaned):
                cols[i].append(val)

    names = ["HJD", "I_mag", "I_err", "seeing", "sky"]
    return {n: np.array(c, dtype=float) for n, c in zip(names, cols)}

# -----------------------------------------------------------------------------------
# Functions for loading and saving files: .pkl, .json, .txt, .csv
# -----------------------------------------------------------------------------------

def load_data(filename: str, DIR: Path = ROOT_DIR) -> pd.DataFrame:
    """Load the raw data from pickle file."""    
    DIR.mkdir(parents=True, exist_ok=True) # make sure the folder exists
    path = DIR / filename
    df = pd.read_pickle(path)
    return df

def save_data(df: pd.DataFrame, filename: str, DIR: Path = ROOT_DIR) -> None:
    """Save dataframe to pickle file.""" 
    DIR.mkdir(parents=True, exist_ok=True) 
    path = DIR / filename
    df.to_pickle(path)
    print(f"Saved DataFrame to {path}")

def save_to_json(dictionary: dict, filename: str, DIR: Path = ROOT_DIR):
    """
    Save a Python dictionary to a JSON file.
    """
    path = DIR / filename    
    with open(path, 'w') as f:
        json.dump(dictionary, f, indent=2)
    print(f"Saved dictionary to {path}")

def read_json(filename: str, DIR: Path) -> dict:
    """
    Read a JSON file to a dictionary
    """
    path = DIR / filename
    
    with open(path, 'r') as f:
        stats = json.load(f)
    
    return stats

def load_events_from_txt(filename: str, DIR: Path = ROOT_DIR) -> list:
    """Load list of confirmed OGLE events and format them to match DataFrame index."""
    path = DIR / filename
    with open(path, 'r') as f:
        events = [line.strip().replace('-', '_', 1).lower() for line in f if line.strip()]

    print(f"\nFound {len(events)} events")
    return events

def save_to_csv(df: pd.DataFrame, filename: str, DIR: Path = ROOT_DIR):
    save_path = DIR / filename
    df.to_csv(save_path)
    print(f"\nSaved to: {save_path}")

def load_architectures_from_csv(filename: str, DIR: Path = ROOT_DIR) -> List[Dict[str, Any]]:
    """
    Load model architectures from a CSV file of previous results.
    
    Args:
        csv_path: Path to the CSV file
        n_models: Number of top models to extract (None for all)
        sort_by: Metric to sort models by ('test_accuracy', 'test_loss', etc.)
    
    Returns:
        List of architecture configurations
    """
    # Read and sort results
    csv_path = DIR / filename
    results_df = pd.read_csv(csv_path)
    
    architectures = []
    for _, row in results_df.iterrows():
        arch = {
            'n_layers': int(row['n_layers']),
            'kernel_sizes': eval(row['kernel_sizes']),
            'filters': eval(row['filters']),
            'pool_sizes': eval(row['pool_sizes']),
            'dense_units': int(row['dense_units']),
            'batch_size': int(row['batch_size']),
            'epochs': int(row['epochs'])
        }
        architectures.append(arch)
    
    return architectures

# -------------------------------------------------------------------------------------
# Basic Functions
# -------------------------------------------------------------------------------------

def normalize_array(x: np.ndarray, range_type: str = 'zero_one') -> np.ndarray:
    """
    Normalize array values to specified range.
    
    Args:
        x: Array of values to normalize
        range_type: String indicating normalization range ('zero_one' or 'minus_plus_one')
    
    Returns:
        Normalized array in specified range
    """

    denom = np.max(x) - np.min(x)
    if denom == 0:
        warnings.warn("Zero range detected in values. Returning zeros.", RuntimeWarning)
        return np.zeros_like(x)
    
    x_norm = (x - np.min(x)) / denom
    
    if range_type == 'minus_plus_one':
        x_norm = 2 * x_norm - 1
    elif range_type != 'zero_one':
        warnings.warn(f"Unknown range_type '{range_type}'. Using [0,1] range.", RuntimeWarning)
    
    return x_norm

# -------------------------------------------------------------------------------------
# Functions for preparing and processing the lightcurves
# -------------------------------------------------------------------------------------

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare and clean the dataframe with initial transformations.
    Change column names and center HJD"""
    selected_columns = ['Tmax', 'tau', "umin", 'I_mag', 'HJD']

    filtered_df = df[selected_columns].rename(columns={
        'Tmax': 't0',
        'tau': 'tE',
        'umin': "u0",
        'HJD': 'HJD',
        'I_mag': 'I'    
    })
    
    filtered_df = filtered_df[pd.to_numeric(filtered_df['tE'], errors='coerce').notna()] # The values of tE empty are omitted
    filtered_df['tE'] = filtered_df['tE'].astype(float)
    
    filtered_df["HJD"] = filtered_df["HJD"] - filtered_df["t0"]
    
    return filtered_df

def lightcurve_normalizer(row) -> Optional[pd.Series]:
    """
    Process a single lightcurve row.
    
    Args:
        row: DataFrame row containing HJD, I, u0, and tE
    
    Returns:
        Series with normalized HJD and I values, or None if processing fails
    """
    try:
       
        HJD = np.array(row['HJD'])
        I_mag = np.array(row['I'])
        te = row['tE']
        
        outer_mask = abs(HJD) > 2*te
        inner_mask = ~outer_mask
        
        
        if not np.any(inner_mask):
            print(f"Empty inner mask for event {row.name}, skipping...")
            return None
        
        if not np.any(outer_mask):
            print(f"Empty outer mask for event {row.name}, skipping...")
            return None
        
        # Filter events with insufficient data points (at least 3 points on each side of peak)
        positive_time_count = (HJD > 0).sum()
        negative_time_count = (HJD < 0).sum()
        
        if positive_time_count < 3 or negative_time_count < 3:
            print(f"Insufficient data points for event {row.name}: {negative_time_count} before peak, {positive_time_count} after peak. Skipping...")
            return None
        
        baseline = np.mean(I_mag[outer_mask]) 
        
        I_Inner = I_mag[inner_mask]
        HJD_Inner = HJD[inner_mask]
        Magnification = 10**(-0.4*(I_Inner-baseline))

        if len(HJD_Inner) < 20:
            print(f"Insufficient data points for event {row.name}, skipping...")
            return None
        
        Magnification_normalized = normalize_array(Magnification, range_type="zero_one")

        series = pd.Series({
            'time': HJD_Inner,
            "mu": Magnification_normalized,
            'tE': te,
            "u0": row["u0"]            
        })
        return series
    
    except Exception as e:
        print(f"Skipping event {row.name} due to processing error: {e}")
        return None
    
def is_binary_lens(magnification: np.ndarray, prominence_mag: float = 0.2, distance_ratio: int = 100, 
                   prominence_deriv: float = 0.2, distance_ratio_deriv: int = 100, 
                   tolerance: float = 0.2, L1_threshold: float = 30, Linf_threshold: float = 0.1, 
                   verbose: bool = False) -> bool:
    """
    Determine if a magnification curve represents clearly a binary lens system. 
    
    Args:
        magnification: Array of magnification values
        prominence_mag: Prominence threshold for magnification peaks (default: 0.1)
        distance_ratio: Distance ratio for magnification peaks (default: 20, meaning min_distance = len/20)
        prominence_deriv: Prominence threshold for derivative peaks (default: 0.2)
        distance_ratio_deriv: Distance ratio for derivative peaks (default: 50, meaning min_distance = len/50)
        tolerance: Tolerance for symmetry check (default: 0.2)
        L1_threshold: L1 norm threshold (default: 30)
        Linf_threshold: Linf norm threshold (default: 0.1)
        verbose: If True, prints detailed analysis (default: False)
    
    Returns:
        bool: True if binary lens system detected, False if single lens
        
    Binary lens criteria (in order of priority):
    1. If magnification peaks > 1 → Binary
    2. If derivative peaks > 2 → Binary
    3. If any peak has L1 norm > L1_threshold AND Linf norm > Linf_threshold → Binary
    4. If any peak has symmetry error > tolerance → Binary
    """
    magnification = np.array(magnification)
    times = np.linspace(0, 1, len(magnification))
    derivative = np.gradient(magnification, times)
    
    # Find peaks in magnification
    normalized_mag = (magnification - np.min(magnification)) / (np.max(magnification) - np.min(magnification))
    min_distance_mag = max(1, len(magnification) // distance_ratio)
    mag_peaks, _ = find_peaks(normalized_mag, prominence=prominence_mag, distance=min_distance_mag)
    num_mag_peaks = len(mag_peaks)
    
    # Find peaks in derivative
    min_distance_deriv = max(1, len(magnification) // distance_ratio_deriv)
    deriv_peaks, _ = find_peaks(np.abs(derivative), prominence=prominence_deriv, distance=min_distance_deriv)
    num_deriv_peaks = len(deriv_peaks)
    
    if verbose:
        print(f"Magnification peaks detected: {num_mag_peaks}")
        print(f"Derivative peaks detected: {num_deriv_peaks}")
    
    # Criterion 1: More than 1 magnification peak
    if num_mag_peaks > 1:
        if verbose:
            print(f"BINARY: More than 1 magnification peak ({num_mag_peaks})")
        return True
    
    # Criterion 2: More than 2 derivative peaks
    if num_deriv_peaks > 2:
        if verbose:
            print(f"BINARY: More than 2 derivative peaks ({num_deriv_peaks})")
        return True
    
    # If no magnification peaks found, classify as single
    if num_mag_peaks == 0:
        if verbose:
            print("SINGLE: No magnification peaks detected")
        return False
    
    # For single peak, perform symmetry analysis
    peak_idx = mag_peaks[0]  # Only one peak to analyze
    
    if verbose:
        print(f"Analyzing single peak at index {peak_idx}")
    
    # Calculate analysis boundaries
    left_boundary = 0
    right_boundary = len(times) - 1
    left_distance = peak_idx - left_boundary
    right_distance = right_boundary - peak_idx
    analysis_distance = min(left_distance, right_distance)
    
    if analysis_distance < 5:
        if verbose:
            print("SINGLE: Insufficient points for symmetry analysis")
        return False
    
    # Extract derivative segments around the peak
    left_start = peak_idx - analysis_distance
    right_end = peak_idx + analysis_distance
    
    left_derivative = derivative[left_start:peak_idx]
    right_derivative = derivative[peak_idx+1:right_end+1]
    
    # Normalize derivatives
    left_normalized = left_derivative / np.max(np.abs(left_derivative)) if np.max(np.abs(left_derivative)) > 0 else left_derivative
    right_normalized = right_derivative / np.max(np.abs(right_derivative)) if np.max(np.abs(right_derivative)) > 0 else right_derivative
    right_inverted_reversed = -right_normalized[::-1]
    
    # Calculate symmetry metrics
    if len(left_normalized) == len(right_inverted_reversed):
        diff = left_normalized - right_inverted_reversed
    else:
        min_length = min(len(left_normalized), len(right_inverted_reversed))
        diff = left_normalized[-min_length:] - right_inverted_reversed[:min_length]
    
    L1_norm = np.linalg.norm(diff, ord=1)
    L2_norm = np.linalg.norm(diff, ord=2)
    Linf_norm = np.linalg.norm(diff, ord=np.inf)
    symmetry_error = np.mean(np.abs(diff))
    
    if verbose:
        print(f"L1 norm: {L1_norm:.4f}")
        print(f"L2 norm: {L2_norm:.4f}")
        print(f"Linf norm: {Linf_norm:.4f}")
        print(f"Symmetry error: {symmetry_error:.4f}")
    
    # Criterion 3: L1 norm > threshold AND Linf norm > threshold
    if L1_norm > L1_threshold and Linf_norm > Linf_threshold:
        if verbose:
            print(f"BINARY: L1 norm > {L1_threshold} ({L1_norm:.4f}) AND Linf norm > {Linf_threshold} ({Linf_norm:.4f})")
        return True
    
    # Criterion 4: Symmetry error > tolerance
    if symmetry_error > tolerance:
        if verbose:
            print(f"BINARY: Symmetry error > {tolerance} ({symmetry_error:.4f})")
        return True
    
    # All criteria for single lens met
    if verbose:
        print("SINGLE: All criteria for single lens met")
    return False

def create_lightcurves_df(params: dict, binary: bool = False, filename: str = "", DIR: Path = ROOT_DIR) -> pd.DataFrame:
    """Create DataFrame with light curve parameters and magnifications.
    
    Args:
        params: Dictionary containing arrays of parameters
               Single lens: {'tE', 'u0'}
               Binary lens: {'tE', 'u0', 'd', 'q', 'start', 'end'} 
                          (note: 'd' is separation distance instead of 'z1')
        binary: Boolean indicating if binary lens calculation should be used
        filename: JSON file with parameter statistics for regenerating parameters during retries
    
    Returns:
        DataFrame with columns ['tE', 'u0', 'time', 'mu'] and proper indexing
    """
    print(f"Creating light curves DataFrame for {'binary' if binary else 'single'} lens...")
    n_curves = len(params['tE'])
    
    label = "binarylens" if binary else "singlelens"
    index = [f'{label}_{i:04d}' for i in range(n_curves)]
    
    df = pd.DataFrame(index=index, columns=['tE', 'u0', 'time', 'mu'])
    
    successful_curves = 0
    
    for i, idx in enumerate(index):
        print(f"Processing curve {i+1}/{n_curves}\n", end="\r")
        
        if binary:
            max_attempts = 1  
            valid_curve = False
            for attempt in range(max_attempts):
                # Use original parameters for first attempt, then regenerate
                if attempt == 0:
                    current_q = params['q'][i]
                    current_d = params['d'][i]
                    current_start = params['start'][i]
                    current_end = params['end'][i]
                else:
                    # Regenerate parameters using get_random_params for retry attempts
                    if filename:
                        # Use get_random_params to generate new parameters statistically
                        new_params = get_random_params(n_samples=1, binary=True, filename=filename, DIR=DIR)
                        current_q = new_params['q'][0]
                        current_d = new_params['d'][0]
                        current_start = new_params['start'][0]
                        current_end = new_params['end'][0]
                    else:
                        # Fallback to manual random generation if no filename provided
                        current_q = 10**np.random.normal(-3.2, 1)
                        current_d = np.random.uniform(0.1, 2.0)
                        
                        # Generate new trajectory points
                        alpha = np.random.uniform(0, 2*np.pi)
                        u0 = params['u0'][i]  # Keep original u0
                        start_points, end_points = get_trajectory_extreme_points(
                            np.array([u0]), 
                            np.array([alpha]), 
                            np.array([current_d])
                        )
                        current_start = start_points[0]
                        current_end = end_points[0]
                
                # Create BinaryLens_Data object with current parameters
                binary_data = BinaryLens_Data(
                    m_t=1.0,  # Total mass fixed at 1.0
                    q=current_q,  # Mass ratio
                    d=current_d,  # Separation distance 
                    start_point=current_start,
                    end_point=current_end,
                    num_points=1000
                )
                
                # Calculate trajectory and magnification
                binary_data.images_magnification_calculation()
                
                # Extract times and magnifications
                times = np.linspace(0, 1, len(binary_data.magnification))
                magnifications = binary_data.magnification
                
                # Check if binary lens (for attempts 0-3), or accept on last attempt
                if attempt < max_attempts - 1:  # Attempts 0-18 (check binary)
                    if is_binary_lens(magnifications, verbose=False):
                        valid_curve = True
                        break
                    else:
                        print("Not valid binary lens, trying again...")
                        continue
                else:  # Last attempt (attempt 19)
                    if is_binary_lens(magnifications, verbose=False):
                        # Even on last attempt, we got a good binary lens!
                        valid_curve = True
                    else:
                        # Last attempt and still not binary - warn but keep the curve
                        print(f"\nWarning: Could not generate valid binary lens curve for {idx} after {max_attempts} attempts")
                        valid_curve = True
                    break
            
            if not valid_curve:
                # This should never happen now, but just in case
                print(f"Unexpected error for index {idx}. Skipping...")
                continue
        else:
            # Single lens - no filtering needed
            # Create SingleLens_Data object
            single_data = SingleLens_Data(
                t_E=params['tE'][i],
                u_0=params['u0'][i],
                num_points=1000
            )
            
            # Calculate trajectory and magnification using the new method
            single_data = trajectory_single_lens(single_data)
            
            # Extract times and magnifications
            times = single_data.t
            magnifications = single_data.magnification
        
        # Add noise to the magnification curve
        magnifications = add_noise_to_curve(magnifications)
        magnifications = normalize_array(magnifications, "zero_one")
        
        # Store the curve
        df.loc[idx] = {
            'tE': params['tE'][i],
            'u0': params['u0'][i],
            'time': times,
            'mu': magnifications
        }
        
        successful_curves += 1
    
    print(f"\nFinished processing {successful_curves} curves")
    
    return df

# ------------------------------------------------------------------------------
# Functions about the statistical distributions
# -----------------------------------------------------------------------------

def get_filtered_distribution(values: np.ndarray, quantile_step: int = 5) -> dict:
    """
    Get filtered parameter distribution using fine-grained quantiles.
    
    Args:
        values: Array of parameter values
        quantile_step: Step in between measured quantiles
    
    Returns:
        dict: Filtered statistics and values for the parameter
    """
    # Calculate quantiles every 5%
    quantiles = np.arange(0, 100, quantile_step)  # 0, 5, 10, ..., 95
    param_quantiles = np.percentile(values, quantiles)
    
    # Filter values (keep everything below 95th percentile)
    filtered_values = values[values <= param_quantiles[-1]]
    
    # Calculate statistics for filtered data
    filtered_stats = {
        'mean': float(np.mean(filtered_values)),
        'std': float(np.std(filtered_values)),
        'median': float(np.median(filtered_values)),
        'quantiles': {str(int(q)): float(v) for q, v in zip(quantiles, param_quantiles)}
    }
    
    return filtered_stats

def generate_random_from_quantiles(distribution_stats: dict, n_samples: int) -> np.ndarray:
    """
    Generate random values following a quantile distribution
    
    Args:
        distribution_stats: Dictionary containing "quantiles" for a distribution
        n_samples: Number of random values to generate
    
    Returns:
        random_values: Array of random values following the parameter's distribution
    """
    quantiles, values = np.array([(float(q)/100, float(v)) for q, v in distribution_stats['quantiles'].items()]).T
    
    random_probs = np.random.uniform(0, 0.95, n_samples)    

    random_values = np.interp(random_probs, quantiles, values)
    
    return random_values

def get_random_params(n_samples: int, binary: bool = False, filename: str = "", DIR: Path = ROOT_DIR) -> dict:
    """
    Generate random parameters for microlensing simulations.
    
    Args:
        n_samples: Number of samples to generate
        binary: Whether to include binary lens parameters
    
    Returns:
        dict: Dictionary with random parameters
               Single lens: {'tE', 'u0'}
               Binary lens: {'tE', 'u0', 'd', 'q', 'start', 'end'}
    """
    if filename == "":
        raise ValueError("Filename must be provided for parameter statistics")
    
    stats = read_json(filename, DIR)
    
    # Generate base random parameters from stats
    random_params = {}
    for param_name, param_stats in stats.items():
        random_params[param_name] = generate_random_from_quantiles(param_stats, n_samples)
    
    if binary:
        # Add binary lens specific parameters
        random_params.update({
            'alpha': np.random.uniform(0, 2*np.pi, n_samples),
            'd': np.random.uniform(0.1, 2.0, n_samples),  
            'q': 10**np.random.normal(-3.2, 1, n_samples)
        })
        
        start_points, end_points = get_trajectory_extreme_points(
            random_params['u0'], 
            random_params['alpha'], 
            random_params['d']
        )
        random_params.update({
            'start': start_points,
            'end': end_points
        })
    
    return random_params    

# ---------------------------------------------------------------------------------------
# Funtions for binary curves calculation
# ---------------------------------------------------------------------------------------
# Note: These functions are kept for compatibility but are superseded by 
# the new dataclass-based approach in SimulationFunctions.py

def get_trajectory_extreme_points(u0_values: np.ndarray, alpha_values: np.ndarray, d_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate start and end points for multiple trajectories using separation distance.
    
    Args:
        u0_values: Array of impact parameters
        alpha_values: Array of angles in radians
        d_values: Array of separation distances between lenses
    
    Returns:
        tuple: (start_points, end_points) as arrays of tuples (x, y)
    """
    # Convert separation distance to z1 position (z1 = d/2)
    z1_values = d_values / 2
    
    # Calculate line parameters (ax + by + c = 0)
    a = np.sin(alpha_values)
    b = -np.cos(alpha_values)
    c = u0_values * np.sqrt(a**2 + b**2)  # Ensures distance to (z1,0) is u0
    
    # Points at t = -2 and t = 2
    x_start = -2*np.cos(alpha_values)
    y_start = -2*np.sin(alpha_values)
    x_end = 2*np.cos(alpha_values)
    y_end = 2*np.sin(alpha_values)
    
    # Shift lines to maintain u0 distance from m1
    denominator = a**2 + b**2
    x_shift = -(a*z1_values)/denominator*a + c/denominator*a
    y_shift = -(a*z1_values)/denominator*b + c/denominator*b
    
    # Apply shifts
    x_start += x_shift
    y_start += y_shift
    x_end += x_shift
    y_end += y_shift
    
    # Convert to arrays of tuples (x, y) for the new dataclass approach
    start_points = [(x_start[i], y_start[i]) for i in range(len(x_start))]
    end_points = [(x_end[i], y_end[i]) for i in range(len(x_end))]
    
    return np.array(start_points), np.array(end_points)

# ----------------------------------------------------------------------------------------
# Functions for resampling
# ----------------------------------------------------------------------------------------

def resample_curve(times: list, magnifications: list, sequence_length: int, interpolation_method: str='linear', return_times: bool=True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Resampling function supporting multiple interpolation methods. All methods use time normalized to [-1, 1].
    
    Args:
        times: Array of time values
        magnifications: Array of magnification values
        sequence_length: Length of output sequence
        interpolation_method: Interpolation method ('linear', 'spline', 'savgol', 'bin', 'gp')
        return_times: Whether to return time array (default=True)
    
    Returns:
        If return_times is True:
            Tuple[np.ndarray, np.ndarray]: (new_times, resampled_magnification)
        If return_times is False:
            np.ndarray: resampled_magnification only
    """
    try:
        # Convert to numpy arrays and ensure they're 1D
        times = np.asarray(times, dtype=float).flatten()
        magnifications = np.asarray(magnifications, dtype=float).flatten()
        
        # Check for valid data
        if len(times) == 0 or len(magnifications) == 0:
            raise ValueError("Empty time or magnification arrays")
        
        if len(times) != len(magnifications):
            raise ValueError(f"Time and magnification arrays have different lengths: {len(times)} vs {len(magnifications)}")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(times)) or np.any(np.isnan(magnifications)):
            raise ValueError("NaN values found in data")
        
        if np.any(np.isinf(times)) or np.any(np.isinf(magnifications)):
            raise ValueError("Infinite values found in data")
        
        # Normalize times
        normed_times = normalize_array(times, "minus_plus_one")
        new_times = np.linspace(-1, 1, sequence_length)
        
        if interpolation_method == 'linear':
            interpolated_points = np.interp(new_times, normed_times, magnifications)
            return (new_times, interpolated_points) if return_times else interpolated_points
        
        elif interpolation_method == 'spline': # Cubic Splines
            from scipy.interpolate import UnivariateSpline        
            
            noise_level = np.std(magnifications) * 0.1  # Adaptive smoothing factor based on data noise
            spline = UnivariateSpline(normed_times, magnifications, k=3, s=noise_level)
            interpolated_points = spline(new_times)
            return (new_times, interpolated_points) if return_times else interpolated_points
        
        elif interpolation_method == 'savgol': # Savitzky-Golay filtering
            from scipy.signal import savgol_filter
            
            window = min(11, len(times) - (len(times) % 2) - 1)  # Ensure odd window
            poly = min(3, window - 1)  # Ensure valid polynomial order
            smoothed = savgol_filter(magnifications, window, poly)
            interpolated_points = np.interp(new_times, normed_times, smoothed)
            return (new_times, interpolated_points) if return_times else interpolated_points
        
        elif interpolation_method == 'bin': # Binned averaging
            bins = np.linspace(-1, 1, sequence_length + 1)
            digitized = np.digitize(normed_times, bins)
            binned = np.array([magnifications[digitized == i].mean() 
                              if len(magnifications[digitized == i]) > 0 
                              else np.nan 
                              for i in range(1, len(bins))])        
            mask = np.isnan(binned) # Fill NaN values through interpolation
            binned[mask] = np.interp(np.flatnonzero(mask), 
                                    np.flatnonzero(~mask), 
                                    binned[~mask])        
            return (new_times, binned) if return_times else binned
        
        elif interpolation_method == 'gp':
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
            
            X = normed_times.reshape(-1, 1)
            y = magnifications.reshape(-1, 1)
                
            # Adaptive kernel parameters based on data characteristics
            data_range = np.max(normed_times) - np.min(normed_times)  # Should be 2 for normalized [-1,1]
            n_points = len(normed_times)
            avg_spacing = data_range / n_points if n_points > 1 else 0.1
            
            # Set more reasonable length scale bounds
            length_scale_init = max(avg_spacing * 10, 0.1)  # Start with larger length scale
            length_scale_bounds = (1e-3, 10.0)  # Much wider bounds
            
            k1 = ConstantKernel(1.0, (1e-3, 1e3))
            k2 = RBF(length_scale=length_scale_init, length_scale_bounds=length_scale_bounds)
            k3 = WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-6, 1.0))
            kernel = k1 * k2 + k3

            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=15,  # More restarts for better optimization
                normalize_y=True,
                alpha=1e-6,
                random_state=42
            )
            
            try:
                with warnings.catch_warnings():
                    from sklearn.exceptions import ConvergenceWarning
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    gp.fit(X, y)
                
                X_new = new_times.reshape(-1, 1)
                y_pred, y_std = gp.predict(X_new, return_std=True)
                
                # Check prediction quality
                pred_variance = np.var(y_pred)
                data_variance = np.var(magnifications)
                
                if pred_variance < data_variance * 0.05:  # Too flat prediction
                    print("GP produced overly smooth prediction, falling back to linear")
                    return resample_curve(times, magnifications, sequence_length, 'linear', return_times)
                
                return (new_times, y_pred.flatten()) if return_times else y_pred.flatten()
                
            except Exception as e:
                print(f"GP interpolation failed ({str(e)}), falling back to linear")
                return resample_curve(times, magnifications, sequence_length, 'linear', return_times)  
        else:
            raise ValueError(f"Unknown method: {interpolation_method}. Use 'linear', 'spline', 'savgol', 'bin', or 'gp'")
            
    except Exception as e:
        print(f"Error in resample_curve: {e}")
        print(f"Times type: {type(times)}, shape: {getattr(times, 'shape', 'no shape')}")
        print(f"Magnifications type: {type(magnifications)}, shape: {getattr(magnifications, 'shape', 'no shape')}")
        if hasattr(times, '__len__') and len(times) > 0:
            print(f"Times sample: {times[:5]}")
        if hasattr(magnifications, '__len__') and len(magnifications) > 0:
            print(f"Magnifications sample: {magnifications[:5]}")
        raise

def apply_resampling(df: pd.DataFrame, sequence_length: int, interpolation_method: str) -> np.ndarray:
    print(f"Resampling curves with {interpolation_method} method")    
    X = np.stack(df.apply(
        lambda row: resample_curve(row["time"], row["mu"], sequence_length, interpolation_method, return_times=False),
        axis=1
    ).to_numpy())
    X = X[..., np.newaxis] # Add channel dimension for Conv1D: (N, L, 1)
    return X

# --------------------------------------------------------------------------------------------------------------
# Noise Functions
# --------------------------------------------------------------------------------------------------------------

def add_noise_to_curve(magnification, signal_to_noise=50) -> np.ndarray:
    """Add Gaussian noise to magnification curve."""
    
    signal_amplitude = np.max(magnification) - np.min(magnification)
    noise_level = signal_amplitude / signal_to_noise
    
    noise = np.random.normal(0, noise_level, size=len(magnification))
    noisy_magnification = magnification + noise
    
    noisy_magnification = np.maximum(noisy_magnification, 0)
    
    return noisy_magnification

def make_noise_curve(n_points: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic noise curve.
    
    Creates a noise curve with irregular sampling cadence, combining white noise, 
    red (correlated) noise, and outliers to simulate realistic observational data.
    
    Args:
        n_points (int): Number of data points to generate
        seed (int, optional): Random seed for reproducibility. Defaults to None.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - t: Array of time points with irregular OGLE-like cadence
            - mu: Array of flux values with combined noise components
    
    Constants:
        RED_NOISE_SIG (float): Amplitude of red noise component (0.005)
        WHITE_NOISE_SIG_BASE (float): Base amplitude of white noise (0.008)
        OUTLIER_FRAC (float): Fraction of points to be outliers (0.02)
        
    Notes:
        - Time points follow irregular sampling with intervals 0.3-1.2 days
        - White noise simulates photometric measurement errors
        - Red noise simulates correlated systematic effects
        - Outliers are added to simulate bad measurements
        - All arrays are returned as float32 for memory efficiency
    """

    RED_NOISE_SIG = 0.005
    WHITE_NOISE_SIG_BASE = 0.008
    OUTLIER_FRAC = 0.02    
    phi = 0.8
    """Generate synthetic noise curve."""
    rng = np.random.default_rng(seed)
    
    # Generate irregular OGLE-like cadence
    dt = rng.uniform(0.3, 1.2, size=n_points)
    times = np.cumsum(dt)
    
    # Add white noise
    sigma_white_noise = (np.log(10)/2.5)*WHITE_NOISE_SIG_BASE #Convert photometric mag error to *fractional* flux error.
    white_noise = rng.normal(0, sigma_white_noise, size=n_points)
    
    # Add red (correlated) noise
    red_noise = np.empty(n_points)
    red_noise[0] = rng.normal(0, RED_NOISE_SIG)
    for i in range(1, n_points):
        red_noise[i] = phi * red_noise[i-1] + rng.normal(0, RED_NOISE_SIG*np.sqrt(1-phi**2))
    
    magnification = 1.0 + white_noise + red_noise
    
    # Add outliers
    k = rng.choice(n_points, size=int(OUTLIER_FRAC * n_points), replace=False)
    magnification[k] += rng.normal(0, 5 * sigma_white_noise, size=len(k))
    magnification = normalize_array(magnification, "zero_one")
    return times, magnification

# ----------------------------------------------------------------------------------------------------------
# Plotting Functions
# ----------------------------------------------------------------------------------------------------------

def plot_curve_with_resampling(times: np.ndarray, magnifications: np.ndarray, sequence_length: int=1000, interpolation_method: str='linear', title: str="Light Curve"):
    """
    Plot original data points and resampled curve.
    """
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    normed_times = normalize_array(times, "minus_plus_one")
    ax1.plot(times, magnifications, 'b.', alpha=0.6, markersize=2, label='Original')
    ax1.set_title('Original Data')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Magnification')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    times_resampled, magnifications_resampled = resample_curve(times, magnifications, sequence_length, interpolation_method)
    
    ax2.plot(times_resampled, magnifications_resampled, 'r-', label=f'{interpolation_method} resampled')
    ax2.plot(normed_times, magnifications, 'b.', alpha=0.3, markersize=2, label='Original')
    ax2.set_title(f'Resampled Data ({interpolation_method})')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Magnification')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.suptitle(title)
    plt.tight_layout()

    plt.show()

def plot_random_events_from_df(df: pd.DataFrame, n_events: int=5, sequence_length: int=1_000, interpolation_method: str="linear", title_prefix: str="") -> np.ndarray:
    """
    Plot n random events from a DataFrame containing microlensing curves. Returns the events plotted.
    """
    random_events = np.random.choice(df.index, size=min(n_events, len(df)), replace=False)
    
    for i, event_id in enumerate(random_events, 1):
        title = f"{title_prefix} Event {i}/{n_events}\nID: {event_id}"
        times = df.loc[event_id, 'time']
        magnifications = df.loc[event_id, 'mu']
        
        plot_curve_with_resampling(times, magnifications, sequence_length, interpolation_method, title=title)
    
    return random_events

def plot_training_history(history):

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_prediction_stats(df: pd.DataFrame):
    """Plot prediction statistics for event types."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot confidence histogram
    ax1.hist(df['Confidence'], bins=20, edgecolor='black')
    ax1.set_title('Prediction Confidence Distribution')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Number of Events')
    ax1.grid(True, alpha=0.3)
    
    # Plot event type distribution pie chart
    type_counts = df['Predicted_Type'].value_counts()
    ax2.pie(type_counts.values, 
            labels=type_counts.index, 
            autopct='%1.1f%%',
            colors=['lightcoral', 'lightgreen', 'lightblue'])
    ax2.set_title('Event Type Distribution')
    
    plt.tight_layout()
    plt.show()

def plot_model_analysis(df: pd.DataFrame, save_filename: str, DIR: Path = ROOT_DIR) -> None:
    """Create and save comparison plots for model results with architecture-specific colors."""
    save_path = DIR / "plots" / save_filename
    
    # Create architecture identifier for consistent coloring
    df['architecture_id'] = df.apply(lambda row: f"{row['n_layers']}C_{str(row['filters'])}", axis=1)
    unique_architectures = df['architecture_id'].unique()
    
    # Create color palette for architectures
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(unique_architectures)))
    color_map = dict(zip(unique_architectures, colors))
    
    # Create three plots in one row with space for legend on the right
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot for each interpolation method (using 'source' column)
    for interp_method in df['source'].unique():
        method_data = df[df['source'] == interp_method]
        
        for arch in unique_architectures:
            arch_data = method_data[method_data['architecture_id'] == arch]
            if len(arch_data) == 0:
                continue
                
            color = color_map[arch]
            
            # Linear: empty circles (edgecolor only)
            # Savgol: filled circles
            if interp_method.lower() == 'linear':
                marker_style = 'o'
                facecolor = 'none'
                edgecolor = color
                alpha = 0.8
            else:  # savgol
                marker_style = 'o'
                facecolor = color
                edgecolor = color
                alpha = 0.8
            
            # Plot 1: Accuracy vs Training Time
            ax1.scatter(arch_data['training_time'], arch_data['accuracy'], 
                       c=facecolor, edgecolors=edgecolor, s=100, 
                       marker=marker_style, alpha=alpha,
                       label=f'{arch}' if interp_method.lower() == 'linear' else '')
            
            # Plot 2: Accuracy vs Model Parameters
            ax2.scatter(arch_data['params'], arch_data['accuracy'], 
                       c=facecolor, edgecolors=edgecolor, s=100, 
                       marker=marker_style, alpha=alpha)
            
            # Plot 3: Accuracy vs Loss
            ax3.scatter(arch_data['loss'], arch_data['accuracy'], 
                       c=facecolor, edgecolors=edgecolor, s=100, 
                       marker=marker_style, alpha=alpha)
    
    # Configure Plot 1
    ax1.set_title('Accuracy vs Tiempo de Entrenamiento', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Tiempo de Entrenamiento (segundos)')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True, alpha=0.3)
    
    # Configure Plot 2
    ax2.set_title('Accuracy vs Parámetros del Modelo', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Número de Parámetros')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    # Configure Plot 3
    ax3.set_title('Accuracy vs Pérdida', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Pérdida')
    ax3.set_ylabel('Accuracy')
    ax3.grid(True, alpha=0.3)
    
    # Create combined legend on the right side of all plots
    # Architecture legend
    arch_handles = ax1.get_legend_handles_labels()[0]
    arch_labels = ax1.get_legend_handles_labels()[1]
    
    # Interpolation method legend
    from matplotlib.lines import Line2D
    interp_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', 
               markeredgecolor='black', markersize=10, label='Lineal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
               markeredgecolor='black', markersize=10, label='Savgol')
    ]
    
    # Combine all legend elements
    all_handles = arch_handles + interp_handles
    all_labels = arch_labels + ['Lineal (vacío)', 'Savgol (relleno)']
    
    # Place legend to the right of all plots
    fig.legend(all_handles, all_labels, bbox_to_anchor=(0.98, 0.5), 
               loc='center right', fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Make space for the legend
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Gráficos de análisis de modelos guardados en: {save_path}")
    plt.close()

def plot_combined_histories(histories: list, labels: list, times: list, saving: bool = True, 
                            save_path: Path = ROOT_DIR / "plots" / 'training_histories.pdf'):
    """Plot all training histories in one figure with training times."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    for i, (history, time) in enumerate(zip(histories, times)):
        # Plot accuracy
        ax1.plot(history.history['accuracy'], 
                label=f'Model {labels[i]} (train) - {time:.1f}s')
        ax1.plot(history.history['val_accuracy'], 
                label=f'Model {labels[i]} (val) - {time:.1f}s')
        
        # Plot loss
        ax2.plot(history.history['loss'], 
                label=f'Model {labels[i]} (train) - {time:.1f}s')
        ax2.plot(history.history['val_loss'], 
                label=f'Model {labels[i]} (val) - {time:.1f}s')
    
    ax1.set_title('Model Accuracy vs Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True)
    
    ax2.set_title('Model Loss vs Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True)
    
    plt.suptitle('Training Histories Comparison', y=1.05)
    plt.tight_layout()
    if saving: 
        # Ensure the directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()

# ---------------------------------------------------------------------------------------------
# Functions for the model
# ---------------------------------------------------------------------------------------------

def model_data_preparation(df: pd.DataFrame, sequence_length: int, interpolation_method: str, 
                           test_fraction: float, validation_fraction: float, 
                           random_seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray , np.ndarray]:
    """
    Prepare data for model training by resampling curves and splitting into train/val/test sets.
    
    This function handles the complete data preparation pipeline:
    1. Resamples all curves to the same length using specified interpolation
    2. Converts event types to one-hot encoded format
    3. Performs stratified train/test split
    4. Further splits training data into train/validation sets
    
    Args:
        df (pd.DataFrame): DataFrame containing light curves with columns:
            - 'time': Time points
            - 'mu': Magnification values
            - 'event_type': Type of event ('Noise', 'SingleLenseEvent', 'BinaryLenseEvent')
        sequence_length (int): Length to resample all curves to
        interpolation_method (str): Method for curve resampling ('linear', 'savgol', 'bin', 'gp')
        test_fraction (float): Fraction of data to use for testing (0-1)
        valuation_fraction (float): Fraction of training data to use for validation (0-1)
        random_seed (int): Seed for reproducible splits
    
    Returns:
        tuple: Six numpy arrays:
            - X_train: Training features (N_train, sequence_length, 1)
            - y_train: Training labels (N_train, n_classes)
            - X_val: Validation features (N_val, sequence_length, 1)
            - y_val: Validation labels (N_val, n_classes)
            - X_test: Test features (N_test, sequence_length, 1)
            - y_test: Test labels (N_test, n_classes)
            
    Notes:
        - Uses stratified splitting to maintain class distributions
        - Prints summary of resulting dataset sizes
        - Features are 3D arrays ready for Conv1D layers
        - Labels are one-hot encoded
    """

    from sklearn.model_selection import train_test_split    
    
    X = apply_resampling(df, sequence_length, interpolation_method)    
    y = pd.get_dummies(df["event_type"]).to_numpy() # Convert event_type to one-hot encoded format
    # print(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_fraction, 
        shuffle=True, 
        stratify=np.argmax(y, axis=1), 
        random_state=random_seed
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=validation_fraction,
        shuffle=True,
        stratify=np.argmax(y_train, axis=1),
        random_state=random_seed
    )
    
    print("\nData split sizes:")
    print(f"Training: {len(X_train)}")
    print(f"Validation: {len(X_val)}")
    print(f"Test: {len(X_test)}")

    return X_train, y_train, X_val, y_val, X_test, y_test

def model_building(config: dict) -> PersonalizedSequential:
    """
    Build model architecture with optional configuration storage.
    
    Args:
        sequence_length: Length of input sequence
        config: Dictionary containing model configuration parameters and architecture details.
    
    Returns:
        PersonalizedModel: Model with stored configuration
    """
    from tensorflow.keras.layers import InputLayer, Dense, Conv1D, MaxPool1D, Flatten, Dropout
    from tensorflow.keras.metrics import Precision, Recall
    from tensorflow.keras.optimizers import Adam

    print("\nBuilding model")

    layers = [InputLayer(shape=(config['sequence_length'], 1))]
    
    # Add Conv1D-MaxPool1D layers
    for i in range(config['n_layers']):
        layers.extend([
            Conv1D(config['filters'][i], config['kernel_sizes'][i], activation="relu"),
            MaxPool1D(config['pool_sizes'][i])
        ])
    
    layers.extend([
        Flatten(),
        Dense(config['dense_units'], activation="relu"),
        Dropout(0.3),
        Dense(3, activation="softmax")
    ])
    metrics = [
        "accuracy",
        Precision(class_id=0, name='precision_noise'),
        Precision(class_id=1, name='precision_single'),
        Precision(class_id=2, name='precision_binary'),
        Recall(class_id=0, name='recall_noise'),
        Recall(class_id=1, name='recall_single'),
        Recall(class_id=2, name='recall_binary')
    ]

    model = PersonalizedSequential(layers, config=config)
    model.compile(
        optimizer=Adam(1e-3),
        loss="categorical_crossentropy",
        metrics= metrics
    )

    print("\nModel architecture:")
    model.summary()

    return model

def model_training(model: PersonalizedSequential, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: np.ndarray, y_val: np.ndarray,
                   epochs: int, batch_size: int, plotting: bool = True):

    from tensorflow.keras.callbacks import EarlyStopping

    print("Training model")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[EarlyStopping(
            patience=5, 
            restore_best_weights=True,
            monitor='val_loss'
        )]
    )
    if plotting:    plot_training_history(history)

    return history

def model_evaluation(model: PersonalizedSequential, X_test: np.ndarray, y_test: np.ndarray, print_bool: bool = True):
    """Evaluate model performance with detailed per-class metrics."""
    from sklearn.metrics import classification_report

    metrics = model.evaluate(X_test, y_test, verbose=0)
    metric_names = ['loss', 'accuracy',
                   'precision_noise', 'precision_single', 'precision_binary',
                   'recall_noise', 'recall_single', 'recall_binary']
    
    metrics_dict = dict(zip(metric_names, metrics))

    # Get detailed classification report
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    if print_bool:
        print("\nTest Set Metrics:")
        print("-" * 50)
        print(f"Overall Accuracy: {metrics_dict['accuracy']:.4f}")
        print(f"Loss: {metrics_dict['loss']:.4f}")
        
        print("\nPer-class Metrics:")
        print("-" * 50)
        for class_name in ['noise', 'single', 'binary']:
            print(f"\n{class_name.capitalize()} Events:")
            print(f"Precision: {metrics_dict[f'precision_{class_name}']:.4f}")
            print(f"Recall: {metrics_dict[f'recall_{class_name}']:.4f}")
        
    
        
        print("\nDetailed Classification Report:")
        print("-" * 50)
        print(classification_report(
            y_test_classes, 
            y_pred_classes,
            target_names=['Noise', 'Single', 'Binary'],
            digits=4
        ))
    return metrics_dict

def model_saving(model: PersonalizedSequential, filename: str, DIR: Path = ROOT_DIR):
    save_path = DIR / filename
    model.save(save_path)
    print(f"\nSaved model to: {save_path}")

def model_checker(df: pd.DataFrame, model: PersonalizedSequential, sequence_length: int, interpolation_method: str):

    event_types = ['Noise', 'SingleLenseEvent', 'BinaryLenseEvent']

    X = apply_resampling(df, sequence_length, interpolation_method)
        
    predictions = model.predict(X, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
        
    results_df = pd.DataFrame({
        'Event': df.index,
        'Predicted_Type': [event_types[i] for i in pred_classes],
        'Confidence': [predictions[i, pred_classes[i]] for i in range(len(pred_classes))]
    })
        
    results_df = results_df.sort_values('Confidence', ascending=False)
        
    print("\nClassification Results:")
    print("-" * 50)
    for _, row in results_df.iterrows():
        print(f"Event: {row['Event']}")
        print(f"Predicted Type: {row['Predicted_Type']}")
        print(f"Confidence: {row['Confidence']:.1%}")
        print("-" * 50)
        
    print("\nPredicted Type Distribution:")
    type_counts = results_df['Predicted_Type'].value_counts()
    for event_type, count in type_counts.items():
        percentage = count / len(results_df) * 100
        print(f"{event_type}: {count} ({percentage:.1f}%)")

def model_loader(filename: str, DIR: Path = ROOT_DIR) -> PersonalizedSequential:

    from tensorflow.keras.models import load_model

    path = DIR / filename
    model = load_model(path, custom_objects={"PersonalizedSequential": PersonalizedSequential})
    print(f"Loaded model from {path} with configuration: \n{model.config}")
    return model

def model_configuration_setup(model_configuration: dict = {}) -> dict:
    
    defaults = {
        # Data processing parameters
        'sequence_length': 1_000,
        'test_fraction': 0.20,
        'validation_fraction': 0.20,
        'interpolation': 'linear',
        
        # Training parameters
        'batch_size': 32,
        'epochs': 30,
        'learning_rate': 1e-3,
        'use_seed': True,
        'random_seed': 42,
        
        # Model architecture
        'n_layers': 2,
        'filters': [32, 64],
        'kernel_sizes': [5, 5],
        'pool_sizes': [2, 2],
        'dense_units': 64,
        'dropout_rate': 0.3
    }
    
    # Update defaults with provided configuration
    for key, default_value in defaults.items():
        if key not in model_configuration:
            model_configuration[key] = default_value
        elif isinstance(default_value, list):
            # Ensure lists have correct length based on n_layers
            if key in ['filters', 'kernel_sizes', 'pool_sizes']:
                provided_list = model_configuration[key]
                if len(provided_list) != model_configuration.get('n_layers', defaults['n_layers']):
                    raise ValueError(f"{key} length must match n_layers")

    return model_configuration

def check_model_config(model: PersonalizedSequential, config_to_check: dict):
    """
    Compare provided configuration with model's stored configuration.
    
    Args:
        model: PersonalizedModel instance with stored configuration
        config_to_check: Dictionary with configuration to compare
    
    Returns:
        bool: True if configurations match, False otherwise
        dict: Dictionary with mismatched parameters
    """
    if not hasattr(model, 'config'):
        print("Model has no stored configuration")
        return False, {}
    
    model_config = model.config
    mismatches = {}
    
    for key, value in config_to_check.items():
        if key not in model_config:
            print(f"Warning: '{key}' not found in model configuration")
            mismatches[key] = ('not found', value)
        elif model_config[key] != value:
            mismatches[key] = (model_config[key], value)
    
    # Print comparison results
    if mismatches:
        print("\nConfiguration mismatches:")
        print("-" * 50)
        for key, (model_val, check_val) in mismatches.items():
            print(f"{key}:")
            print(f"  Model:     {model_val}")
            print(f"  Provided:  {check_val}")
        return False, mismatches
    else:
        print("\nConfigurations match perfectly!")
        return True, {}
    
def model_predictor(model: PersonalizedSequential, df: pd.DataFrame, sequence_length: int, interpolation_method: str, event_types: list) -> pd.DataFrame:

    config_to_check = {"sequence_length": sequence_length, "interpolation": interpolation_method}
    check_model_config(model, config_to_check)

    X = apply_resampling(df, sequence_length, interpolation_method)

    print(f"Analyzing {len(df)} OGLE events...")    
    predictions = model.predict(X, verbose=1)

    pred_classes = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)
    
    results_df = pd.DataFrame({
        'Predicted_Type': [event_types[i] for i in pred_classes],
        'Confidence': confidences
    }, index=df.index)  

    return results_df

def summary_prediction(df: pd.DataFrame, event_types: list):

    top_predictions = get_top_predictions(df)

    for event_type in event_types:
        print(f"\nTop 20 {event_type} Candidates:")
        print("-" * 50)
        for _, row in top_predictions[event_type].iterrows():
            print(f"Event: {row.index}")
            print(f"Confidence: {row['Confidence']:.1%}")
            print("-" * 50)
    
    total = len(df)
    print(f"\nSummary:")
    print(f"Total events analyzed: {total}")
    for event_type in event_types:
        count = sum(df['Predicted_Type'] == event_type)
        print(f"Predicted {event_type}: {count} ({count/total:.1%})")

    plot_prediction_stats(df)

# ----------------------------------------------------------------------------



