import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Functions.NNFunctions import (make_noise_curve, resample_curve, 
                                   normalize_array, create_lightcurves_df, get_random_params,
                                   ROOT_DIR)

def plot_event_types_simple(interpolation_method: Optional[str] = None, 
                           sequence_length: int = 1000,
                           DIR: Optional[Path] = None) -> None:
    """
    Plot 4 events of each type (Noise, Single, Binary) in three rows.
    
    Args:
        interpolation_method: 'linear', 'spline', 'savgol', 'bin' or None for original data only
        sequence_length: Length for resampling if interpolation is used
        DIR: Directory path for data
    """
    if DIR is None:
        DIR = ROOT_DIR / "MicrolensingData"
        
    # Generate new curves for each type
    print("Generating new curves...")
    
    # 1. Generate Noise curves
    noise_curves = []
    for i in range(4):
        times, magnifications = make_noise_curve(1000)
        noise_curves.append({
            'id': f'noise_gen_{i+1}',
            'time': times,
            'mu': magnifications,
            'type': 'Ruido'
        })
    
    # 2. Generate Single Lens curves  
    print("Generating single lens curves...")
    single_params = get_random_params(n_samples=4, binary=False, 
                                    filename="filtered_params_stats.json", DIR=DIR)
    single_df = create_lightcurves_df(single_params, binary=False, 
                                    filename="filtered_params_stats.json", DIR=DIR)
    
    single_curves = []
    for i, (idx, row) in enumerate(single_df.iterrows()):
        single_curves.append({
            'id': f'single_gen_{i+1}',
            'time': row['time'],
            'mu': row['mu'],
            'type': 'Lente Simple'
        })
    
    # 3. Generate Binary Lens curves
    print("Generating binary lens curves...")
    binary_params = get_random_params(n_samples=4, binary=True,
                                    filename="filtered_params_stats.json", DIR=DIR)
    binary_df = create_lightcurves_df(binary_params, binary=True,
                                    filename="filtered_params_stats.json", DIR=DIR)
    
    binary_curves = []
    for i, (idx, row) in enumerate(binary_df.iterrows()):
        binary_curves.append({
            'id': f'binary_gen_{i+1}',
            'time': row['time'],
            'mu': row['mu'],
            'type': 'Lente Binaria'
        })
    
    # Combine all curves
    all_curves = [noise_curves, single_curves, binary_curves]
    type_labels = ['Ruido', 'Lente Simple', 'Lente Binaria']
    
    # Create the plot
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    
    # Plot each type in a row
    for row_idx, (curves, label) in enumerate(zip(all_curves, type_labels)):
        
        for col_idx, curve_data in enumerate(curves):
            ax = axes[row_idx, col_idx]
            
            # Get curve data
            times = np.array(curve_data['time'])
            magnifications = np.array(curve_data['mu'])
            normed_times = normalize_array(times, "minus_plus_one")
            
            # Plot original data
            ax.plot(normed_times, magnifications, 'b.', alpha=0.6, markersize=3, 
                   label='Datos Originales')
            
            if interpolation_method is not None:
                # Plot interpolated curve
                times_resampled, mag_resampled = resample_curve(
                    times.tolist(), magnifications.tolist(), 
                    sequence_length, interpolation_method, return_times=True
                )
                
                method_names = {
                    'linear': 'Lineal',
                    'spline': 'Spline Cúbico', 
                    'savgol': 'Savitzky-Golay',
                    'bin': 'Promedio por Bins'
                }
                method_label = method_names.get(interpolation_method, interpolation_method.capitalize())
                
                ax.plot(times_resampled, mag_resampled, 'r-', linewidth=2, 
                       alpha=0.8, label=f'Interpolación {method_label}')
            
            # Customize subplot
            ax.set_xlabel('Tiempo Normalizado')
            ax.set_ylabel('Magnificación')
            ax.grid(True, alpha=0.3)
            
            if col_idx == 0:
                ax.text(-0.18, 0.5, label, transform=ax.transAxes, 
                       fontsize=14, fontweight='bold', rotation=90,
                       verticalalignment='center')
    
    # Set title and save
    if interpolation_method:
        method_names = {
            'linear': 'Lineal',
            'spline': 'Spline Cúbico', 
            'savgol': 'Savitzky-Golay',
            'bin': 'Promedio por Bins'
        }
        method_label = method_names.get(interpolation_method, interpolation_method.capitalize())
        title = f'Comparación de Tipos de Eventos con Interpolación {method_label}'
        filename = f"tipos_eventos_con_{interpolation_method}.pdf"
    else:
        title = 'Simulación de los distintos tipos de eventos'
        filename = f"tipos_eventos_originales.pdf"

    plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    plots_dir = DIR / "plots"
    plots_dir.mkdir(exist_ok=True)
    filepath = plots_dir / filename
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    print(f"Plot saved to {filepath}")
    plt.show()

def main():
    plot_event_types_simple(
        interpolation_method=None, 
        sequence_length=1000, 
        DIR=Path(__file__).parent / "MicrolensingData"
    )

if __name__ == "__main__":
    main()
