import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Functions.NNFunctions import ROOT_DIR, load_data, resample_curve, normalize_array

def compare_interpolation_methods(event_id: str = None, sequence_length: int = 1000, 
                                DIR: Path = None) -> None:
    """
    Compare different interpolation methods for an OGLE event.
    
    Args:
        event_id: Specific event ID to analyze (if None, picks random)
        sequence_length: Length for resampling
        DIR: Directory path
    """
    
    # Load OGLE data
    df = load_data("ogle_lightcurves.pkl", DIR)
    
    # Select event
    if event_id is None:
        event_id = np.random.choice(df.index)
    
    print(f"Analizando evento: {event_id}")
    
    # Get event data
    times = df.loc[event_id, 'time']
    magnifications = df.loc[event_id, 'mu']
    
    # Interpolation methods to compare
    methods = ['linear', 'spline', 'savgol', 'bin', 'gp']
    method_names = ['Lineal', 'Spline Cúbica', 'Savitzky-Golay', 'Data Binning', 'GPR']
    
    # Create 2x3 subplot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Normalize original times for comparison
    normed_times = normalize_array(times, "minus_plus_one")
    
    # Plot original data in first subplot
    axes[0].plot(normed_times, magnifications, 'b.', alpha=0.6, markersize=3, label='Original')
    # axes[0].set_title('Original Data')
    axes[0].set_xlabel('Tiempo Normalizado')
    axes[0].set_ylabel('Magnificación')
    axes[0].grid(True, alpha=0.3)
    # axes[0].legend()

    # Add "Original Data" label at bottom center with white box
    axes[0].text(0.5, 0.02, 'Datos Originales', transform=axes[0].transAxes, 
                ha='center', va='bottom', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor='black', alpha=0.8))
    
    # Plot each interpolation method
    for i, (method, name) in enumerate(zip(methods, method_names), 1):
        try:
            print(f"Procesando interpolación {name}...")
            times_resampled, mag_resampled = resample_curve(
                times, magnifications, sequence_length, method, return_times=True
            )
            
            # Plot resampled data
            axes[i].plot(times_resampled, mag_resampled, 'r-', linewidth=1.5, label=f'{name}')
            axes[i].plot(normed_times, magnifications, 'b.', alpha=0.4, markersize=2, label='Original')
            # axes[i].set_title(f'{name} Interpolation', fontweight='bold')
            axes[i].set_xlabel('Tiempo Normalizado')
            axes[i].set_ylabel('Magnificación')
            axes[i].grid(True, alpha=0.3)
            # axes[i].legend()
            
            # Add method name at bottom center with white box
            axes[i].text(0.5, 0.02, name, transform=axes[i].transAxes, 
                        ha='center', va='bottom', fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='black', alpha=0.8))
            
        except Exception as e:
            print(f"Error with {name}: {str(e)}")
            axes[i].text(0.5, 0.5, f'Error: {str(e)}', 
                        transform=axes[i].transAxes, ha='center', va='center')
            axes[i].set_title(f'{name} - Falló', fontweight='bold')
            
            # Add method name even for failed plots
            axes[i].text(0.5, 0.02, name, transform=axes[i].transAxes, 
                        ha='center', va='bottom', fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='black', alpha=0.8))

    plt.suptitle(f'Comparación de Métodos de Interpolación - Evento: {event_id}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    # Save to plots directory
    plots_dir = DIR / "plots"
    plots_dir.mkdir(exist_ok=True)
    filename = f"interpolation_comparison_{event_id}.pdf"
    filepath = plots_dir / filename
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    print(f"Gráfico guardado en: {filepath}")
    plt.show()
    
    print(f"Comparación completada para el evento: {event_id}")

if __name__ == "__main__":
    # Compare methods for a single random event
    compare_interpolation_methods(event_id="2011_blg-1150",
                                  sequence_length=1_000,
                                  DIR = ROOT_DIR / "MicrolensingData")