import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

# Add the Functions directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Functions.SimulationFunctions import BinaryLens_Data, calculate_caustics_and_critical_curves_binary_lense

def plot_critical_caustic_curves_only(binary_data: BinaryLens_Data, 
                                     num_points: int = 2000, 
                                     save_path: str = None):
    """
    Plot only the critical and caustic curves for a binary lens system.
    
    Args:
        binary_data (BinaryLens_Data): Binary lens system parameters
        num_points (int): Number of points to calculate for the curves
        save_path (str): Path to save the plot (optional)
    """
    print(f"=== Plotting Critical and Caustic Curves Only ===")
    print(f"System parameters: m_t={binary_data.m_t}, q={binary_data.q}, z1={binary_data.z1}")
    
    # Calculate critical and caustic curves
    print("Calculating critical and caustic curves...")
    binary_data = calculate_caustics_and_critical_curves_binary_lense(binary_data, num_points)
    
    # Create figure with #E8E8E8 background
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.patch.set_facecolor('#E8E8E8')
    ax.set_facecolor('#E8E8E8')
    
    # Plot critical curves in blue
    if len(binary_data.critical_points) > 0:
        ax.scatter(binary_data.critical_points.real, binary_data.critical_points.imag, 
                  s=2, c='blue', alpha=0.7, label='Critical Curves')
        print(f"Critical curves plotted: {len(binary_data.critical_points)} points")
    
    # Plot caustic curves in red
    if len(binary_data.caustic_points) > 0:
        ax.scatter(binary_data.caustic_points.real, binary_data.caustic_points.imag, 
                  s=2, c='red', alpha=0.7, label='Caustic Curves')
        print(f"Caustic curves plotted: {len(binary_data.caustic_points)} points")
    
    # Add lens positions as black circles
    if binary_data.m1 is not None and binary_data.m2 is not None:
        if binary_data.z1 is not None and binary_data.z2 is not None:
            # Lens 1
            lens1_size = 0.08 * np.sqrt(binary_data.m1)
            lens1_circle = Circle((binary_data.z1, 0), lens1_size, 
                                 facecolor='black', edgecolor='white', 
                                 linewidth=2, fill=True, zorder=10)
            ax.add_patch(lens1_circle)
            
            # Lens 2
            lens2_size = 0.08 * np.sqrt(binary_data.m2)
            lens2_circle = Circle((binary_data.z2, 0), lens2_size, 
                                 facecolor='black', edgecolor='white', 
                                 linewidth=2, fill=True, zorder=10)
            ax.add_patch(lens2_circle)
            
            # Add lens labels to legend
            ax.plot([], [], 'ko', markersize=8, markeredgecolor='white', 
                   markeredgewidth=2, label=f'Lenses (m₁={binary_data.m1:.1f}, m₂={binary_data.m2:.1f})')
    
    # Configure plot with single lens axis configuration
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1, 2)  # Same as single lens
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, color='gray')
    ax.set_xlabel(r'$\mathbf{x (\theta_\mathrm{E})}$', fontsize=12, fontweight='bold', color='black')
    ax.set_ylabel(r'$\mathbf{y (\theta_\mathrm{E})}$', fontsize=12, fontweight='bold', color='black')
    ax.set_title('Critical and Caustic Curves - Binary Lens System', 
                fontsize=16, fontweight='bold', color='black', pad=20)
    
    # Configure ticks and labels
    ax.tick_params(labelsize=12, colors='black')
    
    # Add legend with background matching the plot
    legend = ax.legend(fontsize=12, loc='upper right', framealpha=0.9, 
                      facecolor='#E8E8E8', edgecolor='black')
    
    # Add system parameters as text box
    param_text = f'q = {binary_data.q:.1f}\nm_t = {binary_data.m_t:.1f}\nz₁ = {binary_data.z1:.1f}'
    ax.text(0.02, 0.98, param_text, transform=ax.transAxes, 
           fontsize=12, color='black', fontweight='bold', 
           verticalalignment='top', horizontalalignment='left',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='#E8E8E8', 
                    alpha=0.8, edgecolor='black'))
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to: {save_path}")
        plt.close()
    else:
        plt.show()
    
    print("=== Critical and Caustic Curves Plot Completed ===")
    return fig, ax

# Main execution
if __name__ == "__main__":
    # Create output directory
    output_dir = "./Simulacion/Images/BinaryLense"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Critical and Caustic Curves Generation ===")
    
    # Define system parameters as requested
    m_t = 1.0   # Total mass
    q = 1.0     # Mass ratio
    z1 = 0.0    # Position of first lens
    
    # Create binary lens system
    binary_system = BinaryLens_Data(
        m_t=m_t, 
        q=q, 
        z1=z1,
        start_point=(-2, -2),  # Not used for curve calculation, but required
        end_point=(2, 2),      # Not used for curve calculation, but required
        num_points=100         # Not used for curve calculation, but required
    )
    
    print(f"Binary lens system created:")
    print(f"  - Total mass (m_t): {binary_system.m_t}")
    print(f"  - Mass ratio (q): {binary_system.q}")
    print(f"  - First lens position (z1): {binary_system.z1}")
    print(f"  - Individual masses: m1={binary_system.m1:.2f}, m2={binary_system.m2:.2f}")
    print(f"  - Lens positions: z1={binary_system.z1:.2f}, z2={binary_system.z2:.2f}")
    
    # Generate the plot
    save_path = os.path.join(output_dir, 'critical_caustic_curves.png')
    
    plot_critical_caustic_curves_only(
        binary_data=binary_system,
        num_points=2000,  # High resolution for smooth curves
        save_path=save_path
    )
    
    print(f"Plot saved to: {save_path}")
