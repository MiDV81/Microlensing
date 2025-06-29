import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import sys
import os
from typing import Optional

# Add the Functions directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Functions.SimulationFunctions import BinaryLens_Data
from Functions.NNFunctions import get_trajectory_extreme_points, is_binary_lens


"""
Script to test the binary lens detection function with random curves. It generates random binary lens parameters, calculates the magnification curve,
and uses the is_binary_lens function to classify the curve as binary or single lens. 
"""

def test_binary_lens_detection(n_tests: int = 10, seed: Optional[int] = None):
    """
    Test the binary lens detection function with random curves.
    
    Args:
        n_tests: Number of random curves to test
        seed: Random seed for reproducibility (optional)
    """
    if seed is not None:
        np.random.seed(seed)
    
    print(f"\n🔍 Testing binary lens classification with {n_tests} random curves")
    print("=" * 80)
    
    for i in range(n_tests):
        print(f"\n📊 Test {i+1}/{n_tests}:")
        print("-" * 50)
        
        # Generate random parameters
        q = 10**np.random.normal(-2.5, 1.0)  # Mass ratio
        d = np.random.uniform(0.1, 2.4)      # Separation distance
        u0 = np.random.uniform(0.01, 1.0)    # Impact parameter
        alpha = np.random.uniform(0, 2*np.pi) # Trajectory angle
        
        print(f"📌 Parameters: q={q:.3e}, d={d:.2f}, u0={u0:.3f}, α={alpha:.2f} rad")
        
        # Calculate trajectory points
        start_points, end_points = get_trajectory_extreme_points(
            np.array([u0]), 
            np.array([alpha]), 
            np.array([d])
        )
        
        try:
            # Create BinaryLens_Data object
            binary_data = BinaryLens_Data(
                m_t=1.0,
                q=q,
                d=d,
                start_point=start_points[0],
                end_point=end_points[0],
                num_points=1000
            )
            
            # Calculate magnification
            binary_data.images_magnification_calculation()
            magnifications = binary_data.magnification
            
            # Test the binary lens detection function with verbose output
            print("\n🔬 Binary lens classification analysis:")
            is_binary = is_binary_lens(magnifications, verbose=True)
            
            # Plot the curve with analysis
            plot_curve_analysis(magnifications, q, d, is_binary, test_number=i+1)
            
            # Print final result
            result_text = "🔴 BINARY LENS" if is_binary else "🔵 SINGLE LENS"
            print(f"\n✅ FINAL RESULT: {result_text}")
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ Error generating curve: {e}")
            print("-" * 50)
            continue

def plot_curve_analysis(magnifications, q, d, is_binary, test_number=1):
    """
    Plot magnification curve with derivative analysis and detection results.
    
    Args:
        magnifications: Array of magnification values
        q: Mass ratio
        d: Separation distance
        is_binary: Result from is_binary_lens function
        test_number: Test number for plot title
    """
    times = np.linspace(0, 1, len(magnifications))
    derivative = np.gradient(magnifications, times)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Magnification curve with peaks
    ax1.plot(times, magnifications, 'k-', linewidth=2, label='Magnification')
    
    # Find magnification peaks
    normalized_mag = (magnifications - np.min(magnifications)) / (np.max(magnifications) - np.min(magnifications))
    min_distance = max(1, len(magnifications) // 20)
    mag_peaks, _ = find_peaks(normalized_mag, prominence=0.1, distance=min_distance)
    
    if len(mag_peaks) > 0:
        ax1.plot(times[mag_peaks], magnifications[mag_peaks], 'bo', markersize=8, 
                label=f'Mag peaks ({len(mag_peaks)})')
    
    ax1.set_ylabel('Magnification')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Calculate and display norms for symmetry analysis if single peak
    norm_text = ""
    if len(mag_peaks) == 1:
        peak_idx = mag_peaks[0]
        left_boundary = 0
        right_boundary = len(times) - 1
        left_distance = peak_idx - left_boundary
        right_distance = right_boundary - peak_idx
        analysis_distance = min(left_distance, right_distance)
        
        if analysis_distance >= 5:
            # Extract derivative segments around the peak
            left_start = peak_idx - analysis_distance
            right_end = peak_idx + analysis_distance
            
            left_derivative = derivative[left_start:peak_idx]
            right_derivative = derivative[peak_idx+1:right_end+1]
            
            # Normalize derivatives
            left_normalized = left_derivative / np.max(np.abs(left_derivative)) if np.max(np.abs(left_derivative)) > 0 else left_derivative
            right_normalized = right_derivative / np.max(np.abs(right_derivative)) if np.max(np.abs(right_derivative)) > 0 else right_derivative
            right_inverted_reversed = -right_normalized[::-1]
            
            # Calculate norms
            if len(left_normalized) == len(right_inverted_reversed):
                diff = left_normalized - right_inverted_reversed
            else:
                min_length = min(len(left_normalized), len(right_inverted_reversed))
                diff = left_normalized[-min_length:] - right_inverted_reversed[:min_length]
            
            L1_norm = np.linalg.norm(diff, ord=1)
            Linf_norm = np.linalg.norm(diff, ord=np.inf)
            norm_text = f"L1={L1_norm:.2f}, L∞={Linf_norm:.3f}"
    
    # Plot 2: Derivative with peaks
    ax2.plot(times, derivative, 'r-', linewidth=2, label='Derivative')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Find derivative peaks
    min_distance_deriv = max(1, len(magnifications) // 50)
    deriv_peaks, _ = find_peaks(np.abs(derivative), prominence=0.2, distance=min_distance_deriv)
    
    if len(deriv_peaks) > 0:
        ax2.plot(times[deriv_peaks], derivative[deriv_peaks], 'ms', markersize=8, 
                label=f'Derivative peaks ({len(deriv_peaks)})')
    
    ax2.set_xlabel('Normalized Time')
    ax2.set_ylabel('Derivative')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Set title with all information
    result_text = "BINARY" if is_binary else "SINGLE"
    title = f"Test {test_number}: {result_text} - q={q:.3e}, d={d:.2f}"
    if norm_text:
        title += f"\n{norm_text}"
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def main():
    """Main function to test the binary lens detection."""
    print("=" * 60)
    print(" Binary Lens Classification Function Checker & Visualizer")
    print("=" * 60)
    print("\nThis tool generates random binary lens curves and tests the")
    print("is_binary_lens() classification function. Each curve shows:")
    print("• Magnification peaks and derivative peaks")
    print("• L1 and L∞ norms for symmetry analysis")
    print("• Classification result (BINARY/SINGLE)")
    print("\n" + "-" * 60)
    
    # Get user input for number of tests
    try:
        n_tests = int(input("Enter number of tests (default 5): ") or "5")
    except ValueError:
        n_tests = 5
        print("Invalid input, using default: 5 tests")
    
    # Ask about random seed
    seed_input = input("Enter random seed (press Enter for random): ")
    seed = int(seed_input) if seed_input.strip() else None
    
    # Run tests
    test_binary_lens_detection(n_tests=n_tests, seed=seed)
    
    print("\n" + "=" * 60)
    print(f"✓ Completed {n_tests} classification tests!")
    print("=" * 60)

if __name__ == "__main__":
    main()