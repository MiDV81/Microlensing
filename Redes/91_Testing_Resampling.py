from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Functions.NNWorkflows import ModelBuilder
from Functions.NNFunctions import resample_curve, load_data, ROOT_DIR

ROOT_DIR = ROOT_DIR / "MicrolensingData"

def plot_single_curve(event_id, df, seq_len=10_000, interpolation_type="linear"):
    """Plot original and resampled curve for a single event.
    
    Args:
        event_id: Event ID to plot
        df: DataFrame containing the light curves
        seq_len: Number of points for resampling
        interpolation_type: Type of interpolation to use
    """
    if event_id in df.index:
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Get original data
        t_orig = df.loc[event_id, 'time']
        mu_orig = df.loc[event_id, 'mu']
        
        # Get resampled data - resample_curve returns (2, seq_len) array
        resampled_data = resample_curve(t_orig, mu_orig, seq_len, interpolation_type)
        
        # Extract time and magnification from the 2D array
        t_resampled = resampled_data[0]  # First row is time
        mu_resampled = resampled_data[1]  # Second row is magnification
        
        # Plot original curve
        ax1.plot(t_orig, mu_orig, 'b.', alpha=0.6, label='Original')
        ax1.set_title(f'Original Curve - {event_id}')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Magnification')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot resampled curve
        ax2.plot(t_resampled, mu_resampled, 'r-', label='Resampled')
        ax2.set_title(f'Resampled Curve - {event_id}')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Magnification')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    else:
        print(f"Event {event_id} not found in dataset")

def plot_random_events(df, n_examples):
    """Plot random events from dataset."""
    random_events = np.random.choice(df.index, size=n_examples, replace=False)
    
    for i, idx in enumerate(random_events, 1):
        print(f"\nPlotting random event {i}/{n_examples}")
        print(f"Event: {idx}")
        
        plot_single_curve(idx, df, interpolation_type="savgol")
        plt.suptitle(f"Random Event {i}/{n_examples}")
        
        plt.close()

def plot_predicted_events(df, predictions_df, n_examples):
    """Plot top predicted events of each type."""
    event_types = ['Noise', 'SingleLenseEvent', 'BinaryLenseEvent']
    
    for event_type in event_types:
        print(f"\nPlotting {n_examples} examples of predicted {event_type}")
        
        type_events = predictions_df[predictions_df['Predicted_Type'] == event_type] \
                     .sort_values('Confidence', ascending=False) \
                     .head(n_examples)
        
        for i, (idx, row) in enumerate(type_events.iterrows(), 1):
            print(f"\nPlotting {event_type} example {i}/{n_examples}")
            print(f"Event: {idx}, Confidence: {row['Confidence']:.1%}")
            
            plot_single_curve(idx, df, interpolation_type="savgol")
            plt.suptitle(f"Predicted {event_type} (Confidence: {row['Confidence']:.1%})")
            
            plt.close()

def compare_interpolation_methods(event_id, df, seq_len=10_000):
    """Compare different interpolation methods for a single event."""
    if event_id not in df.index:
        print(f"Event {event_id} not found in dataset")
        return
    
    # Get original data
    t_orig = df.loc[event_id, 'time']
    mu_orig = df.loc[event_id, 'mu']
    
    # Interpolation methods to compare
    methods = ['linear', 'cubic', 'savgol']
    
    # Create figure with subplots for each method
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot original data
    axes[0].plot(t_orig, mu_orig, 'b.', alpha=0.6, label='Original')
    axes[0].set_title(f'Original Data - {event_id}')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Magnification')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot each interpolation method
    for i, method in enumerate(methods, 1):
        # Get resampled data
        resampled_data = resample_curve(t_orig, mu_orig, seq_len, method)
        t_resampled = resampled_data[0]
        mu_resampled = resampled_data[1]
        
        # Plot resampled curve
        axes[i].plot(t_resampled, mu_resampled, 'r-', label=f'{method.capitalize()} Interpolation')
        axes[i].plot(t_orig, mu_orig, 'b.', alpha=0.4, markersize=3, label='Original')
        axes[i].set_title(f'{method.capitalize()} Interpolation')
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Magnification')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()

def analyze_resampling_quality(df, sample_size=50, seq_len=10_000):
    """Analyze the quality of resampling across multiple events."""
    # Sample random events
    sample_events = np.random.choice(df.index, size=sample_size, replace=False)
    
    print(f"Analyzing resampling quality for {sample_size} random events...")
    
    interpolation_errors = {'linear': [], 'cubic': [], 'savgol': []}
    
    for event_id in sample_events:
        t_orig = df.loc[event_id, 'time']
        mu_orig = df.loc[event_id, 'mu']
        
        for method in interpolation_errors.keys():
            try:
                # Get resampled data
                resampled_data = resample_curve(t_orig, mu_orig, seq_len, method)
                
                # Calculate some quality metric (e.g., preservation of peak)
                original_peak = np.max(mu_orig)
                resampled_peak = np.max(resampled_data[1])
                peak_error = abs(original_peak - resampled_peak) / original_peak
                
                interpolation_errors[method].append(peak_error)
                
            except Exception as e:
                print(f"Error processing {event_id} with {method}: {e}")
                continue
    
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(interpolation_errors.keys())
    errors = [interpolation_errors[method] for method in methods]

    ax.boxplot(errors, labels=methods)
    ax.set_title('Peak Preservation Error by Interpolation Method')
    ax.set_ylabel('Relative Peak Error')
    ax.set_xlabel('Interpolation Method')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    for method in methods:
        if interpolation_errors[method]:
            mean_error = np.mean(interpolation_errors[method])
            std_error = np.std(interpolation_errors[method])
            print(f"{method.capitalize()}: Mean error = {mean_error:.4f} ± {std_error:.4f}")

def main():
    # Set up paths
    ROOT_DIR_DATA = Path().absolute().parent / "Simulacion" / "MicrolensingData"
    
    # Load light curves
    df = load_data("ogle_lightcurves.pkl", ROOT_DIR_DATA)
    
    print(f"Loaded {len(df)} light curves")
    print(f"Available events: {df.index[:5].tolist()}...")  # Show first 5 events
    
    # Parameters
    n_examples = 5
    
    while True:
        print("\nSelect mode:")
        print("1. Random events")
        print("2. Predicted events") 
        print("3. Compare interpolation methods")
        print("4. Analyze resampling quality")
        print("5. Exit")
        
        choice = input("Enter choice (1-5): ").strip()
        
        if choice == '1':
            plot_random_events(df, n_examples)
        elif choice == '2':
            try:
                predictions_df = pd.read_csv(ROOT_DIR_DATA / "ogle_event_predictions.csv", index_col=0)
                plot_predicted_events(df, predictions_df, n_examples)
            except FileNotFoundError:
                print("Predictions file not found. Run ModelUser.py first.")
        elif choice == '3':
            event_id = input("Enter event ID to analyze: ").strip()
            if event_id in df.index:
                compare_interpolation_methods(event_id, df)
            else:
                print(f"Event {event_id} not found. Available events: {df.index[:10].tolist()}")
        elif choice == '4':
            analyze_resampling_quality(df)
        elif choice == '5':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()