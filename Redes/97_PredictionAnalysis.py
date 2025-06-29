import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Functions.NNFunctions import load_data, plot_curve_with_resampling, ROOT_DIR

def analyze_event_predictions(predictions_file: str = "event_predictions.csv",
                             ogle_data_file: str = "ogle_lightcurves.pkl",
                             DIR: Path = None, n_examples: int = 5) -> None:
    """
    Analyze event predictions and plot binary lens event examples.
    
    Args:
        predictions_file: CSV file with predictions
        ogle_data_file: PKL file with OGLE light curves
        DIR: Directory path
        n_examples: Number of binary lens examples to plot
    """
    if DIR is None:
        DIR = ROOT_DIR / "MicrolensingData"
    
    # Load predictions
    predictions_df = pd.read_csv(DIR / predictions_file, index_col=0)
    
    # Calculate percentages
    total_events = len(predictions_df)
    type_counts = predictions_df['Predicted_Type'].value_counts()
    percentages = (type_counts / total_events * 100).round(2)
    
    print("\nEvent Type Distribution:")
    print("-" * 40)
    for event_type, count in type_counts.items():
        pct = percentages[event_type]
        print(f"{event_type}: {count} events ({pct}%)")
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Pie chart of event types
    colors = ['lightcoral', 'lightgreen', 'lightblue']
    wedges, texts, autotexts = ax1.pie(type_counts.values, 
                                      labels=type_counts.index,
                                      autopct='%1.1f%%',
                                      colors=colors,
                                      startangle=90)
    ax1.set_title('Event Type Distribution', fontsize=14, fontweight='bold')
    
    # 2. Bar chart of event counts
    bars = ax2.bar(type_counts.index, type_counts.values, color=colors)
    ax2.set_title('Event Counts by Type', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Events')
    ax2.grid(True, alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars, type_counts.values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + total_events*0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Confidence distribution by event type
    for event_type, color in zip(type_counts.index, colors):
        mask = predictions_df['Predicted_Type'] == event_type
        confidences = predictions_df.loc[mask, 'Confidence']
        ax3.hist(confidences, bins=20, alpha=0.7, label=event_type, 
                color=color, edgecolor='black', linewidth=0.5)
    
    ax3.set_title('Confidence Distribution by Event Type', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Confidence')
    ax3.set_ylabel('Number of Events')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. High-confidence binary events
    binary_events = predictions_df[predictions_df['Predicted_Type'] == 'BinaryLenseEvent']
    high_conf_binary = binary_events.sort_values('Confidence', ascending=False).head(20)
    
    ax4.scatter(range(len(high_conf_binary)), high_conf_binary['Confidence'], 
               color='lightblue', s=60, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax4.set_title('Top 20 Binary Lens Event Confidences', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Rank')
    ax4.set_ylabel('Confidence')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    # Save the summary plot
    plots_dir = DIR / "plots"
    plots_dir.mkdir(exist_ok=True)
    summary_path = plots_dir / "event_predictions_analysis.pdf"
    plt.savefig(summary_path, bbox_inches='tight', dpi=300)
    print(f"\nSummary plot saved to: {summary_path}")
    plt.show()
    
    # Plot examples of binary lens events
    plot_binary_examples(predictions_df, ogle_data_file, DIR, n_examples)

def plot_binary_examples(predictions_df: pd.DataFrame, ogle_data_file: str, 
                         DIR: Path, n_examples: int = 5) -> None:
    """Plot examples of high-confidence binary lens events."""
    
    # Load OGLE data
    try:
        ogle_df = load_data(ogle_data_file, DIR)
    except Exception as e:
        print(f"Error loading OGLE data: {e}")
        return
    
    # Get high-confidence binary events that exist in OGLE data
    binary_events = predictions_df[predictions_df['Predicted_Type'] == 'BinaryLenseEvent']
    binary_events = binary_events.sort_values('Confidence', ascending=False)
    
    # Find events that exist in both datasets
    available_events = []
    for event_id in binary_events.index:
        if event_id in ogle_df.index:
            available_events.append(event_id)
        if len(available_events) >= n_examples:
            break
    
    if not available_events:
        print("No binary lens events found in OGLE dataset")
        return
    
    print(f"\nPlotting {len(available_events)} Binary Lens Event Examples:")
    print("=" * 60)
    
    for i, event_id in enumerate(available_events, 1):
        confidence = predictions_df.loc[event_id, 'Confidence']
        print(f"\nEvent {i}: {event_id}")
        print(f"Confidence: {confidence:.1%}")
        
        # Get light curve data
        times = ogle_df.loc[event_id, 'time']
        magnifications = ogle_df.loc[event_id, 'mu']
        
        # Plot the light curve
        title = f"Binary Lens Event {i}/{len(available_events)}\nID: {event_id} | Confidence: {confidence:.1%}"
        plot_curve_with_resampling(times, magnifications, 
                                  sequence_length=1000, 
                                  interpolation_method='savgol', 
                                  title=title)

def get_detailed_statistics(predictions_file: str = "event_predictions.csv", 
                           DIR: Path = None) -> dict:
    """Get detailed statistics about the predictions."""
    if DIR is None:
        DIR = ROOT_DIR / "MicrolensingData"
    
    predictions_df = pd.read_csv(DIR / predictions_file, index_col=0)
    
    stats = {}
    for event_type in predictions_df['Predicted_Type'].unique():
        mask = predictions_df['Predicted_Type'] == event_type
        confidences = predictions_df.loc[mask, 'Confidence']
        
        stats[event_type] = {
            'count': len(confidences),
            'percentage': len(confidences) / len(predictions_df) * 100,
            'mean_confidence': confidences.mean(),
            'median_confidence': confidences.median(),
            'min_confidence': confidences.min(),
            'max_confidence': confidences.max(),
            'std_confidence': confidences.std()
        }
    
    # Print detailed statistics
    print("\nDetailed Statistics:")
    print("=" * 80)
    for event_type, stat in stats.items():
        print(f"\n{event_type}:")
        print(f"  Count: {stat['count']} ({stat['percentage']:.1f}%)")
        print(f"  Confidence - Mean: {stat['mean_confidence']:.3f}, Median: {stat['median_confidence']:.3f}")
        print(f"  Confidence - Min: {stat['min_confidence']:.3f}, Max: {stat['max_confidence']:.3f}")
        print(f"  Confidence - Std: {stat['std_confidence']:.3f}")
    
    return stats

if __name__ == "__main__":
    # Set working directory
    DIR = ROOT_DIR / "MicrolensingData"
    
    print("Event Predictions Analysis")
    print("=" * 50)
    
    # Analyze predictions and plot examples
    analyze_event_predictions(
        predictions_file="event_predictions.csv",
        ogle_data_file="ogle_lightcurves.pkl", 
        DIR=DIR,
        n_examples=3  # Number of binary lens examples to plot
    )
    
    # Get detailed statistics
    stats = get_detailed_statistics("event_predictions.csv", DIR)
    
    # Print high-confidence binary events
    predictions_df = pd.read_csv(DIR / "event_predictions.csv", index_col=0)
    binary_events = predictions_df[predictions_df['Predicted_Type'] == 'BinaryLenseEvent']
    top_binary = binary_events.sort_values('Confidence', ascending=False).head(10)
    
    print(f"\nTop 10 Binary Lens Candidates:")
    print("-" * 40)
    for event_id, row in top_binary.iterrows():
        print(f"{event_id}: {row['Confidence']:.1%}")