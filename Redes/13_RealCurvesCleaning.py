from pathlib import Path
import numpy as np
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Functions.NNFunctions import (load_data, prepare_dataframe, get_filtered_distribution, 
                                   save_to_json, lightcurve_normalizer, save_data, ROOT_DIR)


def main():
    """Main function to process all lightcurves."""

    # Load and prepare data
    DIR = ROOT_DIR / "MicrolensingData"
    file = "OGLE_IV_all_events.pkl.gz"

    raw_df = load_data(file, DIR)
    print(raw_df.columns)

    filtered_df = prepare_dataframe(raw_df)

    # Process all lightcurves and remove failed ones
    processed_df = filtered_df.apply(lightcurve_normalizer, axis=1)
    processed_df = processed_df.dropna()

    print(f"Retained {len(processed_df)} events after filtering for minimum data points")
        
    te_values = filtered_df['tE'].to_numpy()
    u0_values = filtered_df['u0'].to_numpy()
    
     # Get filtered distributions
    te_stats = get_filtered_distribution(te_values)
    u0_stats = get_filtered_distribution(u0_values)

    # If you need both in a single dict
    filtered_stats = {
        'tE': te_stats,
        'u0': u0_stats
    }

    # print(np.mean(len(processed_df["time"])))
    stats_file = "filtered_params_stats.json"
    save_to_json(filtered_stats, stats_file, DIR)
    save_data(processed_df, "ogle_lightcurves.pkl", DIR)

    return processed_df, filtered_stats

       

if __name__ == "__main__":
    processed_df, filtered_stats = main()