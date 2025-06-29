import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Functions.NNWorkflows import ArchitecturesEvaluator
from Functions.NNFunctions import load_data, load_architectures_from_csv, ROOT_DIR
from pathlib import Path
import pandas as pd
import numpy as np

if __name__ == "__main__":
    DIR = ROOT_DIR / "MicrolensingData"
    SAVEDIR = ROOT_DIR / "Analysis"
    df = load_data("all_lightcurves.pkl", DIR=DIR)
    print(df)
    
    csv_path = "model_comparison_linear.csv"
    architectures = load_architectures_from_csv(csv_path, DIR = SAVEDIR)
    print(f"\nLoaded {len(architectures)} Architectures")
    data_config = {
    'sequence_length': 1000,
    'interpolation': 'linear',
    'test_fraction': 0.2,
    'validation_fraction': 0.2,
    'use_seed': True,
    'random_seed': 42,
    }
    results_df1 = ArchitecturesEvaluator(df = df, 
                                        architectures = architectures, 
                                        default_config = data_config, 
                                        model_comparison_filename="model_comparison_linear.csv", 
                                        DIR = SAVEDIR,
                                        print_bool=False)
    print("\nArchitecture Comparison Linear:")
    print(results_df1)
    data_config['interpolation'] = "savgol"
    results_df2 = ArchitecturesEvaluator(df = df, 
                                        architectures = architectures, 
                                        default_config = data_config, 
                                        model_comparison_filename="model_comparison_savgol.csv", 
                                        DIR = SAVEDIR,
                                        print_bool=False)
    print("\nArchitecture Comparison Savgol:")
    print(results_df2)

        # Add interpolation method column to distinguish between results
    results_df1['interpolation_method'] = 'linear'
    results_df2['interpolation_method'] = 'savgol'
    
    # Combine dataframes
    combined_results = pd.concat([results_df1, results_df2], ignore_index=True)
    
    # Save combined results
    combined_csv_path = SAVEDIR / "model_comparison.csv"
    combined_results.to_csv(combined_csv_path, index=False)
    
    print(f"Combined results saved to: {combined_csv_path}")
    print(f"Total architectures compared: {len(combined_results)}")
   
