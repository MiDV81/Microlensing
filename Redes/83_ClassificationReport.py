import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Functions.NNWorkflows import ModelComparer
from Functions.NNFunctions import (load_data, model_data_preparation, 
                                    model_building, load_architectures_from_csv,
                                    ROOT_DIR)
from sklearn.metrics import classification_report
import numpy as np
from typing import Optional

def compare_model_performance(df_model: str, label: str = "Model", 
                            data_config: Optional[dict] = None) -> None:
    """
    Compare model performance using sklearn's classification_report.
    
    Args:
        df_model: CSV file with model configurations
        label: Label for the model in the report
        data_config: Data processing configuration
    """
    # Load data and best model config
    df = load_data("combined_lightcurves.pkl", DIR = ROOT_DIR / "MicrolensingData")
    DIR = ROOT_DIR / "Analysis"
    path = DIR / df_model

    data_config_def = {
        'sequence_length': 1000,
        'test_fraction': 0.2,
        'validation_fraction': 0.2,
        'use_seed': True,
        'random_seed': 42,
    }
    model_config = load_architectures_from_csv(path)[0]
    if data_config:
        model_config.update(data_config)
        model_config.update(data_config_def)
    
    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test = model_data_preparation(
        df, 
        model_config['sequence_length'],
        model_config['interpolation'],
        model_config['test_fraction'],
        model_config['validation_fraction'],
        model_config['random_seed']
    )
    
    # Build and train model
    model = model_building(model_config)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=0)
    
    # Get predictions
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Generate and print classification report
    print(f"\nClassification Report for {label}:")
    print("-" * 50)
    print(classification_report(y_true, y_pred, 
                              target_names=['Noise', 'Single', 'Binary'],
                              digits=4))

if __name__ == "__main__":
    df_output = "30models_savgollinear.csv"
    df1 = "model_comparison_savgol.csv"
    df2 = "model_comparison.csv"
    
    # Add detailed classification reports
    data_config_savgol = {'interpolation': 'savgol'}
    data_config_linear = {'interpolation': 'linear'}
    
    compare_model_performance(df1, "Savgol Model", data_config_savgol)
    compare_model_performance(df2, "Linear Model", data_config_linear)