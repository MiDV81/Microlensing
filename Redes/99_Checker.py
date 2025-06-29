from NNFunctions import load_data, model_data_preparation, model_building, model_training
from sklearn.metrics import classification_report
import numpy as np
from pathlib import Path

def evaluate_model_performance(data_config: dict, model_config: dict):
    """
    Build, train and evaluate a model with detailed metrics.
    """
    # Load data
    df = load_data("all_lightcurves.pkl")
    
    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test = model_data_preparation(
        df,
        sequence_length=data_config['sequence_length'],
        interpolation_method=data_config['interpolation'],
        test_fraction=data_config['test_fraction'],
        validation_fraction=data_config['validation_fraction'],
        random_seed=data_config['random_seed']
    )
    # print(df)
    # Build and train model
    print("\nBuilding and training model...")
    model = model_building(model_config)
    history = model_training(
        model, X_train, y_train, X_val, y_val,
        epochs=model_config['epochs'],
        batch_size=model_config['batch_size'],
        plotting=True
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = model.evaluate(X_test, y_test, verbose=0)
    print(metrics)
    # metric_names = ['loss', 'accuracy', 'precision', 'recall']
    
    # print("\nTest Metrics:")
    # print("-" * 50)
    # for name, value in zip(metric_names, metrics):
    #     print(f"{name:15}: {value:.4f}")
    
    # Get detailed classification report
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    print("\nDetailed Classification Report:")
    print("-" * 50)
    print(classification_report(
        y_test_classes, 
        y_pred_classes,
        target_names=['Noise', 'Single', 'Binary'],
        digits=4
    ))

if __name__ == "__main__":
    # Configuration
    data_config = {
        'sequence_length': 1000,
        'interpolation': 'linear',
        'test_fraction': 0.2,
        'validation_fraction': 0.2,
        'use_seed': True,
        'random_seed': 42
    }
    
    model_config = {
        'sequence_length': 1000,
        'n_layers': 2,
        'kernel_sizes': [5, 3],
        'filters': [32, 64],
        'pool_sizes': [2, 2],
        'dense_units': 64,
        'batch_size': 32,
        'epochs': 30
    }
    
    # Run evaluation
    evaluate_model_performance(data_config, model_config)