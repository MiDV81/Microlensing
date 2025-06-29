import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from typing import Tuple, Optional
from tensorflow.keras.callbacks import History
import sys, os
import time 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Functions.NNFunctions import *
from Functions.NNFunctions import ROOT_DIR
from Functions.NNClass import PersonalizedSequential


stats_filename_v = 'filtered_params_stats.json'
save_filename_single = "singlelense_lightcurves.pkl"
save_filename_binary = "binarylense_lightcurves.pkl"
save_filename_noise = "noise_lightcurves.pkl"
save_filename_combined = "all_lightcurves.pkl"
save_filename_ogle = "ogle_lightcurves.pkl"
ogle_events_filename_v = "Confirmed_OGLEEvents.txt"
model_filename_v = "event_classifier.keras"
save_filename_predictions = "event_predictions.csv"
df_previousmodels_v = "model_comparison.csv"
df_newmodels_v = "model_newrun.csv"
plot_models_file_v = "model_comparison_analysis.pdf"

def SingleLensesSimulations(n_samples: int, stats_filename: str = stats_filename_v, 
                            save_filename: str = save_filename_single, DIR: Path = ROOT_DIR) -> pd.DataFrame:
    print("Starting Single Lenses Simulations")
    random_params = get_random_params(n_samples, binary = False, filename = stats_filename, 
                                      DIR = DIR)
    df = create_lightcurves_df(random_params, binary = False)
    save_data(df, filename = save_filename, DIR = DIR)
    
    return df

def BinaryLensesSimulations(n_samples: int, stats_filename: str = stats_filename_v,
                            save_filename: str = save_filename_binary, DIR: Path = ROOT_DIR) -> pd.DataFrame:
    print("Starting Binary Lenses Simulations")
    random_params = get_random_params(n_samples, binary = True, filename = stats_filename, DIR = DIR)
    df = create_lightcurves_df(random_params, binary=True, filename = stats_filename, DIR = DIR)
    save_data(df, filename = save_filename, DIR = DIR)
    return df

def NoiseSimulations(n_sample: int, filename: str = save_filename_noise, 
                     DIR: Path = ROOT_DIR) -> pd.DataFrame:
    print("Starting Noise Simulations")
    data = []
    index = []
    
    N_POINTS_MIN = 250
    N_POINTS_MAX = 450
    RNG = np.random.default_rng(42)

    print("Noise Dataframe creation started")
    for i in range(n_sample):
        number_points = RNG.integers(N_POINTS_MIN, N_POINTS_MAX + 1) 
        times, magnifications = make_noise_curve(number_points, seed=i) 
        # magnifications = normalize_array(magnifications, "zero_one")
        
        data.append({
            "tE": 0,
            "u0": 0,
            "time": times,
            "mu": magnifications
        })
        index.append(f"noise_{i:04d}")

    df = pd.DataFrame(data, index=index)
    
    print(f"Created {len(df)} noise curves")
    save_data(df, filename, DIR)

    return df


def LightCurvesCombinator(single_filename: str = save_filename_single,
                          binary_filename: str = save_filename_binary, 
                          noise_filename: str = save_filename_noise,
                          final_filename: str = save_filename_combined,
                          DIR: Path = ROOT_DIR) -> pd.DataFrame:
    print("Combining Light Curves")

    single_df = load_data(single_filename, DIR)
    binary_df = load_data(binary_filename, DIR)
    noise_df = load_data(noise_filename, DIR)

    single_df['event_type'] = 'SingleLenseEvent'
    binary_df['event_type'] = 'BinaryLenseEvent'
    noise_df['event_type'] = 'Noise'

    for df in [single_df, binary_df, noise_df]:
        df['event_type'] = pd.Categorical(
            df['event_type'], 
            categories=['Noise', 'SingleLenseEvent', 'BinaryLenseEvent']
        )
    df = pd.concat([single_df, binary_df, noise_df], ignore_index=False)

    save_data(df, final_filename, DIR)
    return df

def ModelBuilder(model_configuration: dict =  {}, 
                 load_filename: str = save_filename_combined, model_filename: str = model_filename_v,
                 DIR: Path = ROOT_DIR) -> Tuple[History, PersonalizedSequential]:

    model_configuration = model_configuration_setup(model_configuration)

    SEQ_LEN = model_configuration['sequence_length']
    TEST_FRAC = model_configuration['test_fraction']
    VAL_FRAC = model_configuration['validation_fraction']
    BATCH = model_configuration['batch_size']
    EPOCHS = model_configuration['epochs']
    USE_SEED = model_configuration['use_seed']
    RANDOM_SEED = model_configuration['random_seed'] if USE_SEED else None
    interp = model_configuration['interpolation']
        
    df = load_data(load_filename, DIR)

    event_type_map = {
        'Noise': 0,
        'SingleLenseEvent': 1,
        'BinaryLenseEvent': 2
    }    
    print("\nDataset summary:")
    print(f"Total curves: {len(df)}")
    for event_type in event_type_map:
        print(f"{event_type} curves: {sum(df['event_type'] == event_type)}")
    
    X_train, y_train, X_val, y_val, X_test, y_test = model_data_preparation(df, sequence_length=SEQ_LEN, 
                                    interpolation_method=interp, test_fraction=TEST_FRAC, 
                                    validation_fraction=VAL_FRAC, random_seed=RANDOM_SEED or 42)

    model = model_building(config=model_configuration)
    
    history = model_training(model, X_train, y_train, X_val, y_val, 
                            epochs = EPOCHS, batch_size = BATCH, plotting= True)    
    
    model_evaluation(model, X_test, y_test, print_bool=True)

    model_saving(model, model_filename, DIR)

    return history, model

def ModelChecker(sequence_length: int, interpolation_method: str = "linear",
                 model_filename: str = model_filename_v, ogle_filename: str = save_filename_ogle,
                 ogle_events_filename: str = ogle_events_filename_v,
                 DIR: Path = ROOT_DIR) -> None:
    
    model = model_loader(model_filename, DIR)
    config_to_check = {"sequence_length": sequence_length, "interpolation": interpolation_method}
    check_model_config(model, config_to_check)

    df = load_data(ogle_filename, DIR)
    
    confirmed_events = load_events_from_txt(ogle_events_filename, DIR)    
    
    confirmed_df = df[df.index.isin(confirmed_events)]
    print(f"Found {len(confirmed_df)} matching light curves")
    
    if len(confirmed_df) > 0:
        model_checker(confirmed_df, model, sequence_length, interpolation_method)    
    else:
        print("No matching events found in dataset. Check index formatting.")

def ModelUser(sequence_length: int, interpolation_method: str = "linear",
              model_filename: str = model_filename_v, data_filename: str = save_filename_ogle, 
              csv_out_filename: str = save_filename_predictions,
              DIR: Path = ROOT_DIR) -> pd.DataFrame:
    
    model = model_loader(model_filename, DIR)
    df = load_data(data_filename, DIR)
        
    event_types = ['Noise', 'SingleLenseEvent', 'BinaryLenseEvent'] 
    results_df = model_predictor(model, df, sequence_length, interpolation_method, event_types)

    save_to_csv(results_df, csv_out_filename, DIR)

    return results_df


def ModelComparer(df_1: str = df_previousmodels_v, 
                  df_2: str = df_newmodels_v, 
                  plot_models_file: str = plot_models_file_v,
                  df_output: Optional[str] = df_previousmodels_v,
                  label_1: Optional[str] = None, label_2: Optional[str] = None,
                  DIR: Path = ROOT_DIR) -> pd.DataFrame:
    """
    Compare models from previous runs with new models and plot analysis. Also add the new models to the existing CSV file.
    Args:
        df_previousmodels (str): Name of the CSV file with previous model data.
        df_newmodels (str): Name of the CSV file with new model data.
        plot_models_file (str): Name of the output plot file.
        DIR (Path): Directory (Used directory is DIR / "analysis")."""

    df1 = pd.read_csv(DIR / df_1)
    df2 = pd.read_csv(DIR / df_2)
    if label_1 is not None:
        df1['source'] = label_1
    if label_2 is not None:
        df2['source'] = label_2

    combined_df = pd.concat([df1, df2], ignore_index=True)

    # Print comparison table in terminal
    print("\n=== MODEL COMPARISON TABLE ===")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(combined_df.to_string(index=False))
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    
    plot_model_analysis(combined_df, plot_models_file, DIR)

    if df_output is not None:
        save_to_csv(combined_df, df_output, DIR)

    return combined_df

data_config = {
    'sequence_length': 1000,
    'interpolation': 'linear',
    'test_fraction': 0.2,
    'validation_fraction': 0.2,
    'use_seed': True,
    'random_seed': 42,
}

def ArchitecturesEvaluator(df: pd.DataFrame, architectures: list, 
                           default_config: dict = data_config, 
                           model_comparison_filename: str = df_newmodels_v, 
                           DIR: Path = ROOT_DIR, print_bool: bool = True) -> pd.DataFrame:
    
    """Test different architectures and collect metrics."""
    results = []
    histories = []
    training_times = []

    random_seed = default_config['random_seed'] if default_config['use_seed'] else None
    # Prepare data once
    X_train, y_train, X_val, y_val, X_test, y_test = model_data_preparation(
        df, default_config['sequence_length'], default_config['interpolation'],
        test_fraction=default_config['test_fraction'], 
        validation_fraction=default_config['validation_fraction'],
        random_seed=random_seed or 42
    )
    
    for i, arch in enumerate(architectures, 1):
        print(f"\nTesting architecture {i}/{len(architectures)}:")
        print(f"Layers: {arch['n_layers']}, Filters: {arch['filters']}")
        
        # Add default_config parameters to arch
        arch.update({
            'sequence_length': default_config['sequence_length'],
            'interpolation': default_config['interpolation'],
            'test_fraction': default_config['test_fraction'],
            'validation_fraction': default_config['validation_fraction'],
            'use_seed': default_config['use_seed'],
            'random_seed': default_config['random_seed']
        })
        
        start_time = time.time()
        arch = model_configuration_setup(arch)
        model = model_building(arch)
        
        history = model.fit(
            X_train, y_train,
            epochs=arch['epochs'],
            batch_size=arch['batch_size'],
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        training_time = time.time() - start_time
        histories.append(history)
        training_times.append(training_time)
        
        # Evaluate and store results
        metrics_dict = model_evaluation(model, X_test, y_test, print_bool=print_bool)
        model_dict = {
            'model': i,
            'n_layers': arch['n_layers'],
            'filters': arch['filters'],
            'kernel_sizes': arch['kernel_sizes'],
            'pool_sizes': arch['pool_sizes'],
            'dense_units': arch['dense_units'],
            'epochs': arch['epochs'],
            'batch_size': arch['batch_size'],
            'training_time': training_time,
            'best_val_accuracy': max(history.history['val_accuracy']),
            'best_val_loss': min(history.history['val_loss']),
            'params': model.count_params()
        }
        model_dict.update(metrics_dict)
        results.append(model_dict)
        print(model_dict)   
    results = pd.DataFrame(results)     
    save_to_csv(results, model_comparison_filename, DIR)
    plot_combined_histories(histories, list(range(1, len(architectures) + 1)), training_times)
    
    
    
    return results