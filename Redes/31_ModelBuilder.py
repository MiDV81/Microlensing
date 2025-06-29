import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Functions.NNWorkflows import ModelBuilder
from Functions.NNFunctions import ROOT_DIR

if __name__ == "__main__":
    model_configuration = {
        "sequence_length": 100,
        "test_fraction": 0.2,
        "validation_fraction": 0.1,
        "batch_size": 32,
        "epochs": 2,
        "use_seed": True,
        "random_seed": 42,
        "interpolation": "linear"
    }
    ModelBuilder(model_configuration=model_configuration,
                 load_filename="combined_lightcurves.pkl",
                 model_filename="event_classifier.keras",
                 DIR=ROOT_DIR / "MicrolensingData",)