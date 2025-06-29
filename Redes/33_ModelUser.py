import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Functions.NNWorkflows import ModelUser
from Functions.NNFunctions import ROOT_DIR

if __name__ == "__main__":
    ModelUser(sequence_length=100, 
              interpolation_method="linear",
              model_filename="event_classifier.keras",
              data_filename="ogle_lightcurves.pkl",
              csv_out_filename="event_predictions.csv",
              DIR=ROOT_DIR / "MicrolensingData")