import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Functions.NNWorkflows import ModelChecker
from Functions.NNFunctions import ROOT_DIR
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ModelChecker(sequence_length=1000, interpolation_method="linear",
                 model_filename="event_classifier.keras",
                 ogle_filename="ogle_lightcurves.pkl",
                 ogle_events_filename="Confirmed_OGLEEvents.txt",
                 csv_out_filename="event_predictions_confirmed.csv",
                 DIR=ROOT_DIR / "MicrolensingData")

