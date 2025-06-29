import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Functions.NNWorkflows import ModelChecker
from Functions.NNFunctions import ROOT_DIR

if __name__ == "__main__":
    ModelChecker(sequence_length=100, interpolation_method="linear",
                 model_filename="event_classifier.keras",
                 ogle_filename="ogle_lightcurves.pkl",
                 ogle_events_filename="Confirmed_OGLEEvents.txt",
                 DIR=ROOT_DIR / "MicrolensingData")