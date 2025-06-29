import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Functions.NNWorkflows import LightCurvesCombinator
from Functions.NNFunctions import ROOT_DIR

if __name__ == "__main__":
    LightCurvesCombinator(single_filename="singlelenses_lightcurves.pkl",
                          binary_filename="binarylenses_lightcurves.pkl",
                          noise_filename="noise_lightcurves.pkl",
                          final_filename="combined_lightcurves.pkl",
                          DIR = ROOT_DIR / "MicrolensingData")