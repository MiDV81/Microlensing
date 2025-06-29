import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Functions.NNWorkflows import NoiseSimulations
from Functions.NNFunctions import ROOT_DIR

if __name__ == "__main__":
    df_noise = NoiseSimulations(n_sample = 1_000,
                                filename="noise_lightcurves.pkl",
                                DIR=ROOT_DIR / "MicrolensingData")