import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Functions.NNWorkflows import SingleLensesSimulations
from Functions.NNFunctions import ROOT_DIR

if __name__ == "__main__":
    lightcurves_df = SingleLensesSimulations(n_samples = 1_000, 
                                             stats_filename="filtered_params_stats.json",
                                             save_filename="singlelenses_lightcurves.pkl", 
                                             DIR= ROOT_DIR / "MicrolensingData")