import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Functions.NNWorkflows import ModelComparer
from Functions.NNFunctions import ROOT_DIR

if __name__ == "__main__":
    df_output = None
    df1 = "model_comparison_savgol.csv"
    df2 = "model_comparison_linear.csv"
    results = ModelComparer(df_1=df1, 
                            df_2=df2, 
                            df_output=df_output, 
                            plot_models_file="comparison_plot_savgollinear.pdf",
                            label_1="Savgol", 
                            label_2="Linear",
                            DIR = ROOT_DIR / "Analysis")
    print(results.head())