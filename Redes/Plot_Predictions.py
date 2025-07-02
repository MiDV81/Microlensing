import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Functions.NNWorkflows import ModelChecker
from Functions.NNFunctions import ROOT_DIR, resample_curve, normalize_array
import pandas as pd
import matplotlib.pyplot as plt
# Load predictions (assume predictions are saved as CSV with columns: id, predicted_label)
pred_path = ROOT_DIR / "MicrolensingData" / "event_predictions_confirmed.csv"
ogle_path = ROOT_DIR / "MicrolensingData" / "ogle_lightcurves.pkl"
preds = pd.read_csv(pred_path, index_col=0)
print(preds)
ogle_df = pd.read_pickle(ogle_path)
print(ogle_df)
# Select 4 single and 4 binary
single_ids = preds[preds['Predicted_Type'] == 'SingleLenseEvent'].sample(4).index
binary_ids = preds[preds['Predicted_Type'] == 'BinaryLenseEvent'].sample(4).index
print("Single IDs:", list(single_ids))
print("Binary IDs:", list(binary_ids))
print("OGLE index:", ogle_df.index[:10])
print("OGLE columns:", ogle_df.columns)
fig, axes = plt.subplots(2, 4, figsize=(18, 10))

# Add row labels
fig.text(0.05, 0.7, 'Lentes Únicas', fontsize=16, fontweight='bold', rotation=90, va='center')
fig.text(0.05, 0.25, 'Lentes Binarias', fontsize=16, fontweight='bold', rotation=90, va='center')

for i, idx in enumerate(single_ids):
    lc = ogle_df.loc[idx]
    
    # Normalize time to [-1, 1] using NNFunctions normalize_array
    normalized_time = normalize_array(lc['time'], range_type='minus_plus_one')
    
    # Apply savgol interpolation for smoother curves
    smoothed_mu = resample_curve(normalized_time, lc['mu'], sequence_length=1000, interpolation_method='savgol', return_times=False)
    smoothed_time = resample_curve(normalized_time, lc['mu'], sequence_length=1000, interpolation_method='savgol', return_times=True)[0]
    
    # Plot red line for interpolation and blue points with transparency for original data
    axes[0, i].plot(smoothed_time, smoothed_mu, color='red', linewidth=2, label='Interpolación')
    axes[0, i].scatter(normalized_time, lc['mu'], color='blue', alpha=0.6, s=20, label='Datos originales')
    
    # Event name in white box at the bottom center
    axes[0, i].text(0.5, 0.05, idx, transform=axes[0, i].transAxes, fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black', alpha=0.8),
                   ha='center', va='bottom')
    
for i, idx in enumerate(binary_ids):
    lc = ogle_df.loc[idx]
    
   # Normalize time to [-1, 1] using NNFunctions normalize_array
    normalized_time = normalize_array(lc['time'], range_type='minus_plus_one')
    
    # Apply savgol interpolation for smoother curves
    smoothed_mu = resample_curve(normalized_time, lc['mu'], sequence_length=1000, interpolation_method='savgol', return_times=False)
    smoothed_time = resample_curve(normalized_time, lc['mu'], sequence_length=1000, interpolation_method='savgol', return_times=True)[0]
    
    # Plot red line for interpolation and blue points with transparency for original data
    axes[1, i].plot(smoothed_time, smoothed_mu, color='red', linewidth=2, label='Interpolación')
    axes[1, i].scatter(normalized_time, lc['mu'], color='blue', alpha=0.6, s=20, label='Datos originales')
    
    # Event name in white box at the bottom center
    axes[1, i].text(0.5, 0.05, idx, transform=axes[1, i].transAxes, fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black', alpha=0.8),
                   ha='center', va='bottom')

for ax in axes.flat:
    ax.set_xlabel('Tiempo Normalizado', fontsize=10)
    ax.set_ylabel('Magnificación', fontsize=10)
    ax.grid(True, alpha=0.3)

# Add legend to the first subplot
# axes[0, 0].legend(loc='upper right', fontsize=10)

# plt.suptitle('Clasificación de Eventos de Microlensing Gravitacional', fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()
plt.subplots_adjust(left=0.1, top=0.92)

# Save the figure
output_path = ROOT_DIR / "MicrolensingData" / "plots" / "clasificacion_confirmed.pdf"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figura guardada en: {output_path}")
plt.show()