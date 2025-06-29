import pandas as pd
import numpy as np
from pathlib import Path
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Functions.NNFunctions import ROOT_DIR

def csv_to_latex_table(csv_path: str, output_path: str = None, top_n: int = 15):
    """
    Convert top N models from CSV to LaTeX table format.
    
    Args:
        csv_path: Path to the CSV file
        output_path: Path to save the LaTeX file (optional)
        top_n: Number of top models to include
    """
    
    # Load the CSV
    df = pd.read_csv(csv_path)
    
    # Sort by accuracy (descending) and get top N
    top_models = df.nlargest(top_n, 'accuracy')
    
    # Select and rename columns for the table
    columns_to_include = [
        'model', 'n_layers', 'filters', 'kernel_sizes', 'pool_sizes', 
        'dense_units', 'epochs', 'batch_size', 'training_time', 
        'accuracy', 'loss',
        'precision_noise', 'precision_single', 'precision_binary',
        'recall_noise', 'recall_single', 'recall_binary',
        'params', 'interpolation_method'
    ]
    
    # Check which columns exist
    available_columns = [col for col in columns_to_include if col in top_models.columns]
    
    # If interpolation_method doesn't exist, use 'source'
    if 'interpolation_method' not in available_columns and 'source' in top_models.columns:
        available_columns = [col if col != 'interpolation_method' else 'source' for col in available_columns if col != 'interpolation_method']
        available_columns.append('source')
    
    table_df = top_models[available_columns].copy()
    
    # Format numerical columns with proper decimal places
    if 'accuracy' in table_df.columns:
        table_df['accuracy'] = table_df['accuracy'].apply(lambda x: f"{x:.4f}")
    
    if 'loss' in table_df.columns:
        table_df['loss'] = table_df['loss'].apply(lambda x: f"{x:.4f}")
    
    if 'training_time' in table_df.columns:
        table_df['training_time'] = table_df['training_time'].apply(lambda x: f"{x:.1f}")

    # Combine precision columns into a list
    if all(col in table_df.columns for col in ['precision_noise', 'precision_single', 'precision_binary']):
        table_df['precision'] = table_df.apply(lambda row: 
            f"[{row['precision_noise']:.3f}, {row['precision_single']:.3f}, {row['precision_binary']:.3f}]", axis=1)
        table_df = table_df.drop(['precision_noise', 'precision_single', 'precision_binary'], axis=1)
    
    # Combine recall columns into a list
    if all(col in table_df.columns for col in ['recall_noise', 'recall_single', 'recall_binary']):
        table_df['recall'] = table_df.apply(lambda row: 
            f"[{row['recall_noise']:.3f}, {row['recall_single']:.3f}, {row['recall_binary']:.3f}]", axis=1)
        table_df = table_df.drop(['recall_noise', 'recall_single', 'recall_binary'], axis=1)
      
    # Format large numbers (params)
    if 'params' in table_df.columns:
        table_df['params'] = table_df['params'].apply(lambda x: f"{x:,.0f}")
    
    # Reorder columns to ensure precision and recall come after accuracy
    desired_order = [
        'model', 'n_layers', 'filters', 'kernel_sizes', 'pool_sizes', 
        'dense_units', 'epochs', 'batch_size', 'training_time', 
        'accuracy', 'loss', 'precision', 'recall',
        'params'
    ]
    
    # Add interpolation method at the end
    if 'interpolation_method' in table_df.columns:
        desired_order.append('interpolation_method')
    elif 'source' in table_df.columns:
        desired_order.append('source')
    
    # Keep only columns that exist in the dataframe
    final_order = [col for col in desired_order if col in table_df.columns]
    table_df = table_df[final_order]
    # Rename columns to Spanish
    column_names = {
        'model': 'Modelo',
        'n_layers': 'Capas',
        'filters': 'Filtros',
        'kernel_sizes': 'Kernels',
        'pool_sizes': 'Pooling',
        'dense_units': 'Densas',
        'epochs': 'Épocas',
        'batch_size': 'Lote',
        'training_time': 'Tiempo (s)',
        'accuracy': 'Accuracy',
        'loss': 'Loss',
        'precision': 'Precisión',
        'recall': 'Recall',
        'params': '# Parámetros',
        'interpolation_method': 'Interpolación',
        'source': 'Interpolación'        
    }
    
    table_df = table_df.rename(columns=column_names)
    
    # Generate LaTeX table using pandas to_latex
    latex_table = table_df.to_latex(
        index=False,
        escape=False,
        column_format='|' + 'c|' * len(table_df.columns),
        caption=f'Top {top_n} modelos por precisión',
        label='tab:top_models',
        position='htbp'
    )
    
    # Print to console
    print("=== TOP {} MODELS - LATEX TABLE ===".format(top_n))
    print(latex_table)
    
    # Save to file if output_path is provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_table)
        print(f"\nLaTeX table saved to: {output_path}")
    
    return latex_table

if __name__ == "__main__":
    # Configuration
    csv_file = "model_comparison.csv"  # Adjust path as needed
    output_file = "top_models_table.tex"  # Output LaTeX file
    top_n = 30
    csv_path = ROOT_DIR / "Analysis" / csv_file
    output_path = ROOT_DIR / "Analysis" / output_file
    # Check if file exists
    if not Path(csv_path).exists():
        print(f"Error: File {csv_path} not found!")
        print("Available files in Analysis folder:")
        analysis_dir = Path("Analysis")
        if analysis_dir.exists():
            for file in analysis_dir.glob("*.csv"):
                print(f"  - {file.name}")
        exit(1)
    
    # Generate LaTeX table
    try:
        latex_code = csv_to_latex_table(str(csv_path), str(output_path), top_n)
        print("\n" + "="*50)
        print("INSTRUCTIONS FOR USE:")
        print("1. Copy the LaTeX code above")
        print("2. Include these packages in your LaTeX document:")
        print("   \\usepackage{booktabs}")
        print("   \\usepackage{array}")
        print("3. Paste the table code in your document")
        print("4. Compile with pdflatex")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
