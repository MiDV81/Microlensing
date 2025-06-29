from tensorflow.keras.utils import plot_model
from pathlib import Path
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Functions.NNFunctions import model_loader, ROOT_DIR

def plot_model_schematic(model_file: str = "event_classifier.keras", 
                        save_dir: Path = ROOT_DIR / "MicrolensingData"):
    """Create and save visual schematic of the neural network architecture."""
       
    try:
        # Load existing model
        print(f"Loading model from: {save_dir / model_file}")
        model = model_loader(model_file, save_dir)
        
        # Create plots directory if it doesn't exist
        plots_dir = save_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Save schematic in different formats
        detailed_path = plots_dir / "model_schematic_detailed.png"
        simple_path = plots_dir / "model_schematic_simple.png"

        print("Generating detailed schematic...")
        plot_model(
            model,
            to_file=str(detailed_path),  # Convert Path to string
            show_shapes=True,
            show_layer_names=True,
            show_layer_activations=True,
            expand_nested=True,
            dpi=200
        )
        
        print("Generating simplified schematic...")
        plot_model(
            model,
            to_file=str(simple_path),  # Convert Path to string
            show_shapes=False,
            show_layer_names=True,
            show_layer_activations=False,
            expand_nested=False,
            dpi=200
        )
        
        print(f"\nModel schematics saved to:\n{plots_dir}")
        print("\nModel Summary:")
        model.summary()
        
        # Open both schematics
        if detailed_path.exists() and simple_path.exists():
            os.startfile(str(detailed_path))
            os.startfile(str(simple_path))
        else:
            print("Error: Schematics were not generated")
            
    except Exception as e:
        print(f"Error generating model schematic: {e}")
        sys.exit(1)

if __name__ == "__main__":
    plot_model_schematic()