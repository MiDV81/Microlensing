# 61_SimulationApp.py – FULL single‑file implementation
# ----------------------------------------------------------------------
# • Everything lives in this file: GUI, workflows hooks, and the embedded
#   FileStorage dataclass (no external *filestorage.py* needed).
# • Both the *Simulations* and *Model* panels have their own ⚙ buttons
#   again, plus the global "⚙ Working dir" button in the header.
# • Nothing from the original file has been dropped – the helper dialogs,
#   preview/plot utilities, and entry‑point code are all here.
#
# USE:  python 61_SimulationApp.py
# ----------------------------------------------------------------------
from __future__ import annotations

import copy
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from dataclasses import dataclass, field, fields
from typing import Dict, Any, Iterable
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# Heavy, optional imports are delayed to runtime (see workflows calls)
from Functions.NNWorkflows import (
    SingleLensesSimulations,
    BinaryLensesSimulations,
    NoiseSimulations,
    LightCurvesCombinator,
    ModelBuilder,
    ModelChecker,
)
from Functions.NNFunctions import load_data, plot_random_events_from_df, ROOT_DIR

# ════════════════════════════════════════════════════════════════════
# 0. Shared FileStorage (single source‑of‑truth for every filename)
# ════════════════════════════════════════════════════════════════════

_DEFAULT_BASE = ROOT_DIR / "MicrolensingData"

@dataclass
class FileStorage:
    """Centralised holder for *all* filenames + base directory."""

    # base directory (user‑editable via the ⚙ Working dir dialog)
    base_dir: Path = field(default_factory=lambda: _DEFAULT_BASE)

    # simulation artefacts
    stats_file: str = "filtered_params_stats.json"
    single_lens_file: str = "singlelense_lightcurves.pkl"
    binary_lens_file: str = "binarylense_lightcurves.pkl"
    noise_file: str = "noise_lightcurves.pkl"
    combined_file: str = "all_lightcurves.pkl"

    # real data + model artefacts
    ogle_lightcurves: str = "ogle_lightcurves.pkl"
    ogle_events: str = "Confirmed_OGLEEvents.txt"
    model_file: str = "event_classifier.keras"
    predictions_file: str = "event_predictions.csv"

    # ── helpers ──────────────────────────────────────────────
    def path(self, attr: str) -> Path:
        """Return *base_dir / getattr(self, attr)* as a Path."""
        return self.base_dir / getattr(self, attr)

    @property
    def file_attrs(self) -> Iterable[str]:
        return (f.name for f in fields(self) if f.name != "base_dir")

    def reset_defaults(self):
        defaults = FileStorage()
        for f in fields(self):
            setattr(self, f.name, getattr(defaults, f.name))

# ════════════════════════════════════════════════════════════════════
# 1. Tiny utility classes (InterfaceController & StatusBar)
# ════════════════════════════════════════════════════════════════════

class InterfaceController:
    """Enable/disable arbitrary Tk widgets as a group."""

    def __init__(self):
        self.widgets: list[tk.Widget] = []

    def add_widgets(self, widgets):
        self.widgets.extend(widgets)

    def disable(self):
        for w in self.widgets:
            try:
                w.configure(state="disabled")
            except tk.TclError:
                pass  # Widget doesn't support state parameter

    def enable(self):
        for w in self.widgets:
            try:
                w.configure(state="normal")
            except tk.TclError:
                pass  # Widget doesn't support state parameter

class StatusBar:
    def __init__(self, parent):
        self.frame = ttk.Frame(parent)
        self._msg = tk.StringVar(value="Ready")
        ttk.Label(self.frame, textvariable=self._msg).pack(side="left", padx=5)

    def set_status(self, msg: str):
        self._msg.set(msg)

# ════════════════════════════════════════════════════════════════════
# 2. Settings dialogs
# ════════════════════════════════════════════════════════════════════

_INTERP_METHODS = ["savgol", "linear", "bin", "gp", "spline"]

class SimulationSettingsDialog:
    """Tweaks #samples/interp‑method + per‑simulation filenames."""

    def __init__(self, parent, current_len: int, current_interp: str, fs: FileStorage):
        self.fs = fs
        self.result: Dict[str, Any] | None = None

        dlg = self.dlg = tk.Toplevel(parent)
        dlg.title("Simulation settings")
        dlg.grab_set()

        main = ttk.Frame(dlg, padding=10)
        main.pack(fill="both", expand=True)

        # sequence length
        row = ttk.Frame(main); row.pack(fill="x", pady=2)
        ttk.Label(row, text="Sequence length:").pack(side="left")
        self.seq_len = tk.StringVar(value=str(current_len))
        ttk.Entry(row, textvariable=self.seq_len, width=8).pack(side="right")

        # interpolation
        row = ttk.Frame(main); row.pack(fill="x", pady=2)
        ttk.Label(row, text="Interpolation:").pack(side="left")
        self.interp = tk.StringVar(value=current_interp)
        ttk.Combobox(row, textvariable=self.interp, values=_INTERP_METHODS, state="readonly").pack(side="right")

        # filenames
        files_frame = ttk.LabelFrame(main, text="File names")
        files_frame.pack(fill="x", pady=6)
        self.vars: dict[str, tk.StringVar] = {}
        for attr in [
            "stats_file",
            "single_lens_file",
            "binary_lens_file",
            "noise_file",
            "combined_file",
        ]:
            r = ttk.Frame(files_frame); r.pack(fill="x", pady=1)
            ttk.Label(r, text=f"{attr}:", width=18, anchor="w").pack(side="left")
            v = tk.StringVar(value=getattr(fs, attr))
            ttk.Entry(r, textvariable=v, width=34).pack(side="right")
            self.vars[attr] = v

        # buttons
        btns = ttk.Frame(main); btns.pack(pady=6)
        ttk.Button(btns, text="OK",     command=self._ok).pack(side="left", padx=4)
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side="left")
        ttk.Button(btns, text="Default", command=self._reset_defaults).pack(side="left", padx=4)
        self._center()

    def _reset_defaults(self):
        """Return every field to the factory settings."""
        self.seq_len.set("1000")                            # hard-coded default
        self.interp.set(_INTERP_METHODS[0])                  # first option (“savgol”)
        fresh = FileStorage()                                # new empty instance
        for attr, var in self.vars.items():                  # restore file names
            var.set(getattr(fresh, attr))

    def _center(self):
        self.dlg.update_idletasks()
        w, h = self.dlg.winfo_width(), self.dlg.winfo_height()
        x = (self.dlg.winfo_screenwidth() - w) // 2
        y = (self.dlg.winfo_screenheight() - h) // 2
        self.dlg.geometry(f"{w}x{h}+{x}+{y}")

    def _ok(self):
        try:
            seq = int(self.seq_len.get())
        except ValueError:
            messagebox.showerror("Input error", "Sequence length must be an integer")
            return
        self.result = {
            "sequence_length": seq,
            "interpolation": self.interp.get(),
            "filenames": {k: v.get() for k, v in self.vars.items()},
        }
        self.dlg.destroy()

class ModelConfigDialog:
    """Edit model-training hyper-parameters (batch, epochs, …)."""

    def __init__(self, parent, cfg: Dict[str, Any], defaults: Dict[str, Any] | None = None):
        self.orig      = cfg
        self.defaults  = copy.deepcopy(defaults or cfg)   # keep pristine copy
        self.result: Dict[str, Any] | None = None

        dlg = self.dlg = tk.Toplevel(parent)
        dlg.title("Model hyper-parameters")
        dlg.grab_set()

        main = ttk.Frame(dlg, padding=10)
        main.pack(fill="both", expand=True)

        # keep references so we can enable / disable the seed field
        self.vars: dict[str, tk.Variable] = {}
        self._random_seed_entry: ttk.Entry | None = None

            # Group parameters
        parameter_groups = {
            "Data Processing": [
                "sequence_length", "interpolation", "test_fraction", 
                "validation_fraction", "use_seed", "random_seed"
            ],
            "Model Architecture": [
                "n_layers", "kernel_sizes", "filters", "pool_sizes",
                "dense_units", "dropout_rate"
            ],
            "Training Parameters": [
                "batch_size", "epochs", "learning_rate"
            ]
        }

        for group_name, params in parameter_groups.items():
            group_frame = ttk.LabelFrame(main, text=group_name, padding=5)
            group_frame.pack(fill="x", pady=5)
            
            for k in params:
                v = cfg[k]
                row = ttk.Frame(group_frame)
                row.pack(fill="x", pady=2)
                
                ttk.Label(row, text=f"{k}:", width=18, anchor="w").pack(side="left")
                
                # Special handling for list parameters
                if isinstance(v, list):
                    var = tk.StringVar(value=str(v)[1:-1])  # Remove brackets
                    entry = ttk.Entry(row, textvariable=var, width=20)
                    ttk.Label(row, text="(csv)").pack(side="right", padx=2)
                
                # Special widgets for specific parameters
                elif k == "use_seed":
                    var = tk.BooleanVar(value=v)
                    entry = ttk.Checkbutton(row, variable=var, command=self._toggle_seed)
                
                elif k == "interpolation":
                    var = tk.StringVar(value=str(v))
                    entry = ttk.Combobox(
                        row,
                        textvariable=var,
                        values=_INTERP_METHODS,
                        state="readonly",
                        width=10
                    )
                
                else:
                    var = tk.StringVar(value=str(v))
                    entry = ttk.Entry(row, textvariable=var, width=10)
                
                entry.pack(side="right")
                self.vars[k] = var

        # initial enable / disable for the random-seed field
        self._toggle_seed()

        btns = ttk.Frame(main); btns.pack(pady=6)
        ttk.Button(btns, text="OK",     command=self._ok).pack(side="left", padx=4)
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side="left")
        ttk.Button(btns, text="Default", command=self._default).pack(side="left", padx=4)
        self._center()

    def _default(self):
        """Fill widgets with the factory values."""
        for key, var in self.vars.items():
            value = self.defaults[key]
            if isinstance(value, list):
                # Convert list to comma-separated string without brackets
                var.set(", ".join(str(x) for x in value))
            else:
                var.set(str(value))

    def _toggle_seed(self):
        """Enable or disable the random-seed entry based on the checkbox."""
        if self._random_seed_entry is None:
            return
        state = "normal" if self.vars["use_seed"].get() else "disabled"
        self._random_seed_entry.configure(state=state)

    def _center(self):
        self.dlg.update_idletasks()
        w, h = self.dlg.winfo_width(), self.dlg.winfo_height()
        x = (self.dlg.winfo_screenwidth() - w) // 2
        y = (self.dlg.winfo_screenheight() - h) // 2
        self.dlg.geometry(f"{w}x{h}+{x}+{y}")

    def _ok(self):
        out: Dict[str, Any] = {}
        try:
            for k, var in self.vars.items():
                ref = self.orig[k]
                if isinstance(ref, list):
                    # Convert comma-separated string to list of appropriate type
                    values = [x.strip() for x in var.get().split(',')]
                    out[k] = [type(ref[0])(x) for x in values]
                elif isinstance(ref, bool):
                    out[k] = var.get()
                else:
                    out[k] = type(ref)(var.get())
                
                # Validate n_layers matches list lengths
                if k == 'n_layers':
                    n_layers = out[k]
                    for param in ['kernel_sizes', 'filters', 'pool_sizes']:
                        if param in out and len(out[param]) != n_layers:
                            raise ValueError(f"{param} must have exactly {n_layers} values")
            
            # Print configuration
            print("\nModel Configuration:")
            print("-" * 50)
            for k, v in out.items():
                print(f"{k:20}: {v}")
            print("-" * 50)
            
            self.result = out
            self.dlg.destroy()
                    
        except ValueError as e:
            messagebox.showerror("Input error", str(e))
            return

class ModelFilesDialog:
    """Edit filenames used by the model panel."""

    def __init__(self, parent, fs: FileStorage):
        self.fs = fs
        self.result = None
        dlg = self.dlg = tk.Toplevel(parent)
        dlg.title("Model Files Configuration")
        dlg.grab_set()
        
        main = ttk.Frame(dlg, padding=10)
        main.pack(fill="both", expand=True)

        # Files in logical order with descriptions
        self.vars: dict[str, tk.StringVar] = {}
        files_config = [
            ("model_file", "Model File:", "event_classifier.keras"),
            ("combined_file", "Data for Model:", "all_lightcurves.pkl"),
            ("ogle_lightcurves", "Data to Predict:", "ogle_lightcurves.pkl"),
            ("ogle_events", "Events Known:", "Confirmed_OGLEEvents.txt"),
            ("predictions_file", "Predictions:", "event_predictions.csv")
        ]

        for attr, label, default in files_config:
            r = ttk.Frame(main)
            r.pack(fill="x", pady=4)  # Increased padding
            ttk.Label(r, text=label, width=15, anchor="w").pack(side="left")
            v = tk.StringVar(value=getattr(fs, attr))
            ttk.Entry(r, textvariable=v, width=35).pack(side="right")
            self.vars[attr] = v

        # Buttons
        btns = ttk.Frame(main)
        btns.pack(pady=10)  # Increased padding
        ttk.Button(btns, text="OK", command=self._ok).pack(side="left", padx=4)
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side="left")
        ttk.Button(btns, text="Reset to Defaults", command=self._reset_defaults).pack(side="left", padx=4)
        
        self._center()

    def _reset_defaults(self):
        """Reset all files to default values."""
        defaults = {
            "model_file": "event_classifier.keras",
            "combined_file": "all_lightcurves.pkl",
            "ogle_lightcurves": "ogle_lightcurves.pkl",
            "ogle_events": "Confirmed_OGLEEvents.txt",
            "predictions_file": "event_predictions.csv"
        }
        for attr, var in self.vars.items():
            var.set(defaults[attr])

    def _center(self):
        self.dlg.update_idletasks()
        w, h = self.dlg.winfo_width(), self.dlg.winfo_height()
        x = (self.dlg.winfo_screenwidth() - w) // 2
        y = (self.dlg.winfo_screenheight() - h) // 2
        self.dlg.geometry(f"{w}x{h}+{x}+{y}")

    def _ok(self):
        self.result = {k: v.get() for k, v in self.vars.items()}; self.dlg.destroy()

class WorkingDirDialog:
    """Pick a new *base_dir* for FileStorage."""

    def __init__(self, parent, fs: FileStorage):
        self.fs = fs
        dlg = self.dlg = tk.Toplevel(parent)
        dlg.title("Working directory"); dlg.grab_set()
        ttk.Label(dlg, text="Microlensing data directory:").pack(side="left", padx=5, pady=10)
        self.var = tk.StringVar(value=str(fs.base_dir))
        ttk.Entry(dlg, textvariable=self.var, width=48).pack(side="left", padx=2)
        ttk.Button(dlg, text="📁", command=self._browse, width=3).pack(side="left")
        ttk.Button(dlg, text="OK", command=self._ok).pack(side="left", padx=4)
        ttk.Button(dlg, text="Cancel", command=dlg.destroy).pack(side="left")
        self._center()

    def _browse(self):
        from tkinter import filedialog
        folder = filedialog.askdirectory(initialdir=self.var.get())
        if folder:
            self.var.set(folder)
    def _ok(self):
        self.fs.base_dir = Path(self.var.get()); self.dlg.destroy()
    def _center(self):
        self.dlg.update_idletasks(); w,h = self.dlg.winfo_width(), self.dlg.winfo_height()
        x = (self.dlg.winfo_screenwidth()-w)//2; y = (self.dlg.winfo_screenheight()-h)//2
        self.dlg.geometry(f"{w}x{h}+{x}+{y}")

# ════════════════════════════════════════════════════════════════════
# 3. Simulation panel
# ════════════════════════════════════════════════════════════════════

class SimulationPanel:
    """Run synthetic‑curve workflows and preview results."""

    def __init__(self, parent, status_cb, ic: InterfaceController, fs: FileStorage):
        self.fs = fs; self.status_cb = status_cb; self.ic = ic
        self.frame = ttk.LabelFrame(parent, text="Simulations", padding=10)
        self.widgets: list[tk.Widget] = []
        self.n_samples = tk.StringVar(value="1000")
        self.seq_len = 1000; self.interp = "savgol"
        self._build()

    # ─── UI helpers ─────────────────────────────────────────────
    def _build(self):
        r = ttk.Frame(self.frame); r.pack(fill="x", pady=4)
        ttk.Label(r, text="Samples per run:").pack(side="left")
        entry = ttk.Entry(r, textvariable=self.n_samples, width=8); entry.pack(side="right"); self.widgets.append(entry)

        for lbl, fn, attr in [
            ("Single‑lens", self.run_single, "single_lens_file"),
            ("Binary‑lens", self.run_binary, "binary_lens_file"),
            ("Noise", self.run_noise, "noise_file"),
        ]:
            self._row(lbl, fn, attr)

        ttk.Separator(self.frame, orient="horizontal").pack(fill="x", pady=6)
        bottom = ttk.Frame(self.frame); bottom.pack(fill="x")
        b = ttk.Button(bottom, text="Combine all curves", command=self.combine_curves); b.pack(side="left"); self.widgets.append(b)
        ttk.Button(bottom, text="⚙", width=3, command=self._edit_cfg).pack(side="right")


    def _row(self, text, run_cmd, attr):
        r = ttk.Frame(self.frame); r.pack(fill="x", pady=1)
        run = ttk.Button(r, text=text, command=run_cmd); run.pack(side="left", padx=1)
        show = ttk.Button(r, text="Show", command=lambda a=attr: self.show_df(a)); show.pack(side="left", padx=1)
        plot = ttk.Button(r, text="Plot", command=lambda a=attr: self.plot_random(a)); plot.pack(side="left", padx=1)
        self.widgets += [run, show, plot]

    # ─── config dialog ─────────────────────────────────────────
    def _edit_cfg(self):
        dlg = SimulationSettingsDialog(self.frame, self.seq_len, self.interp, self.fs)
        self.frame.wait_window(dlg.dlg)
        if dlg.result:
            self.seq_len = dlg.result["sequence_length"]
            self.interp = dlg.result["interpolation"]
            for k, v in dlg.result["filenames"].items():
                setattr(self.fs, k, v)
            self.status_cb("Simulation settings updated")

    # ─── runners ----------------------------------------------------

    def _wrap(self, label, fn, *args):
        self.ic.disable(); self.status_cb(f"Running {label}…")
        try:
            fn(*args, DIR=self.fs.base_dir)
            self.status_cb(f"✓ {label} finished")
        except Exception as e:
            messagebox.showerror("Simulation failed", str(e)); self.status_cb(f"✗ {label} failed")
        finally:
            self.ic.enable()

    def run_single(self):
        n = self._get_samples()
        if n:
            self._wrap(
                "Single-lens simulation",
                SingleLensesSimulations,
                n,
                self.fs.stats_file,          
                self.fs.single_lens_file         
            )

    def run_binary(self):
        n = self._get_samples()
        if n:
            self._wrap(
                "Binary-lens simulation",
                BinaryLensesSimulations,
                n,
                self.fs.stats_file,          
                self.fs.binary_lens_file    
            )

    def run_noise(self):
        n = self._get_samples()
        if n:
            self._wrap(
                "Noise simulation",
                NoiseSimulations,
                n,
                self.fs.noise_file          
            )

    def combine_curves(self):
        self._wrap(
            "Curve combination",
            LightCurvesCombinator,              
            self.fs.single_lens_file,       
            self.fs.binary_lens_file,       
            self.fs.noise_file,             
            self.fs.combined_file         
        )

    # ─── data preview / plotting -----------------------------------
    def show_df(self, attr):
        try:
            file_path = self.fs.path(attr)
            df = load_data(getattr(self.fs, attr), self.fs.base_dir)
            print(f"\nPreview of {attr}")
            print(f"Path: {file_path}")
            print(f"Rows: {len(df)}\n")
            print(df.head())
            self.status_cb("Data preview sent to console")
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def plot_random(self, attr):
        try:
            file_path = self.fs.path(attr)
            df = load_data(getattr(self.fs, attr), self.fs.base_dir)
            print(f"\nPlotting random events from:")
            print(f"Path: {file_path}")
            plot_random_events_from_df(
                df, 
                n_events=5, 
                sequence_length=self.seq_len, 
                interpolation_method=self.interp,
                title_prefix=f"{attr}"
            )
            self.status_cb("Random events plotted")
        except Exception as e:
            messagebox.showerror("Plot error", str(e))

    def get_widgets(self): return self.widgets

    def _get_samples(self):
        """Return the integer in the ‘#samples’ entry or None on error."""
        try:
            return int(self.n_samples.get().strip())
        except (ValueError, AttributeError) as e:
            print(e)
            messagebox.showerror("Input error", "Number of samples must be an integer")
            return None

# ════════════════════════════════════════════════════════════════════
# 4. Model panel
# ════════════════════════════════════════════════════════════════════

class ModelPanel:
    def __init__(self, parent, status_cb, ic: InterfaceController, fs: FileStorage):
        self.fs = fs; self.status_cb = status_cb; self.ic = ic
        self.frame = ttk.LabelFrame(parent, text="Model", padding=10)
        self.widgets: list[tk.Widget] = []

        self.cfg: Dict[str, Any] = {
            # Data processing parameters
            "sequence_length": 1000,
            "interpolation": "linear",
            "test_fraction": 0.2,
            "validation_fraction": 0.2,
            "use_seed": True,
            "random_seed": 42,
            
            # Model architecture
            "n_layers": 2,
            "kernel_sizes": [5, 3],
            "filters": [32, 64],
            "pool_sizes": [2, 2],
            "dense_units": 64,
            "dropout_rate": 0.3,
            
            # Training parameters
            "batch_size": 32,
            "epochs": 30,
            "learning_rate": 1e-3
        }
        self._cfg_default = copy.deepcopy(self.cfg)

        self.history = None; self.model = None
        self._build()

    def _build(self):
        for lbl, cmd in [
            ("Load model", self.load_model),
            ("Train", self.train_model),
            ("Plot history", self.plot_history),
            ("Check", self.check_model),
            ("Predict", self.use_model),            
        ]:
            b = ttk.Button(self.frame, text=lbl, command=cmd); b.pack(fill="x", pady=2); self.widgets.append(b)
        
        btn_row = ttk.Frame(self.frame); btn_row.pack(pady=4)
        cfg_btn   = ttk.Button(btn_row, text="⚙", width=3, command=self._edit_cfg)
        file_btn  = ttk.Button(btn_row, text="📁", width=3, command=self._edit_files)
        reset_btn = ttk.Button(btn_row, text="↺", width=3, command=self._reset_defaults)
        for b in (cfg_btn, file_btn, reset_btn):
           b.pack(side="left", padx=2)
           self.widgets.append(b)

    # ─── dialogs ----------------------------------------------------
    def _edit_cfg(self):
        dlg = ModelConfigDialog(self.frame, self.cfg, self._cfg_default)
        self.frame.wait_window(dlg.dlg)
        if dlg.result:
            self.cfg.update(dlg.result)
            self.status_cb("Model configuration updated")

    def _edit_files(self):
        dlg = ModelFilesDialog(self.frame, self.fs); self.frame.wait_window(dlg.dlg)
        if dlg.result:
            for k, v in dlg.result.items(): setattr(self.fs, k, v)
            self.status_cb("Model filenames updated")

    # ─── callbacks --------------------------------------------------
    def load_model(self):
        self.ic.disable(); self.status_cb("Loading model…")
        try:
            from Functions.NNFunctions import model_loader
            self.model = model_loader(self.fs.model_file, self.fs.base_dir)
            self.status_cb("✓ Model loaded")
        except Exception as e:
            messagebox.showerror("Load error", str(e)); self.status_cb("✗ Load failed")
        finally: self.ic.enable()

    def train_model(self):
        self.ic.disable()
        data_path = self.fs.path('combined_file')
        self.status_cb(f"Training model using data from:\n{data_path}")
        print(f"\nLoading training data from:\n{data_path}")
        
        try:
            self.history, self.model = ModelBuilder(
                self.cfg,
                load_filename=self.fs.combined_file,
                model_filename=self.fs.model_file,
                DIR=self.fs.base_dir
            )
            self.status_cb("✓ Training complete")
        except Exception as e:
            messagebox.showerror("Training error", str(e))
            self.status_cb("✗ Training failed")
        finally:
            self.ic.enable()

    def check_model(self):
        self.ic.disable(); self.status_cb("Checking…")
        try:
            print(f'{self.cfg["sequence_length"]=}')
            ModelChecker(sequence_length=self.cfg["sequence_length"], interpolation_method=self.cfg["interpolation"],
                         model_filename=self.fs.model_file, ogle_filename=self.fs.ogle_lightcurves,
                         ogle_events_filename=self.fs.ogle_events, DIR=self.fs.base_dir)
            self.status_cb("✓ Check complete")
        except Exception as e:
            messagebox.showerror("Check error", str(e)); self.status_cb("✗ Check failed")
        finally: self.ic.enable()

    def use_model(self):
        self.ic.disable(); self.status_cb("Predicting…")
        try:
            from Functions.NNWorkflows import ModelUser
            ModelUser(sequence_length=self.cfg["sequence_length"], interpolation_method=self.cfg["interpolation"],
                      model_filename=self.fs.model_file, data_filename=self.fs.ogle_lightcurves,
                      csv_out_filename=self.fs.predictions_file, DIR=self.fs.base_dir)
            self.status_cb(f"✓ Predictions saved → {self.fs.path('predictions_file')}")
        except Exception as e:
            messagebox.showerror("Prediction error", str(e)); self.status_cb("✗ Prediction failed")
        finally: self.ic.enable()

    def plot_history(self):
        if not self.history:
            messagebox.showinfo("Info", "Train the model first"); return
        from Functions.NNFunctions import plot_training_history; plot_training_history(self.history)

    def get_widgets(self): return self.widgets

    def _reset_defaults(self):
        """Restore model hyper-params *and* filenames to factory settings."""
        self.cfg = copy.deepcopy(self._cfg_default)
        self.fs.reset_defaults()
        self.status_cb("Model configuration and filenames reset")

# ════════════════════════════════════════════════════════════════════
# 5. Main application shell
# ════════════════════════════════════════════════════════════════════

class SimulationApp:
    def __init__(self, root: tk.Tk):
        self.root = root; root.title("Neural‑network microlensing playground")
        self.fs = FileStorage()

        # header
        hdr = ttk.Frame(root); hdr.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        ttk.Label(hdr, text="Neural Network for Microlensing", font=("Helvetica", 16, "bold")).pack(side="left")
        ttk.Button(hdr, text="⚙ Working dir", command=self._edit_dir).pack(side="right")

        # status + interface controller
        self.ic = InterfaceController(); self.status = StatusBar(root); self.status.frame.grid(row=2, column=0, columnspan=2, sticky="ew")

        # panels
        self.sim_panel = SimulationPanel(root, self.status.set_status, self.ic, self.fs); self.sim_panel.frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.mod_panel = ModelPanel(root, self.status.set_status, self.ic, self.fs); self.mod_panel.frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

        self.ic.add_widgets(self.sim_panel.get_widgets()); self.ic.add_widgets(self.mod_panel.get_widgets())

        # responsive resizing
        root.columnconfigure(0, weight=1); root.columnconfigure(1, weight=1); root.rowconfigure(1, weight=1)

    def _edit_dir(self):
        WorkingDirDialog(self.root, self.fs); self.status.set_status(f"Working dir → {self.fs.base_dir}")

# entry‑point ---------------------------------------------------------

def main():
    root = tk.Tk(); SimulationApp(root); root.mainloop()

if __name__ == "__main__":
    main()
