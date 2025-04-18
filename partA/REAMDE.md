```
# CNN on iNaturalist‑12K

This repository contains code to train and evaluate a configurable convolutional neural network (CNN) on the iNaturalist‑12K dataset using PyTorch Lightning and Weights & Biases (wandb). You can run a hyperparameter sweep to find the best configuration, then train and evaluate the final model.

---

## Repository Structure

```
.
├── model.py                # Defines ConfigurableCNN LightningModule
├── Question-2.py                # Sweep script: data loading, sweep setup, `train_model()`
├── Question-4.py           # Best‑run script: load best wandb run, train, save, evaluate
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Prerequisites

- Python 3.8+  
- CUDA‑enabled GPU (optional but recommended)  
- Git LFS (if your dataset is large)  

---

## Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/<your‑username>/cnn‑inaturalist12k.git
   cd cnn‑inaturalist12k
   ```

2. **Create a virtual environment**  
   ```bash
   python -m venv venv
   source venv/bin/activate       # on Windows: venv\Scripts\activate
   ```

3. **Install dependencies**  
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Authenticate with Weights & Biases**  
   ```bash
   wandb login
   ```

---

## Data Preparation

1. Download the **iNaturalist‑12K** archive (e.g. `nature_12K.zip`) from your source.
2. Unzip into a folder with the structure:
   ```
   data/
     inaturalist_12K/
       train/
         class_0/
         class_1/
         … 
       test/
         class_0/
         class_1/
         …
   ```
3. Edit the `base_dir` variable in both `Question-2.py` and `Question-4.py` to point to `data/inaturalist_12K`.

---

## Usage

### 1. Hyperparameter Sweep

Run a Bayesian sweep over activation functions, filter sizes, dropout rates, etc.:

```bash
python Question-2.py
```

- This will:
  - Log experiments to wandb under project **CNN_inaturalist_12K**  
  - Use `WandbLogger` to track `val_loss` and `val_acc`  
  - Save the sweep ID — wandb will spin up 5 agents by default

### 2. Train & Evaluate Best Model

After the sweep completes, run:

```bash
python Question-4.py
```

What this script does:

1. **Fetch the best run** from `mrsagarbiswas-iit-madras/CNN_inaturalist_12K` by lowest `val_loss` via the wandb API.  
2. **Rebuild** the `ConfigurableCNN` with that configuration.  
3. **Train** for `max_epochs` (from the best config) using PyTorch Lightning.  
4. **Save** the trained weights to `model_state_dict.pth`.  
5. **Load** weights and **evaluate** on the test set, printing:
   ```
   Test Loss: 1.7517    Test Accuracy: 39.50%
   ```
6. **Visualizations** (logged to wandb and shown locally):
   - Random grid of predictions (actual vs. predicted)
   - First‐layer convolutional filters
   - Guided backprop maps for 10 random CONV5 neurons

---

## Code Details

### `model.py`  
Defines `ConfigurableCNN(pl.LightningModule)`:

- **Configurable blocks**  
  - `num_conv_layers`: number of conv blocks (default 5)  
  - `filters`: list of channel sizes per block  
  - `kernel_sizes`: list of kernel sizes  
  - `activation`: one of `ReLU`, `GELU`, `SiLU`, `Mish`  
  - `use_batchnorm`: apply `BatchNorm2d` after each conv  
  - `dropout_rate`: spatial dropout in conv blocks  
- **Classifier head**  
  - Flatten → Dense(`dense_neurons`) → Activation → Dropout(`dropout_rate_dense`) → Dense(`num_classes=10`)  
- **Training & validation** steps log `train_loss`, `train_acc`, `val_loss`, `val_acc`  
- Uses **Adam** optimizer with configurable `lr`

### `Question-2.py`  
- Builds **lazy‐loading** datasets (paths only, load images on demand)
- Defines `train_model()` which:
  - Initializes a `WandbLogger` with a `default_config`
  - Constructs `ConfigurableCNN` from merged sweep config
  - Runs `pl.Trainer.fit(...)`
- Sets up a **Bayesian sweep** (`wandb.sweep(...)`) over:
  - `activation`, `filters`, `use_batchnorm`, `dropout_rate`, `dense_neurons`, `lr`, `max_epochs`, `batch_size`

### `Question-4.py`  
- Installs dependencies to Colab (optional)  
- **Mounts Google Drive** (if using Colab)  
- Fetches best run via `wandb.Api()`  
- Splits train/val with `StratifiedShuffleSplit`  
- Defines `LazyLoadDataset`, DataLoaders, and Lightning `Trainer`  
- **Training** with mixed‑precision (`precision=16`) if GPU available  
- **Evaluation** via a custom `evaluate_model()`  
- **Visualizations** using Matplotlib and wandb (predictions grid, filters, guided backprop)

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

**Note:**  
- Update dataset paths (`base_dir`) before running.  
- Ensure `wandb.login()` has been called once on your machine or Colab.  
- Adjust `num_workers`, `batch_size`, or `max_epochs` in the configs to suit your hardware.