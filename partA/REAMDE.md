
# CNN on iNaturalist‑12K

This repository contains code to train and evaluate a configurable convolutional neural network (CNN) on the iNaturalist‑12K dataset using PyTorch Lightning and Weights & Biases (wandb).

----------

## Repository Structure

```plaintext
.
├── model.py                # Defines ConfigurableCNN LightningModule
├── Question-2.py           # Sweep script: data loading, sweep setup, train_model()
├── Question-4.py           # Best‑run script: load best wandb run, train, save, evaluate
├── requirements.txt        # Python dependencies
└── README.md               # This file

```

----------

## Prerequisites

-   Python 3.8+
    
-   CUDA‑enabled GPU (optional but recommended)
    
-   Git LFS (if your dataset is large)
    

----------

## Installation

1.  **Clone the repo**
    
    ```bash
    git clone https://github.com/MrSagarBiswas/da6401_assignment2/
    cd da6401_assignment2/partA
    
    ```
    
2.  **Create a virtual environment**
    
    ```bash
    python -m venv venv
    source venv/bin/activate       # on Windows: venv\Scripts\activate
    
    ```
    
3.  **Install dependencies**
    
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    
    ```
    
4.  **Authenticate with Weights & Biases**
    
    ```bash
    wandb login
    
    ```
    

----------

## Data Preparation

1.  Download the **[iNaturalist‑12K](https://storage.googleapis.com/wandb_datasets/nature_12K.zip)** archive (e.g. `nature_12K.zip`) and unzip into:
    
    ```plaintext
    content/drive/
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
    
2.  Edit the `base_dir` variable in both `Question-2.py` and `Question-4.py` to point to `data/inaturalist_12K`.
    

----------

## Usage

### 1. Hyperparameter Sweep

```bash
python Question-2.py

```

This runs a Bayesian sweep over activation functions, filter sizes, dropout rates, etc., logging to your wandb project.

### 2. Train & Evaluate Best Model

```bash
python Question-4.py

```

Fetches the best sweep run by validation loss, retrains, saves weights, and evaluates on the test set:

```
Test Loss: 1.7517    Test Accuracy: 39.50%

```

----------

## Code Details

### `model.py`

-   **ConfigurableCNN** extends `pl.LightningModule`:
    
    -   Conv blocks with customizable layers, filters, activations, batchnorm, dropout.
        
    -   Dense head with two linear layers and dropout.
        
    -   Logs `train_loss`, `train_acc`, `val_loss`, `val_acc`.
        
    -   Uses Adam optimizer with learning rate `lr` from config.
        

### `Question-2.py`

-   Lazy-loading dataset using file paths only.
    
-   Defines `train_model()` with `WandbLogger`.
    
-   Sets up a Bayesian sweep over:
    
    -   `activation`, `filters`, `use_batchnorm`, `dropout_rate`, `dense_neurons`, `lr`, `max_epochs`, `batch_size`.
        

### `Question-4.py`

-   Retrieves best run from wandb API.
    
-   Splits training data with `StratifiedShuffleSplit`.
    
-   Trains Lightning `Trainer` with mixed precision (if GPU).
    
-   Evaluates on the test set.
    
-   Visualizes sample predictions, filters, and guided backprop maps.
    

----------

## License

MIT License. See [LICENSE](https://opensource.org/license/mit) for details.