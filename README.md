# DA6401 Assignment 2: CNN on iNaturalist-12K

This repository contains two parts of Assignment 2 for DA6401, focusing on convolutional neural networks (CNNs) applied to the iNaturalist‑12K dataset.

- **Part A: Configurable CNN & Hyperparameter Sweeps**
- **Part B: Fine‑Tuning a Pre‑trained ResNet50**

A combined WandB report and all source code are available below.

---

## Repository Structure

```plaintext
da6401_assignment2/
├── partA/                # Configurable CNN hyperparameter sweep
│   ├── model.py          # Defines ConfigurableCNN LightningModule
│   ├── Question-2.py     # Bayesian sweep script (data loading, sweep setup)
│   ├── Question-4.py     # Best‑run training, saving, evaluation, visualizations
│   └── requirements.txt  # Dependencies for Part A
│   └── README.md         # Specific instructions for Part A
│
└── partB/                # ResNet50 fine‑tuning pipeline
    ├── resnet50.py       # Training & evaluation script with freezing strategies
    ├── requirements.txt  # Dependencies for Part B
    └── README.md         # Specific instructions for Part B
```  

---

## Part A: Configurable CNN & Hyperparameter Sweeps

- Implements a customizable CNN using PyTorch Lightning (`model.py`).  
- Performs a Bayesian hyperparameter sweep (`Question-2.py`) over activations, filter sizes, dropout, learning rates, batch sizes, etc., logging to Weights & Biases.  
- Selects the best run, retrains on full training fold, saves the best weights, evaluates on test data, and visualizes filters and guided backprop maps (`Question-4.py`).

See `partA/README.md` for detailed usage instructions.

---

## Part B: Fine‑Tuning a Pre‑trained ResNet50

- Loads ImageNet‑pretrained ResNet50 and adapts the final FC layer for 10 classes.  
- Demonstrates three layer‑freezing strategies (freeze only FC, freeze up to `layer3`, or freeze first two blocks).  
- Leverages PyTorch Lightning and mixed‑precision training, with stratified 80/20 train/validation split and test evaluation.  
- Logs metrics (`train_loss`, `val_acc`, `test_acc`, etc.) to Weights & Biases and saves the fine‑tuned weights (`fine_tuned_resnet50.pth`).

See `partB/README.md` for detailed usage instructions.

---

## Dependencies

Each part has its own `requirements.txt`; install via:

```bash
# From root directory
pip install -r partA/requirements.txt
pip install -r partB/requirements.txt
```

---

## WandB Report & Code Links

- **Weights & Biases Report:**  
  [DA6401 Assignment 2 on W&B](https://wandb.ai/mrsagarbiswas-iit-madras/CNN_inaturalist_12K/reports/DA6401-Assignment-2--VmlldzoxMjIxNzA4MQ?accessToken=1s9qxr1ougqaa6pb9r90v4ttatyz3abepexi20rwhblgsuyg5y1hvd6ju5lovcmt)

- **GitHub Repository:**  
  [github.com/MrSagarBiswas/da6401_assignment2](https://github.com/MrSagarBiswas/da6401_assignment2)

---

## License

This project is released under the MIT License. See [`LICENSE`](https://opensource.org/license/mit) for details.

