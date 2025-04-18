# Fine-Tuning ResNet50 on iNaturalist 12K

This repository demonstrates how to fine-tune a pre-trained ResNet50 model on a custom dataset (iNaturalist 12K) using PyTorch Lightning. The pipeline covers data preparation, stratified splitting, model freezing strategies, training, validation, testing, manual evaluation, and saving the fine-tuned weights.

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Freezing Strategies](#freezing-strategies)
- [Training](#training)
- [Evaluation](#evaluation)
- [Logging with Weights & Biases](#logging-with-weights--biases)
- [Saving the Model](#saving-the-model)
- [Usage Example](#usage-example)
- [License](#license)

## Project Structure
```
├── resnet50.py           # Main training and evaluation script
├── fine_tuned_resnet50.pth  # Saved model weights (after training)
└── README.md             # This file
```

## Prerequisites

- Python 3.8+
- CUDA-enabled GPU (optional but recommended)
- Access to the iNaturalist 12K dataset structured as:
  ```
  DATA_DIR/
  ├── train/
  │   ├── class0/
  │   │   ├── img1.jpg
  │   │   └── ...
  │   └── class1/
  │       └── ...
  └── test/
      ├── class0/
      │   ├── img1.jpg
      │   └── ...
      └── class1/
          └── ...
  ```

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/MrSagarBiswas/da6401_assignment2.git
   cd da6401_assignment2/partB
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\\Scripts\\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   If `requirements.txt` is not present, you can install directly:
   ```bash
   pip install torch torchvision pytorch-lightning wandb scikit-learn
   ```

## Data Preparation

- Set the `DATA_DIR` in `resnet50.py` to point to your dataset root.
- Ensure the training data resides in `DATA_DIR/train` and test data in `DATA_DIR/test`.
- The script automatically performs a stratified split of 80% training and 20% validation using `StratifiedShuffleSplit`.

## Freezing Strategies

You can choose one of three freezing strategies by uncommenting the corresponding function call in `resnet50.py`:

1. **Freeze All Layers Except the Fully Connected (FC) Layer**
   ```python
   freeze_all_but_fc(model)
   ```
2. **Freeze Up to a Specific Layer** (e.g., `layer3`)
   ```python
   freeze_up_to_layer(model, 'layer3')
   ```
3. **Freeze the First Two Residual Blocks** (`layer1` and `layer2`)
   ```python
   freeze_first_two_blocks(model)
   ```

By default, the script uses the `freeze_up_to_layer(model, 'layer3')` strategy.

## Training

The script uses PyTorch Lightning for organized training and multi-GPU support.

- **Batch size**: 32
- **Image size**: 224×224 with ImageNet normalization
- **Optimizer**: Adam with learning rate `1e-5` (configurable in `FineTuneModule`)
- **Epochs**: 10
- **Precision**: 16-bit if GPU available, otherwise 32-bit

To start training:
```bash
python resnet50.py
```

The Lightning `Trainer` will automatically train and validate the model. Training metrics (`train_loss_resnet50`, `train_acc_resnet50`, `val_loss_resnet50`, `val_acc_resnet50`) are logged every 10 steps.

## Evaluation

### Automatic Test Evaluation

After training, `trainer.test(...)` runs on the held-out test set and logs:
- `test_loss_resnet50`
- `test_acc_resnet50`

### Manual Evaluation

The script also defines a helper function `evaluate_model(model, dataloader, device)`:
```python
def evaluate_model(model, dataloader, device):
    ...
    return accuracy
```

After testing, it prints:
```
Train Accuracy: XX.XX%
Validation Accuracy: XX.XX%
Test Accuracy: XX.XX%
```

## Logging with Weights & Biases

- The script initializes a W&B logger:
  ```python
  wandb_logger = WandbLogger(
      project="CNN_inaturalist_12K",
      name=f"fine_tune_resnet50_{Strategy}"
  )
  ```
- Metrics and model gradients are automatically tracked in your W&B dashboard.
- You can view training curves and compare different freezing strategies side by side.

## Saving the Model

After training and evaluation, the model state dictionary is saved to:
```bash
torch.save(model.state_dict(), 'fine_tuned_resnet50.pth')
```

You can load the model later with:
```python
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load('fine_tuned_resnet50.pth'))
model.eval()
```

## Usage Example

```bash
# Train and validate
python resnet50.py

# After completion, view results in W&B and local console output.
```


## License

This project is released under the MIT License. See [LICENSE](https://opensource.org/license/mit) for details.

