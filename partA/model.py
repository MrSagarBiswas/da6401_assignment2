import torch
import torch.nn as nn
import pytorch_lightning as pl

# Define the configurable CNN model.
class ConfigurableCNN(pl.LightningModule):
    def __init__(self, config):
        super(ConfigurableCNN, self).__init__()
        self.save_hyperparameters()  # Saves config to checkpoint

        self.config = config

        # Map activation string to function
        activation_map = {
            'ReLU': nn.ReLU,
            'GELU': nn.GELU,
            'SiLU': nn.SiLU,
            'Mish': nn.Mish,
        }
        activation_fn = activation_map.get(config.get('activation', 'ReLU'), nn.ReLU)

        # Build 5 convolution blocks: Conv -> Activation -> (BatchNorm) -> MaxPool -> (Dropout)
        conv_layers = []
        in_channels = 3  # Assuming RGB images
        num_layers = config.get('num_conv_layers', 5)
        filters = config.get('filters', [32, 64, 128, 256, 512])
        kernel_sizes = config.get('kernel_sizes', [3] * num_layers)
        use_batchnorm = config.get('use_batchnorm', False)
        dropout_rate = config.get('dropout_rate', 0.0)

        for i in range(num_layers):
            out_channels = filters[i]
            kernel_size = kernel_sizes[i]
            padding = kernel_size // 2  # To maintain spatial dimensions before pooling
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
            block = [conv, activation_fn()]
            if use_batchnorm:
                block.append(nn.BatchNorm2d(out_channels))
            # Append max pooling layer to reduce spatial dimensions by a factor of 2
            block.append(nn.MaxPool2d(2))
            if dropout_rate > 0:
                block.append(nn.Dropout2d(dropout_rate))
            conv_layers.extend(block)
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)

        # After 5 pooling operations, the spatial dimensions of a 128x128 image become:
        # 128 / (2^5) = 4 (assuming the dimensions divide evenly)
        final_spatial = 4
        fc_input_dim = filters[-1] * (final_spatial ** 2)

        # Define the fully connected part: Dense layer and final output layer with 10 classes.
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_dim, config.get('dense_neurons', 256)),
            activation_fn(),
            nn.Dropout(config.get('dropout_rate_dense', 0.0)),
            nn.Linear(config.get('dense_neurons', 256), 10)
        )

        # Loss function and learning rate
        self.criterion = nn.CrossEntropyLoss()
        self.lr = config.get('lr', 1e-3)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer