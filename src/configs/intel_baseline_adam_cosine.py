import torch
from torch import nn
from torchvision.transforms import v2

from src.training.config import RunConfig


class IntelCNN(nn.Module):
    def __init__(self, num_classes: int, in_channels: int):
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(16 * 4 * 4, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def __repr__(self):
        return f"IntelCNN(num_classes={self.num_classes}, in_channels={self.in_channels})"

    def architecture(self):
        return str(self.features), str(self.classifier)


def build_config():
    image_size = (150, 150)
    in_channels = 3

    def build_model(num_classes):
        return IntelCNN(num_classes=num_classes, in_channels=in_channels)

    def build_preprocessing_transform(mean, std):
        return v2.Compose([
            v2.ToImage(),
            v2.Resize(image_size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
            v2.ToPureTensor(),
        ])

    def build_optimizer(model):
        return torch.optim.Adam(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-4,
        )

    def build_scheduler(optimizer, total_steps):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            eta_min=1e-6,
            T_max=total_steps,
        )

    return RunConfig(
        run_name="intel_baseline_adam_cosine",
        batch_size=64,
        epochs=100,
        num_workers=16,
        validate_every_n_epochs=1,
        checkpoint_every_n_epochs=10,
        checkpoint_dir="checkpoints",
        seed=None,
        in_channels=in_channels,
        in_width=image_size[0],
        in_height=image_size[1],
        stats_transform=v2.Compose([
            v2.ToImage(),
            v2.Resize(image_size),
            v2.ToDtype(torch.float32, scale=True),
            v2.ToPureTensor(),
        ]),
        build_model=build_model,
        build_train_transform=build_preprocessing_transform,
        build_val_transform=build_preprocessing_transform,
        build_optimizer=build_optimizer,
        build_scheduler=build_scheduler,
    )
