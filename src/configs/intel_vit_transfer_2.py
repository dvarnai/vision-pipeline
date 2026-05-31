import torch
import torchvision
from torchvision.transforms import v2

from src.training.config import RunConfig

def build_config():
    image_size = (224, 224)
    in_channels = 3

    def build_model(num_classes):
        vit = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)

        head = vit.heads.pop(-1)
        vit.heads.add_module("head", torch.nn.Linear(head.in_features, num_classes))

        return vit

    def build_preprocessing_transform(mean, std):
        return v2.Compose([
            v2.ToImage(),
            v2.Resize(image_size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
            v2.ToPureTensor(),
        ])

    def build_optimizer(model):
        return torch.optim.SGD(
            model.parameters(),
            lr=1e-4
        )

    def build_scheduler(optimizer, total_steps):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            eta_min=1e-6,
            T_max=total_steps,
        )

    return RunConfig(
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
        train_mean=[0.485, 0.456, 0.406],
        train_std=[0.229, 0.224, 0.225],
        build_scheduler=build_scheduler,
    )
