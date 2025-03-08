from model.unet_autoenc import BeatGANsAutoencConfig, BeatGANsAutoencModel
from model.lit_model import LitModel
from dataset import ImageNet
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from templates import imagenet256_autoenc


print("Model Config")
# model_conf = BeatGANsAutoencConfig(
#     image_size=64, in_channels=3, model_channels=128,
#     enc_out_channels=256, dropout=0.1
# )

print("Train config")
train_conf = imagenet256_autoenc()

print("Dataset")
dataset = ImageNet(
    "validation",
    [151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200],
    None,
    "/mnt/evafs/groups/mi2lab/ljaremek"
    )

print("Dataloader")
train_loader = DataLoader(
    dataset, batch_size=train_conf.batch_size, shuffle=True
    )

print(train_conf.model_name)

print("Model")
model = BeatGANsAutoencModel(conf=train_conf.model_conf)
model = LitModel(train_conf)

print("Checkpoint")
checkpoint_callback = ModelCheckpoint(
    dirpath=train_conf.logdir, save_last=True, save_top_k=1
    )

print("LR")
lr_monitor = LearningRateMonitor(logging_interval="step")

print("Trainer")
trainer = Trainer(
    max_steps=train_conf.total_samples // train_conf.batch_size_effective,
    devices=1,
    precision=16 if train_conf.fp16 else 32,
    callbacks=[checkpoint_callback, lr_monitor]
    )

print("FIT")
trainer.fit(model, train_loader)
