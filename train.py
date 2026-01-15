import json

import pytorch_lightning as pl
import torch
from prefigure.prefigure import get_all_args
from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict
from stable_audio_tools.training import create_training_wrapper_from_config


class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f"{type(err).__name__}: {err}")


class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config


def main():
    torch.multiprocessing.set_sharing_strategy("file_system")
    args = get_all_args()
    seed = args.seed

    pl.seed_everything(seed, workers=True)

    with open(args.model_config) as f:
        model_config = json.load(f)

    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    train_dl = create_dataloader_from_config(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
        random_subset_percentage=args.random_subset_percentage,
    )

    val_dl = None

    if args.val_dataset_config:
        with open(args.val_dataset_config) as f:
            val_dataset_config = json.load(f)

        val_dl = create_dataloader_from_config(
            val_dataset_config,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sample_rate=model_config["sample_rate"],
            sample_size=model_config["sample_size"],
            audio_channels=model_config.get("audio_channels", 2),
            shuffle=False,
        )

    model = create_model_from_config(model_config)

    if args.pretrained_ckpt_path:
        copy_state_dict(model, load_ckpt_state_dict(args.pretrained_ckpt_path))

    training_wrapper = create_training_wrapper_from_config(model_config, model)

    exc_callback = ExceptionCallback()

    checkpoint_dir = args.save_dir if args.save_dir else None

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        every_n_epochs=1,
        dirpath=checkpoint_dir,
        save_top_k=-1,
        save_weights_only=True,
    )
    save_model_config_callback = ModelConfigEmbedderCallback(model_config)

    trainer = pl.Trainer(
        devices=args.num_gpus,
        accelerator="gpu",
        strategy="auto",
        precision=args.precision,
        callbacks=[ckpt_callback, exc_callback, save_model_config_callback],
        logger=None,
        log_every_n_steps=1,
        max_epochs=args.max_epochs,
        default_root_dir=args.save_dir,
        gradient_clip_val=args.gradient_clip_val,
        reload_dataloaders_every_n_epochs=0,
        num_sanity_val_steps=0,  # If you need to debug validation, change this line
    )

    trainer.fit(training_wrapper, train_dl, val_dl)


if __name__ == "__main__":
    main()
