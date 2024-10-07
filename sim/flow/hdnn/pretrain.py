# from __future__ import annotations

# import logging

# from omegaconf import DictConfig
# from pytorch_lightning import Trainer, seed_everything
# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.loggers import TensorBoardLogger

# from sim.datasets import load_dataset
# from sim.models import LitModel
# from sim.utils import get_params_from_loader


# def pretrain(args: DictConfig) -> None:
#     logger = logging.getLogger("pretrain")

#     # seed
#     seed_everything(args["seed"])

#     # Load dataset
#     dataset: str = args["data"]["dataset"]
#     train_loader, val_loader, _ = load_dataset(dataset, args["data"], False)

#     # get params from loader
#     num_classes, input_channel, cnn_output_dim = get_params_from_loader(train_loader)

#     # model
#     lit_model = LitModel(num_classes, input_channel, cnn_output_dim)

#     # checkpoint
#     checkpoint_callback = ModelCheckpoint(
#         monitor="val_acc",  # Specify the metric to monitor
#         dirpath="my_model/",  # Directory where the checkpoints will be saved
#         filename=dataset,  # Checkpoint file name
#         save_top_k=1,  # Save only the best checkpoint
#         mode="max",  # Maximize val acc
#     )

#     # logger
#     train_logger = TensorBoardLogger("lightning_logs", name="pretrain")

#     # trainer
#     epochs: int = args["pretrain"]["epochs"]
#     trainer = Trainer(
#         callbacks=[checkpoint_callback],
#         max_epochs=epochs,
#         accelerator="cuda",
#         log_every_n_steps=10,
#         logger=train_logger,
#     )
#     trainer.fit(lit_model, train_loader, val_loader)
