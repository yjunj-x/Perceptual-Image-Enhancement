import pytorch_lightning as pl
import torch

from dataset.pie_dataset import PIEDataModule
from models.pie_model import PIENet


def main():
    datamodule = PIEDataModule(root_dir='/mnt/d/Datasets-hmap/barcode',
                                   input_size=(200, 320),
                                   batch_size=18,
                                   )
    datamodule.setup()
    model = PIENet(1, 1, [16, 32, 64], [64, 32])

    # Initialize trainer with callbacks
    trainer = pl.Trainer(max_epochs=100,)

    # Start training
    trainer.fit(model,
                datamodule=datamodule,
                )


if __name__ == '__main__':
    # Start main function with experiment name
    main()
