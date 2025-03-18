from lightning.pytorch.cli import LightningCLI
import lightning.pytorch as L
import pyclassify.model
import pyclassify.module
import pyclassify.datamodule

cli = LightningCLI(subclass_mode_data=True, subclass_mode_model=True)
