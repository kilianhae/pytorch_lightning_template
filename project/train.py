# here set up everything for training and testing with lightning. This is the first file getting called.

from pathlib import Path

import lightning.pytorch as pl
#import callbacks
from lightning.pytorch.callbacks import ModelCheckpoint, Callback, Timer
# import loggers
from lightning.pytorch.loggers import WandbLogger

import wandb

import torch.utils as utils

from project.dataset import Dataset, load_dataset

from project.model import *

from project.conf import OUTPUTDIR, WANDB_PROJECT, WANDB_MODE

from project.model import PointMLP, MLP


def select_model(problem_type: str):
    if problem_type == "Points":
        LightningModel = PointMLP
    else:
        raise ValueError(f"Problem type {problem_type} not implemented")
    return LightningModel

def select_net(model_type: str):
    if model_type == "MLP":
        Net = MLP
    else:
        raise ValueError(f"Model type {model_type} not implemented")
    return Net

class PROJECTTrainer:
    def __init__(self, model_type, problem_type, data_kwargs, dataset_kwargs, model_kwargs, training_kwargs, lightning_kwargs, global_kwargs, resume=True, log_model=True, OUTPUTDIR=OUTPUTDIR):
        self.model_type = model_type
        self.problem_type = problem_type
        self.data_kwargs = data_kwargs
        self.dataset_kwargs = dataset_kwargs
        self.model_kwargs = model_kwargs
        self.global_kwargs = global_kwargs
        self.training_kwargs = training_kwargs
        self.valid_kwargs = training_kwargs
        self.lightning_kwargs = lightning_kwargs
        self.name = self._get_name()
        self.OUTPUTDIR = OUTPUTDIR
        self.save_dir = OUTPUTDIR / Path(self.name)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.wandb_session = wandb.init(project=WANDB_PROJECT, mode=WANDB_MODE, dir=self.save_dir, name=self.short_name, config=self.wandb_config, resume=resume)
        self.resume = resume
        self.log_model = log_model

        pl.seed_everything(global_kwargs["seed"], workers=True)

        self._init_dataset()
        self._init_model()
        self._init_trainer()

    @property
    def wandb_config(self):
        config = {}
        config["problem_type"] = self.problem_type
        config["model_type"] = self.model_type
        config["data_kwargs"] = self.data_kwargs
        config["dataset_kwargs"] = self.dataset_kwargs
        config["valid_kwargs"] = self.valid_kwargs
        config["model_kwargs"] = self.model_kwargs
        config["training_kwargs"] = self.training_kwargs
        config["lightning_kwargs"] = self.lightning_kwargs
        return config
    
    @property
    def short_name(self):
        name = f"{self.problem_type}_{self.model_type}"
        for key, value in self.lightning_kwargs.items():
            name += f"_{key}_{value}"
        for key, value in self.data_kwargs.items():
            name += f"_{key}_{value}"
        return name

    def _get_name(self):
        name = f"{self.problem_type}_{self.model_type}"
        #for key, value in self.data_kwargs.items():
        #    name += f"_{key}_{value}"
        for key, value in self.dataset_kwargs.items():
            name += f"_{key}_{value}"
        for key, value in self.model_kwargs.items():
            name += f"_{key}_{value}"
        for key, value in self.global_kwargs.items():
            name += f"_{key}_{value}"
        for key, value in self.training_kwargs.items():
            name += f"_{key}_{value}"
        for key, value in self.lightning_kwargs.items():
            name += f"_{key}_{value}"
        assert len(name) < 255
        return name
    
    def _init_dataset(self):
        # load train, validation and potentially the test dataset and set the corresponding dataloaders
        # the problem_type specifies which lightning mdel we have to use. e.g. are we using RNNs, Graphs, Images, etc. They will all need different fit and predict functions!
        
        # load the data or give the path to the data
        data = load_dataset(problem_type=self.problem_type, **self.data_kwargs)

        # instantiate the datasets
        self.dataset_train = Dataset(data, **self.dataset_kwargs, validation=False)
        self.dataset_valid = Dataset(data, **self.dataset_kwargs, validation=True)

        self.train_loader = utils.data.DataLoader(self.dataset_train, batch_size=self.training_kwargs["batch_size"], shuffle=True,  num_workers=8)
        self.validation_loader = utils.data.DataLoader(self.dataset_valid, self.valid_kwargs["batch_size"], shuffle=False,  num_workers=8)

    def _init_model(self):
        # load the pytorch net that will be wrapped by the lightning model
        Net = select_net(self.model_type)
        self.net = Net(1, **self.model_kwargs)

        # load the LightningModule model
        LightningModel = select_model(self.problem_type)
        self.model = LightningModel(self.net, **self.lightning_kwargs)


    def _init_trainer(self):
        self.wandb_logger = WandbLogger(experiment=self.wandb_session, project=WANDB_PROJECT, log_model=self.log_model, save_dir=self.save_dir, name=self.short_name, config=self.wandb_config)
        
        ## define your Checkpoint callbacks here
        self.checkpoint_callback = ModelCheckpoint(
             every_n_epochs = 1,
             dirpath = self.save_dir / Path("checkpoints"), # if left away this defaults to default_root_dir of trainer
             filename = "model-{epoch:02d}-{train_epoch_total_loss:.6f}",
             monitor = "train_epoch_MSELoss",
             save_top_k = 10,
             save_last = True
        )

        self.trainer = pl.Trainer(accelerator='mps', devices=1, logger=self.wandb_logger, callbacks=[self.checkpoint_callback], max_epochs=5000,  default_root_dir=self.save_dir)
        
    def train(self):
        # check wether we should resume our model or not an start the training loop
        if self.resume:
            self.trainer.fit(self.model, self.train_loader, self.validation_loader, ckpt_path = "last")
        else:
            self.trainer.fit(self.model, self.train_loader, self.validation_loader)

        self.trainer.validate(self.model, self.validation_loader, ckpt_path = "best")