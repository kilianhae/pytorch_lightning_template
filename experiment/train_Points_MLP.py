"""Train a 2D conditional temperature network with a GNN architecture."""

if __name__ == "__main__":
    from project.train import PROJECTTrainer

    # Parameters
    resume = True

    # defines the pytorch model to use
    model_type = "MLP"

    # defines the lightning model to use (and potentially (even with combination with model_type) the data to load))
    problem_type = "Points"

    # defines the parameters on what data to load (e.g. we have different iterations of the data, or want to load only partial features, etc.)
    data_kwargs = {"features_num": 1}


    global_kwargs = {"seed": 42}

    # defines genral paramenters for the datset (universal to train and eval)
    dataset_kwargs = {"offset": 10, "Tm": 0.3, "seed": 42, "contours": True, "out_type": "polar", "graph": True, "in_type":"t_angle"}

    # defines the parameters for the model (e.g. number of layers, number of nodes, etc.)
    model_kwargs = {"emb_dim": 64, "num_layer": 3, "norm": "BatchNorm"}

    # passed to the dataloaders (e.g. number of epochs, batch size, etc.)
    loader_kwargs = {"batch_size": 32}

    # passed to the trainer: number of epochs, number of devices, accelerator
    trainer_kwargs = {"max_epochs": 50, "accelerator": "auto", "devices": 1}

    # passed to the lightning module (e.g. optimizer, loss, etc.)
    lightning_kwargs = {"optimizer": "NADAM", "loss_type": "L2", "weight_decay": 0, "lr": 1e-04}

    trainer = PROJECTTrainer(model_type, problem_type, data_kwargs, dataset_kwargs, model_kwargs, loader_kwargs, lightning_kwargs, global_kwargs, trainer_kwargs, resume)
    trainer.train()

