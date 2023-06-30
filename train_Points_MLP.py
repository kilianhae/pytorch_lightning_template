"""Train a 2D conditional temperature network with a GNN architecture."""

if __name__ == "__main__":
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    from project.train import PROJECTTrainer

    # Parameters
    resume = True
    model_type = "MLP"
    problem_type = "Points"

    data_kwargs = {"features_num": 1}

    global_kwargs = {"seed": 42}
    dataset_kwargs = {"offset": 10, "Tm": 0.3, "seed": 42, "contours": True, "out_type": "polar", "graph": True, "in_type":"t_angle"}
    model_kwargs = {"emb_dim": 128, "num_layer": 5, "norm": "BatchNorm"}
    training_kwargs = { "max_epochs": 5022, "batch_size": 32}
    valid_kwargs = {"batch_size": 1}
    lightning_kwargs = {"optimizer": "LBFGS"}
    trainer = PROJECTTrainer(model_type, problem_type, data_kwargs, dataset_kwargs, model_kwargs, training_kwargs, lightning_kwargs, global_kwargs, resume)
    trainer.train()
    