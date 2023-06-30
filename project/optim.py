import torch.optim as optim
import numpy as np

def get_optimizer(params, optimizer_type: str="LBFGS"):
    """ Returns optimizer for the given parameters.
    
    Parameters
    ----------
    params : torch.nn.parameter.Parameter
        Parameters of the network.
    optimizer_type : str, optional
        Type of optimizer to use. The default is "LBFGS".
        The optimizer types are:
            LBFGS: Limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm
            LBFGS2: Limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm with more iterations
            ADAM: Adaptive Moment Estimation
            NADAM: Nesterov Adam

    Returns
    -------
    optimizer : torch.optim
        Optimizer for the given parameters.

    """
    # optimizer = torch.optim.NAdam(self.net.parameters())
    # optimizer = optim.LBFGS(self.net.parameters(), lr=float(0.5), max_iter=5000, max_eval=100000, history_size=150,
    #                     line_search_fn="strong_wolfe",
    #                     tolerance_change=1.0 * np.finfo(float).eps, tolerance_grad=1.0 * np.finfo(float).eps)
    if optimizer_type == "LBFGS":
        optimizer = optim.LBFGS(params, lr=float(0.5), max_iter=20, max_eval=50, history_size=150,
                            line_search_fn="strong_wolfe",
                            tolerance_change=1.0 * np.finfo(float).eps, tolerance_grad=1.0 * np.finfo(float).eps)
    elif optimizer_type == "LBFGS2":
        optimizer = optim.LBFGS(params, lr=float(0.5), max_iter=5000, max_eval=100000, history_size=150,
                        line_search_fn="strong_wolfe",
                        tolerance_change=1.0 * np.finfo(float).eps, tolerance_grad=1.0 * np.finfo(float).eps)           
    elif optimizer_type == "ADAM":
        optimizer = optim.Adam(params, lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    elif optimizer_type == "NADAM":
        optimizer = optim.NAdam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    else:
        raise ValueError("Optimizer type not recognized. Please choose from LBFGS, ADAM, NADAM.")
    return optimizer