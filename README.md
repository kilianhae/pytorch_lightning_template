
This is how I build projects using pytorch lighnting :)

It serves as a template to provide structure, organization, replacability and some scalability.


# Installation
Poetry is used to manage the dependencies of this project.
To install this project along with its dependencies run the following commands:

0. If you want to use your own projects name the replace all mentions of the string 'project' with your 'projectname'


2. Clone the repository:
```bash
git clone git@github.com:kilianhae/pytorch_lightning_template.git
cd pytorch_lightning_template
```

3. Setup your environment:
(creates an env, installs dependencies and the porject directory as a package)
```bash
poetry init
poetry install
```

4. Launch your environment
```bash
poetry shell
exit  #to leave the environment again
```
5. (If you have not worked with wandb before, then login)
```bash
wandb login
```

6. Configure and run :)
```bash
python experiment/train_Points_MLP.py
```

# Features
## Lightning and Model Structure
The overall model structure enables clean implementation of multiple modelling apporaches using various different models. The _problem_type_ variable indicates whch lightning Module to use from the selection in _project/models/lightning_ and the _model_type_ indicates which specific model to use within the lightning module. Oftentimes we have several modelling approaches for example using an MLP and a GNN which need two different dataformats, different step function etc. which is why they need two seperate custom lightning modules. 
However within the family of GNN apporaches we might want to use a platora of different models, these are often times invariant to the training loop and can thus share the same Lightning Module loaded with different models.
This is only for using entirely different models of the same species!
 If you have small changes to the models then define them in the _model_kwargs_.

## Custom Datasets
Define your custom datasets within project/dataset.py.
Use the dataset_kwargs to define all necessary customizations (and the problem_type variable). 
For now its designed to use a single dataset that adjusts to the dataformat as hinted by dataset_kwargs and  problem_type, but this can be extended by adding a select_dataset function in project/train.py

## Logging
Logging is done online using Weights & Biases as it is my preferred option.
Basic logging of train and validation metrics is implemented and can be easily extended to logging gradients or media by adding custom callbacks.
  