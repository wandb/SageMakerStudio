# Using Wandb and SageMaker Studio

This repo is acompanied of a blog post in AWS and on [fully-connected](http://wandb.me/aws_studio)

## Install
This repo can be configured using the pip: [requirements.txt](requirements.txt) or the conda: [environment.yml](environment.yml). SageMaker Studio can be configured using both.

## Running the experiments
- [01_data_processing.ipynb](01_data_processing.ipynb) presents how to log the CamVid dataset and how to do initial EDA.
- [02_semantic_segmentation.ipynb](02_semantic_segmentation.ipynb) Train a model to performn semantic segmentation on the logged dataset.
- [train.py](train.py) is a refactor of the training notebook into a script to make use of sweeps on the terminal.

## Sweep
We will perform an Hyper Parameter Sweep using the [sweep.yaml](sweep.yaml) configuration file. To run the sweep with defaults values, use:

```bash
$ wandb sweep sweep.yaml
```

and then launch an agent using the suggested command line output of command above

```bash
$ wandb agent <sweep_id>
```

> Note: this repo uses `ml_collections` to configure parameters. You can check the [config.py](config.py) file to inspect defaults.
