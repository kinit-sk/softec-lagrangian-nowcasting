# Softec weather nowcasting with Lagrangian Convolutional Neural Network

Repo is still work in progress. This repo contains set of scripts and notebooks for training and evaluating UNet for weather nowcasting. Majority of code was used from source code for article 
"Advection-free Convolutional Neural Network for Convective Rainfall Nowcasting" by Ritvanen et al. (2023) from https://github.com/fmidev/lagrangian-convolutional-neural-network/tree/main.

## Prerequisites

It's necessary to have the packages in [`requirements.txt`](requirements.txt) installed. If you're using Python 3.10, it might be necessary to change import of `collections` to `collections.abc` in `attrdict` module when setting up the environment for the first time (see: [StackOverflow thread](https://stackoverflow.com/questions/69381312/importerror-cannot-import-name-mapping-from-collections-using-python-3-10)).

## How to use

1. Create the training, validation and test datasets in Lagrangian coordinates. On Ubuntu, you can run your training in tmux by using following command in bash:

```bash
tmux new-session -d -s <name of your session> \; send-keys "python transform_shmu_to_lagrangian.py <config-sub-path> <train/valid/test> --nworkers <no-of-workers>" Enter
```

where `<name-of-your-session>` is the name of the tmux session on which the script will be run, `<config-sub-path>` is a sub-directory of the `config` directory where the configuration files are located, `<train/valid/test>` is the name of the dataset to be generated (that will be injected to the `{split}` placeholder in the `date_list` variable), and `<no-of-workers>` indicates the number of dask processes used to run the transformation.
2. Train the L-CNN model. On Ubuntu, you can run your training in tmux by using following command in bash:

```bash
tmux new-session -d -s <name-of-your-session> \; send-keys "python train_model.py <config-sub-path> &> <training-name>.out" Enter
```

where `<name-of-your-session>` is the name of the tmux session on which the script will be run, `<config-sub-path>` is a sub-directory of the `config` directory where the configuration files are located and `<training-name>` is the name you would like to give to your output file (e.g. same as the other outputs).
3. Create nowcasts for the L-CNN model. On Ubuntu, you can get your predictions in tmux by using following command in bash:

```bash
tmux new-session -d -s <name-of-your-session> \; send-keys "python predict_model.py <path-to-model-checkpoint>.ckpt <config-sub-path> -l <train/valid/test> &> <pred-name>.out" Enter
```

where `<name-of-your-session>` is the name of the tmux session on which the script will be run, `<path-to-model-checkpoint>` is the path to your model's best result in form of checkpoint, `<config-sub-path>` is a sub-directory of the `config` directory where the configuration files are located, `<train/valid/test>` is the name of the dataset to be generated (that will be injected to the `{split}` placeholder in the `date_list` variable) and `<pred-name>` is the name you would like to give to your output file (e.g. same as the other outputs).

## Scripts

| Script | Description |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `transform_shmu_to_lagrangian.py` | Script for transforming SHMU radar composite to Lagrangian coordinates. Reads parameters from `transform_shmu_to_lagrangian_datasets.yaml` and `transform_shmu_to_lagrangian_params.yaml` configuration files. |
| `train_model.py` | Script for training the L-CNN model. Reads parameters from `lcnn.yaml`, `lagrangian_datasets.yaml` and `output.yaml` configuration files. |
| `predict_model.py` | Script for producing predictions from the premade datelist in HDF5 format. Reads parameters from `lcnn.yaml`, `lagrangian_datasets.yaml` and `output.yaml` configuration files. |

## Subfolders

| Subfolder | Description |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `configs` | Collection of all configuration files necessary for correct functionality of scripts. |
| `datamodules` | Contains custom PyTorch Lightning LightningDataModules. |
| `datasets` | Contains custom PyTorch dataset modules. |
| `modelcomponents` | Contains neural network model components implemented using PyTorch. |
| `models` | Contains neural network models implemented using PyTorch and Lightning. |
| `utils` | Contains utility functions that are of use when training neural network. |



