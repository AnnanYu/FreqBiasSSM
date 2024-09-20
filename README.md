# FreqBiasSSM

This is the code repository accompanying the manuscript ''Tuning Frequency Bias of State Space Models." The repository is heavily adapted from the ''state-spaces" GitHub repository (https://github.com/HazyResearch/state-spaces.git). While it contains references to existing papers and code repositories, it includes no information that reveals the identities of the manuscript authors.

## Setup

### Requirements
This repository requires Python 3.9+ and Pytorch 1.10+.
It has been tested up to Pytorch 1.13.1.
Other packages are listed in [requirements.txt](./requirements.txt).
Some care may be needed to make some of the library versions compatible, particularly torch/torchvision/torchaudio/torchtext.

Example installation:
```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Data

Basic datasets are auto-downloaded, including MNIST, CIFAR, and Speech Commands.
All logic for creating and loading datasets is in [src/dataloaders](./src/dataloaders/) directory.
The README inside this subdirectory documents how to download and organize other datasets.

### Configs and Hyperparameters

Configurations can be changed in [configs/experiment/lra](./configs/experiment/lra/).

### WandB

Logging with [WandB](https://wandb.ai/site) is built into this repository.
In order to use this, simply set your `WANDB_API_KEY` environment variable, and change the `wandb.project` attribute of [configs/config.yaml](configs/config.yaml) (or pass it on the command line e.g. `python -m train .... wandb.project=s4`).

Set `wandb=null` to turn off WandB logging.

## Execution

### Wave Prediction

The model that predicts magnitudes of waves can be trained using the script [waves.py](./waves.py). Different configurations of alpha and beta can be found in the script. Before running the script, one should first generate the data by calling [waves_gen.py](./waves_gen.py).

### Image Denoising

To train the SSM for image denoising, one has to first download the [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset to the [data](./data/) directory. Then, the model can be trained using the script [denoising.py](./denoising.py). The hyperparameters alpha and beta can be modified in the script. We also provide a Jupyter notebook at [show_faces.ipynb](./show_faces.ipynb) for visualization of the model outputs.

### LRA Benchmarks

The Long-Range Arena benchmarks can be tested by running
```
python -m train experiment=lra/bias-s4-foo
```
where `foo` is the name of the problem, choosing from `listops`, `imdb`, `aan`, `cifar`, `pathfinder`, and `pathx`.

### MovingMNIST Prediction

We use the same setup as the original [ConvSSM](https://github.com/NVlabs/ConvSSM) repository. One only needs to regenerate the modified colorful MovingMNIST data described in our manuscript using the script [MovingMNIST_gen.py](./MovingMNIST_gen.py).


