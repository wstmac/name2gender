# name2gender

Using character sequences in first names to predict gender. 

In this approach, I feed characters in a name one by one through a character level long short-term memory (LSTM) network built in PyTorch in the hopes of learning the latent space of all character sequences that denote gender without having to define them a priori.


## Dependencies and Installation

- Python >= 3.6 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))

(optional for GPU training)
- [PyTorch >= 1.7](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

1. Clone repo

   ```bash
   git clone https://github.com/wstmac/name2gender.git
   ```

1. Install Dependencies

   ```bash
   cd name2gender
   pip install -r requirements.txt
   ```


## Dataset Preparation

1. go the the data directory by typing:
    ```bash
    cd data
    ```

1. clean the first name and split the whole data into train, val and testing.
    ```bash
    python clean_data.py
    ```

## Train
All logging files in the training process, *e.g.*, log message, checkpoints, and snapshots, will be saved to `./results` and `./logging` directory.

1. Modify config files.
   ```bash
   configuration.py
   ```

1. Run training code.
   ```bash
   python train.py
   ```

## Hyperparamter tuning
I implement grid search for the hyperparameter tuning. You can run multiple hyperparameter combination with the bash script.
   ```bash
   bash gird_search.sh
   ```