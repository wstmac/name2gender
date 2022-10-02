# name2gender

Using character sequences in first names to predict gender. 

In this approach, I feed characters in a name one by one through a character level long short-term memory (LSTM) network built in PyTorch in the hopes of learning the latent space of all character sequences that denote gender without having to define them a priori.


## Dependencies and Installation

- Python = 3.6 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))

(optional for GPU training)
- [PyTorch = 1.7](https://pytorch.org/)
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
   \* If you cannot successfully install pytorch, you can install it with conda command, which you can find https://pytorch.org/get-started/previous-versions/ e.g.
   ```bash
   conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
   ```

## Dataset Preparation

1. go the the data directory by typing:
    ```bash
    cd data
    ```

1. clean the first name and split the whole data into train, val and testing. Defualt split ratio are 0.8, 0.1, 0.1 for train, val and test respectively.
    ```bash
    python clean_data.py
    ```

## Train
All logging files in the training process, *e.g.*, log message, checkpoints, and snapshots, will be saved to `./results` and `./logging` directory.

1. Modify config files.
   ```bash
   configuration.py
   ```

1. Run training code (I have tunned some hyperparameters and set them as the default value. <b>You can just run below command to train the network</b>). 
   ```bash
   python train.py
   ```

## Hyperparamter tuning
I implement grid search for the hyperparameter tuning. You can run multiple hyperparameter combination with the bash script.
   ```bash
   bash gird_search.sh
   ```