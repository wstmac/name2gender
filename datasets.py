import random
import unicodedata
import csv
import string
import time
from os import getcwd
import torch
from torch.utils import data
from torch.autograd import Variable


random.seed(123)

FILE_NAME = './data/clean_name_gender.csv' # default path of csv file that contains whole data
TRAIN_FILE_NAME = './data/train_name_gender.csv' # default path of training data
VAL_FILE_NAME = './data/val_name_gender.csv' # default path of validation data
TEST_FILE_NAME = './data/test_name_gender.csv' # default path of testing data

# constant
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
ALL_LETTERS = string.ascii_lowercase
ALL_GENDERS = ["M", "F"]
N_LETTERS = len(ALL_LETTERS)
N_GENDERS = len(ALL_GENDERS)
LONGEST_NAME = 16


# data accessors
def load_names(filename=FILE_NAME):
    """loads all names and genders from the dataset
    Args:
        filename (optional): path to the desired csv file
            (default: DATASET_FN)
    Return:
        (names, genders):
            names: list of names - e.g., ["john", "bob", ...]
            namelens: list of namelens - e.g., [4, 3, ...]
            genders: list of genders - e.g., ["M", "M", "F", ...]
    """

    names = []
    namelens = []
    genders = []

    with open(filename) as csv_data_file:
        csv_reader = csv.reader(csv_data_file)
        next(csv_reader, None) # skip the headers
        for row in csv_reader:
            names.append(row[0])
            namelens.append(int(row[1]))
            genders.append(row[2])

    return names, namelens, genders


def load_dataset(filename=FILE_NAME):
    """Returns the name->gender dataset ready for processing
    Args:
        filename (string, optional): path to dataset file
    Return:
        namelist (list(String,String)): list of (name, gender) records
    """
    names, namelens, genders = load_names(filename)
    namelist = list(zip(names, namelens, genders))
    return namelist


def load_datasets(train_file=TRAIN_FILE_NAME, val_file=VAL_FILE_NAME, test_file=TEST_FILE_NAME):
    """
    Load train, val and test csv file.
    """
    train_set = load_dataset(train_file)
    val_set = load_dataset(val_file)
    test_set = load_dataset(test_file)

    return train_set, val_set, test_set


def split_dataset(train_pct=TRAIN_SPLIT, val_pct=VAL_SPLIT, filename=FILE_NAME):
    """
    split the who dataset into train, val and test datasets
    """
    dataset = load_dataset(filename)
    n = len(dataset)
    tr = int(n * train_pct)
    va = int(tr + n * val_pct)
    return dataset[:tr], dataset[tr:va], dataset[va:]  # Trainset, Valset, Testset


def dataset_dicts(dataset=load_dataset()):
    """
    build dict for name gender pair
    """
    name_gender = {}
    gender_name = {}
    for name, gender in dataset:
        name_gender[name] = gender
        gender_name.setdefault(gender, []).append(name)
    return name_gender, gender_name


# data manipulators
def name_to_tensor(name, cuda=False):
    """converts a name to a vectorized numerical input for use with a nn
    each character is converted to a one hot (n, 1, 26) tensor
    Args:
        name (string)
    Return:
        tensor (torch.tensor)
    """

    # name = clean_str(name)
    tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    torch.zeros(len(name), N_LETTERS, out=tensor)
    for li, letter in enumerate(name):
        letter_index = ALL_LETTERS.find(letter)
        tensor[li][letter_index] = 1
    return tensor


def tensor_to_name(name_tensor):
    ret = ""
    for letter_tensor in name_tensor.split(1):
        nz = letter_tensor.data.nonzero()
        if torch.numel(nz) != 0:
            ret += (string.ascii_lowercase[nz[0, 1]])
    return ret


def gender_from_output(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    gender_i = top_i[0][0]
    return ALL_GENDERS[gender_i], gender_i


class NameGenderDataset(data.Dataset):
    def __init__(self, data, longest_name=LONGEST_NAME):
        """data should be a list of (name, gender) string pairs"""
        self.data = data
        self.longest_name = longest_name
        self.names, self.namelens, self.genders = zip(*data)

    def __getitem__(self, index):
        name = [-1]*self.longest_name
        for i, letter in enumerate(self.names[index]):
            name[i] = ALL_LETTERS.find(letter)
        name = torch.LongTensor(name)
        namelen = self.namelens[index]
        gender = ALL_GENDERS.index(self.genders[index])

        return name, namelen, gender

    def index_of(self, name):
        return self.names.index(name)

    def __len__(self):
        return len(self.data)