import csv
import argparse
import os
import unicodedata
import string
import random

random.seed(123)

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
FILE_NAME = 'clean_name_gender.csv'


def clean_string(name):
    """
    clean the name by removing non-letter char from name string, 
    and return lower case string
    """
    clean_name = ''.join(
        c for c in unicodedata.normalize('NFD', name)
        if unicodedata.category(c) != 'Mn'
        and c in string.ascii_letters
    )
    return clean_name.lower()


def clean_file(data_folder, file_name):
    """
    clean the input csv file by remove non-letter char from name field, and 
    save it to the new csv file: clean_{file_name}

    Args:
        data_folder (str): path to dataset root directory
        file_name (str): csv file name to be cleaned
    """


    write_csv_file = open(os.path.join(data_folder, f'clean_{file_name}'), mode='w')
    header = ['name', 'namelen', 'gender']
    clean_name_gender_writer = csv.writer(write_csv_file, delimiter=',')
    clean_name_gender_writer.writerow(header)


    with open(os.path.join(data_folder, file_name)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                # print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                c_name = clean_string(row[0])
                gender = row[1]
                clean_name_gender_writer.writerow([c_name, len(c_name), gender])
                line_count += 1
        # print(f'Processed {line_count} lines.')

    write_csv_file.close()


def split(train_pct=TRAIN_SPLIT, val_pct=VAL_SPLIT, filename=FILE_NAME):
    """
    read filename and split it to train, val, and test file

    Args:
        TRAIN_SPLIT (float): portion of training samples
        VAL_SPLIT (float): portion of validation samples
        file_name (str): csv file contains all data
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
    
    namelist = list(zip(names, namelens, genders))
    random.shuffle(namelist)

    n = len(namelist)
    tr = int(n * train_pct)
    va = int(tr + n * val_pct)

    save_csv(namelist[:tr], 'train_name_gender.csv')
    save_csv(namelist[tr:va], 'val_name_gender.csv')
    save_csv(namelist[va:], 'test_name_gender.csv')


def save_csv(datalist, filename):
    """
    save datalist to csv file

    Args:
        datalist ([(name, namelen, gender), ...]): list of tuple
        file_name (str): csv file name to be saved
    """
    csv_file = open(filename, mode='w')
    header = ['name', 'namelen', 'gender']
    csv_writer = csv.writer(csv_file, delimiter=',')
    csv_writer.writerow(header)

    for (name, namelen, gender) in datalist:
        csv_writer.writerow([name, namelen, gender])

    csv_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', help='Specify path to dataset root directory', 
                        type=str, default='.')
        
    parser.add_argument('--file_name', help='csv file_name to be cleaned', 
                        type=str, default='name_gender.csv')
    
    args = parser.parse_args()

    clean_file(args.data_folder, args.file_name)
    split()

