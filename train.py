import math
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.backends.cudnn as cudnn
import argparse
import time
import os
import logging
import shutil
from tqdm import tqdm
from tensorboardX import SummaryWriter


from datasets import *
from models import *
import configuration



def main():
    """
    Training and validation.
    """
    global args, summary_writer, log
    args = configuration.parser_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # folder name
    save_folder = f'results/{args.arch}_{args.batch_size}_{args.dropout}_{args.n_hidden}_{args.learning_rate}_{args.weight_decay}_{args.epoch}'
    if os.path.exists(save_folder):
        timestr = time.strftime("%Y-%m-%d-%H:%M:%S")
        save_folder = save_folder + '_' + timestr
    else:
        os.mkdir(save_folder)


    # initial logger
    log = setup_logger(save_folder + '/training.log')
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    # SummarWriter
    summary_writer = SummaryWriter(f'logs/{args.arch}_{args.n_hidden}_{args.learning_rate}_{args.epoch}')


    #load model
    # model = RNN(LONGEST_NAME, n_hidden, N_GENDERS)
    model = LSTM(LONGEST_NAME, args.n_hidden, N_GENDERS)
    model = model.to(device)

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, 
                                weight_decay=args.weight_decay)

    # loss function
    criterion = nn.NLLLoss()

    # load datasets
    # collate_fn = name_gender_collate_cuda if args.cuda else name_gender_collate
    train_set, val_set, test_set = load_datasets()
    train_loader = data.DataLoader(NameGenderDataset(train_set), args.batch_size, num_workers=args.num_workers,
                                  shuffle=True, pin_memory=True)
    val_loader = data.DataLoader(NameGenderDataset(val_set), args.batch_size, num_workers=args.num_workers,
                                  shuffle=True, pin_memory=True)
    test_loader = data.DataLoader(NameGenderDataset(test_set), args.batch_size, num_workers=args.num_workers,
                                  shuffle=True, pin_memory=True)

    best = 0.5
    best_epoch = -1
    for epoch in tqdm(range(args.start_epoch, args.epoch)):
        train(train_loader, model, criterion, optimizer, epoch, device)
        acc = val(train_loader, model, criterion, epoch, device)

        is_best = acc > best
        best = max(acc, best)

        if is_best:
            best_epoch = epoch
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'validate_acc': acc,
                'optimizer': optimizer.state_dict(),
            }, folder=save_folder, filename='model_best.pth')
        
        if (epoch+1) % args.saving_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'validate_acc': acc,
                'optimizer': optimizer.state_dict(),
            }, folder=save_folder, filename=f'{epoch}.pth')
    
    summary_writer.add_scalar("Best Validation Accuracy", best, best_epoch)

    # test with best model
    checkpoint = torch.load(f'{save_folder}/model_best.pth')
    best_model = LSTM(LONGEST_NAME, args.n_hidden, N_GENDERS)
    best_model.load_state_dict(checkpoint['state_dict'])
    best_model = best_model.to(device)
    test(train_loader, best_model, device)



def train(train_loader, model, criterion, optimizer, epoch, device):
    """
    Train the model
    """
    model.train()
    losses = AverageMeter()

    for i, (names, name_lengths, genders) in enumerate(train_loader):
        # move data to the device
        names = names.to(torch.float32).to(device)
        name_lengths = name_lengths.to(device)
        genders = genders.to(device)
    
        outputs, sort_ind = model(names, name_lengths)
        genders = genders[sort_ind]

        loss = criterion(outputs, genders)

        # record loss
        losses.update(loss.item(), names.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.logging_freq == 0:
            log.info(f'Epoch [{epoch}][{i}/{len(train_loader)}] {losses.val}({losses.avg})')
    
    summary_writer.add_scalar("Training Loss", losses.avg, epoch)


def val(val_loader, model, criterion, epoch, device):
    """
    Compute accuracy on validation datasets with the input model
    """
    model.eval()
    losses = AverageMeter()
    acc = AverageMeter()

    for i, (names, name_lengths, genders) in enumerate(val_loader):
        batch_size = names.size(0)

        names = names.to(torch.float32).to(device)
        name_lengths = name_lengths.to(device)
        genders = genders.to(device)
    
        outputs, sort_ind = model(names, name_lengths)
        genders = genders[sort_ind]
        
        loss = criterion(outputs, genders)
        losses.update(loss.item(), names.size(0))

        # compute accuracy
        _, topi = torch.max(outputs, 1)
        curr_acc = (topi == genders).sum().item()/batch_size
        acc.update(curr_acc)

    log.info(f'Epoch [{epoch}]: validate loss: {losses.avg}\t'
            f'accuracy: {acc.avg}'    )

    summary_writer.add_scalar("Validation Loss", losses.avg, epoch)
    summary_writer.add_scalar("Validation Accuracy", acc.avg, epoch)
    return acc.avg


def test(test_loader, model, device):
    """
    Compute accuracy on test datasets with the input model
    """
    model.eval()
    acc = AverageMeter()

    for i, (names, name_lengths, genders) in enumerate(test_loader):
        batch_size = names.size(0)

        names = names.to(torch.float32).to(device)
        name_lengths = name_lengths.to(device)
        genders = genders.to(device)
    
        outputs, sort_ind = model(names, name_lengths)
        genders = genders[sort_ind]

        # compute accuracy
        _, topi = torch.max(outputs, 1)
        curr_acc = (topi == genders).sum().item()/batch_size
        acc.update(curr_acc)

    log.info(f'Test accuracy: {acc.avg:.4f}')
    print(f'Test accuracy: {acc.avg:.4f}')

    summary_writer.add_scalar("Test Accuracy", acc.avg)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, folder='result/default', filename='checkpoint.pth'):
    torch.save(state, folder + '/' + filename)


def setup_logger(filepath):
    file_formatter = logging.Formatter(
        "[%(asctime)s %(filename)s:%(lineno)s] %(levelname)-8s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger('example')
    # handler = logging.StreamHandler()
    # handler.setFormatter(file_formatter)
    # logger.addHandler(handler)

    file_handle_name = "file"
    if file_handle_name in [h.name for h in logger.handlers]:
        return
    if os.path.dirname(filepath) is not '':
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
    file_handle = logging.FileHandler(filename=filepath, mode="a")
    file_handle.set_name(file_handle_name)
    file_handle.setFormatter(file_formatter)
    logger.addHandler(file_handle)
    logger.setLevel(logging.DEBUG)
    return logger


if __name__ == '__main__':
    main()