import configargparse


def parser_args():
    parser = configargparse.ArgParser(description='PyTorch Name2Gender Training')
    # dataset
    parser.add_argument('--num_workers', default=2, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--epoch', default=500, type=int, 
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', '-b', default=128, type=int, 
                        help='Batch size for training')
    # network setting
    parser.add_argument('--arch', '-a', metavar='ARCH', default='LSTM',
                        help='network architecture')
    parser.add_argument('--dropout', '-d', default=0.5, type=float, 
                        help='dropout rate')
    parser.add_argument('--n-hidden', default=128, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--learning-rate', '-lr', default=0.005, type=float, 
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='momentum')
    parser.add_argument('--weight-decay', '-wd', default=5e-4, type=float, 
                        help='Weight decay for SGD')
    parser.add_argument('--disable-he-initialization', action='store_true',
                        help='use He initialization for the model, ebable it in default')
    # others
    parser.add_argument('--logging-freq', default=50, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--saving-freq', default=50, type=int,
                        help='saving frequency (default: 10)')

    return parser.parse_args()