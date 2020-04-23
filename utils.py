import sys
import argparse


def parse_args(args):
    """Argument parsing for run_covid_ct.py"""
    parser = argparse.ArgumentParser(description='COVIDNet-CT Train/Val/Infer Script')
    parser.add_argument('-md', '--model_dir', type=str, default='models/COVIDNet-CT', help='Model directory')
    parser.add_argument('-mf', '--meta_file', type=str, default='model.meta', help='Model meta file')
    parser.add_argument('-ck', '--ckpt', type=str, default='model', help='Checkpoint prefix')
    parser.add_argument('-ih', '--input_height', type=int, default=512, help='Image height')
    parser.add_argument('-iw', '--input_width', type=int, default=512, help='Image width')
    if args[0] == 'train':
        parser.add_argument('-od', '--output_dir', type=str, required=True, help='Output directory')
        parser.add_argument('-dd', '--data_dir', type=str, default='data/COVIDx-CT', help='Data directory')
        parser.add_argument('-tf', '--train_split_file', type=str,
                            default='train_COVIDx-CT_v1.txt', help='Train split file')
        parser.add_argument('-vf', '--val_split_file', type=str,
                            default='test_COVIDx-CT_v1.txt', help='Val split file')
        parser.add_argument('-ep', '--epochs', type=int, default=100, help='Training epochs')
        parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Batch size')
        parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Optimizer learning rate')
        parser.add_argument('-mo', '--momentum', type=float, default=0.9, help='Optimizer momentum')
        parser.add_argument('-fc', '--fc_only', action='store_true', help='Flag to freeze feature extractor '
                                                                          'and train only the FC layer')
        parser.add_argument('-li', '--log_interval', type=int, default=100, help='Logging interval in steps')
        parser.add_argument('-vi', '--val_interval', type=int, default=1000,
                            help='Validation interval in steps. Set to 0 to skip validation during training.')
        parser.add_argument('-si', '--save_interval', type=int, default=1000, help='Save interval in steps')
    elif args[0] == 'val':
        parser.add_argument('-dd', '--data_dir', type=str, default='data/COVIDx-CT', help='Data directory')
        parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Batch size')
        parser.add_argument('-vf', '--val_split_file', type=str,
                            default='test_COVIDx-CT_v1.txt', help='Val split file')
        parser.add_argument('-pc', '--plot_confusion', action='store_true', help='Flag to plot confusion matrix')
    elif args[0] == 'infer':
        parser.add_argument('-im', '--image_file', type=str, required=True, help='Image file')
    elif args[0] in ('-h', '--help'):
        print('COVIDNet-CT Train/Val/Infer Script\nUse run_covid_ct.py {train, val, infer} -h '
              'to see help message for each run option')
        sys.exit(0)
    else:
        raise ValueError('Mode must be one of {train, val, infer} or {-h, --help}')

    parsed_args = parser.parse_args(args[1:])
    if args[0] == 'infer':
        parsed_args.data_dir = None


    return args[0], parsed_args
