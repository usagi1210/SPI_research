import argparse


DEFAULT_EVAL_CRS = [0.01, 0.04, 0.10, 0.25, 0.50]


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {'true', '1', 'yes', 'y'}:
        return True
    if normalized in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError('Expected a boolean value.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--pretrained_model_path', default=None, type=str)
    parser.add_argument('--test_model_path', default=None, type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument(
        '--solver',
        default='hqs',
        choices=['hqs', 'admm'],
        help='Proximal unrolling solver: hqs (HQS) or admm (ADMM)',
    )
    parser.add_argument('--color', default=False, type=str2bool)
    parser.add_argument('--lr', default=1e-3, type=float, help='Peak learning rate (cosine start)')
    parser.add_argument('--lr_min', default=1e-4, type=float, help='Minimum learning rate (cosine end)')
    parser.add_argument('--color_channel', default=1, type=int)
    parser.add_argument('--dim', default=48, type=int)
    parser.add_argument('--mid_blocks', default=2)
    parser.add_argument('--enc_blocks', default=[2, 2, 2])
    parser.add_argument('--dec_blocks', default=[2, 2, 2])
    parser.add_argument('--save_model_step', default=1, type=int)
    parser.add_argument('--save_train_image_step', default=2000, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--iter_step', default=2000, type=int)
    parser.add_argument('--test_flag', default=False, type=str2bool)
    parser.add_argument(
        '--train_cr',
        default=None,
        type=float,
        help='Fixed training CR. Omit to retain the original multi-CR training schedule.',
    )
    parser.add_argument(
        '--eval_crs',
        nargs='+',
        type=float,
        default=None,
        help='CRs evaluated during training. Defaults to train_cr for fixed-CR training.',
    )
    parser.add_argument(
        '--test_crs',
        nargs='+',
        type=float,
        default=None,
        help='CRs evaluated by test_proxunroll.py.',
    )
    parser.add_argument(
        '--test_every',
        default=1,
        type=int,
        help='Run validation once every N epochs when --test_flag true.',
    )
    parser.add_argument(
        '--eval_color',
        default=False,
        type=str2bool,
        help='Also evaluate CBSD68 during training. Gray Set11 evaluation always runs.',
    )
    parser.add_argument(
        '--test_color',
        default=False,
        type=str2bool,
        help='Also evaluate CBSD68 in test_proxunroll.py.',
    )
    parser.add_argument(
        '--train_sizes',
        default='256_321_512',
        choices=['256_321', '256_321_512'],
        help='Training resolutions: 256_321 (256x256 + 321x481) or 256_321_512 (+ 512x512)',
    )
    parser.add_argument('--train_data_path', type=str, default='/home/wangping/datasets/BSDS400')
    parser.add_argument('--test_data_path', type=str, default='/home/wangping/datasets/Set11')
    parser.add_argument('--test_color_data_path', type=str, default='/home/wangping/datasets/CBSD68')
    parser.add_argument('--distributed', default=False, type=str2bool)
    parser.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor')
    parser.add_argument(
        '--result_dir',
        default='../../results/ProxUnroll',
        type=str,
        help='Root directory for standardized run outputs.',
    )
    parser.add_argument(
        '--run_id',
        default=None,
        type=str,
        help='Optional run timestamp/identifier. Defaults to YYYYMMDD_HHMMSS.',
    )
    parser.add_argument(
        '--primary_cr',
        default=None,
        type=float,
        help='Validation CR used to choose best_crXX.pth during training.',
    )

    args = parser.parse_args()
    if args.train_cr is not None and not 0.0 < args.train_cr <= 1.0:
        parser.error('--train_cr must be in (0, 1].')
    if args.test_every < 1:
        parser.error('--test_every must be at least 1.')

    if args.eval_crs is None:
        args.eval_crs = [args.train_cr] if args.train_cr is not None else list(DEFAULT_EVAL_CRS)
    if args.test_crs is None:
        args.test_crs = list(DEFAULT_EVAL_CRS)
    if args.primary_cr is None:
        args.primary_cr = args.train_cr if args.train_cr is not None else 0.25
    if args.primary_cr not in args.eval_crs:
        parser.error('--primary_cr must be included in --eval_crs.')

    args.decoder_type = '{}_proxunroll'.format(args.solver)
    args.num_train_crops = 2 if args.train_sizes == '256_321' else 3
    return args
