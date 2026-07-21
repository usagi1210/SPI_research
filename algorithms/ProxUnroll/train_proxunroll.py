import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from model.proxunroll import ProxUnroll
from test_proxunroll import test
import torch.optim as optim
import os
import cv2
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from opts import parse_args
import time
import einops
import random
import datetime
from utils import (
    load_checkpoint,
    checkpoint,
    TrainData,
    Logger,
    create_run_layout,
    format_cr,
    save_per_image_metrics,
)


STAGE_WEIGHTS = [0.01, 0.01, 0.01, 0.01, 0.01, 0.95]


def compute_pt_loss(criterion, outputs, prox_outputs):
    stage_outputs = outputs[-6:]
    loss = sum(
        w * torch.sqrt(criterion(stage_outputs[i], prox_outputs[i]))
        for i, w in enumerate(STAGE_WEIGHTS)
    )
    return loss


def train(
    args,
    network,
    optimizer,
    scheduler,
    logger,
    checkpoint_dir,
    metrics_dir,
    train_vis_dir,
    test_vis_dir=None,
):
    criterion = nn.MSELoss().to(args.device)
    rank = dist.get_rank() if args.distributed else 0
    dataset = TrainData(args.train_data_path, train_sizes=args.train_sizes)
    num_crops = args.num_train_crops

    if args.distributed:
        dist_sampler = DistributedSampler(dataset, shuffle=True, drop_last=True, seed=args.seed)
        train_data_loader = DataLoader(
            dataset=dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, drop_last=True, pin_memory=True,
            sampler=dist_sampler,
        )
    else:
        dist_sampler = None
        train_data_loader = DataLoader(
            dataset=dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers,
        )

    train_crs = [args.train_cr] if args.train_cr is not None else [
        0.01, 0.04, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    ]
    checkpoint_cr_label = format_cr(args.train_cr) if args.train_cr is not None else 'multi'
    best_primary_psnr = float('-inf')
    for epoch in range(args.pretrain_epoch + 1, args.pretrain_epoch + args.epochs + 1):
        if dist_sampler is not None:
            dist_sampler.set_epoch(epoch)
        epoch_loss = 0
        network = network.train()
        start_time = time.time()
        for iteration, data in enumerate(train_data_loader):
            train_cr = train_crs[iteration % len(train_crs)]
            for gt in data:
                b, h, w = gt.shape
                gt = gt.float().to(args.device)
                optimizer.zero_grad()
                outputs, prox_outputs, images = network(gt, train_cr)
                loss = compute_pt_loss(criterion, outputs, prox_outputs)

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                if rank == 0 and (iteration % args.iter_step) == 0:
                    lr = optimizer.param_groups[0]['lr']
                    logger.info(
                        'epoch: {:<3d}, iter: {:<4d}, size: [{}, {}, {}], cr: {:.2f}, loss: {:.4f}, lr: {:.6f}.'.format(
                            epoch, iteration, b, h, w, train_cr, loss.item(), lr,
                        )
                    )

                if rank == 0 and (iteration % args.save_train_image_step) == 0:
                    image_path = os.path.join(
                        train_vis_dir,
                        'epoch_{}_iter_{}_cr_{}_reso_{}_{}.png'.format(epoch, iteration, train_cr, h, w),
                    )
                    result_img = einops.rearrange(images[0].detach(), 'c s h w -> (c h) (s w)')
                    result_img = (result_img.cpu().numpy() * 255).astype(np.float32)
                    cv2.imwrite(image_path, result_img)

        end_time = time.time()
        scheduler.step()

        if rank == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                'epoch: {}, avg. loss: {:.5f}, lr: {:.6f}, time: {:.2f}s.\n'.format(
                    epoch, epoch_loss / (num_crops * (iteration + 1)), lr, end_time - start_time,
                )
            )

        if rank == 0 and (epoch % args.save_model_step) == 0:
            model_out_path = os.path.join(checkpoint_dir, 'epoch_{}.pth'.format(epoch))
            model = network.module if args.distributed else network
            checkpoint(epoch, model, optimizer, model_out_path, scheduler=scheduler)
            checkpoint(
                epoch,
                model,
                optimizer,
                os.path.join(checkpoint_dir, 'latest_cr{}.pth'.format(checkpoint_cr_label)),
                scheduler=scheduler,
            )

        if rank == 0 and args.test_flag and epoch % args.test_every == 0:
            logger.info('epoch: {}, psnr and ssim test results:'.format(epoch))
            model = network.module if args.distributed else network
            eval_colors = [False, True] if args.eval_color else [False]
            for color in eval_colors:
                for test_cr in args.eval_crs:
                    mode = 'color' if color else 'gray'
                    test_path = os.path.join(
                        test_vis_dir,
                        'epoch_{:03d}'.format(epoch),
                        mode,
                        'cr{}'.format(format_cr(test_cr)),
                    )
                    os.makedirs(test_path, exist_ok=True)
                    logger.info('CR: {}.'.format(test_cr))
                    psnr_dict, ssim_dict = test(args, test_cr, color, model, logger, test_path, epoch=epoch)
                    logger.info('psnr: {}.'.format(psnr_dict))
                    logger.info('ssim: {}.'.format(ssim_dict))
                    mean_psnr = float(np.mean(list(psnr_dict.values())))
                    mean_ssim = float(np.mean(list(ssim_dict.values())))
                    logger.info(
                        'Epoch {:03d} | {} | CR {:>2}% | PSNR {:.4f} dB | SSIM {:.6f}'.format(
                            epoch,
                            'CBSD68' if color else 'Set11',
                            format_cr(test_cr),
                            mean_psnr,
                            mean_ssim,
                        )
                    )
                    save_per_image_metrics(
                        psnr_dict,
                        ssim_dict,
                        os.path.join(test_path, 'per_image_metrics.csv'),
                    )

                    if not color and abs(test_cr - args.primary_cr) < 1e-8:
                        primary_psnr = mean_psnr
                        if primary_psnr > best_primary_psnr:
                            best_primary_psnr = primary_psnr
                            checkpoint(
                                epoch,
                                model,
                                optimizer,
                                os.path.join(checkpoint_dir, 'best_cr{}.pth'.format(format_cr(test_cr)),),
                                scheduler=scheduler,
                            )
                            save_per_image_metrics(
                                psnr_dict,
                                ssim_dict,
                                os.path.join(metrics_dir, 'best_per_image_metrics.csv'),
                            )
                            logger.info(
                                'Saved best CR {:.2f} checkpoint (PSNR {:.4f}).'.format(
                                    test_cr, best_primary_psnr,
                                )
                            )


if __name__ == '__main__':
    torch.set_float32_matmul_precision('highest')
    args = parse_args()
    args.pretrain_epoch = 0

    local_rank = 0
    rank = 0
    if args.distributed:
        local_rank = int(os.environ['LOCAL_RANK'])
        args.device = torch.device('cuda', local_rank)
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
    elif not torch.cuda.is_available():
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(args.device)

    run_id = args.run_id or datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = metrics_dir = train_vis_dir = test_vis_dir = log_path = None
    if rank == 0:
        model_name = 'ProxUnroll-{}'.format(args.solver.upper())
        run_cr_label = format_cr(args.train_cr) if args.train_cr is not None else 'multi'
        layout = create_run_layout(args.result_dir, model_name, run_cr_label, run_id=run_id)
        checkpoint_dir = layout['checkpoints']
        metrics_dir = layout['base_dir']
        train_vis_dir = os.path.join(layout['vis'], 'train')
        test_vis_dir = os.path.join(layout['vis'], 'test')
        os.makedirs(train_vis_dir, exist_ok=True)
        if args.test_flag:
            os.makedirs(test_vis_dir, exist_ok=True)
        log_path = os.path.join(layout['logs'], 'train.log')

    logger = Logger(log_path) if rank == 0 else None

    if rank == 0:
        logger.info(
            '\n' + 'Run ID: ' + run_id + '\n'
            + 'Result Directory: {}'.format(layout['base_dir']) + '\n'
            + 'Solver: {}'.format(args.solver) + '\n'
            + 'Network Architecture: {}: {}, {}-{}-{}'.format(
                args.decoder_type, args.dim, args.enc_blocks, args.mid_blocks, args.dec_blocks,
            ) + '\n'
            + 'Batch Size: {}'.format(args.batch_size) + '\n'
            + 'Learning Rate: {:.6f} -> {:.6f} (CosineAnnealingLR, T_max={})'.format(
                args.lr, args.lr_min, args.epochs,
            ) + '\n'
            + 'Train Epochs: {}'.format(args.epochs) + '\n'
            + 'Train Sizes: {} ({} crops/iter)'.format(args.train_sizes, args.num_train_crops) + '\n'
            + 'Train CR: {}'.format(args.train_cr if args.train_cr is not None else 'multi') + '\n'
            + 'Validation: {} (every {} epoch(s), CRs: {})'.format(
                args.test_flag, args.test_every, args.eval_crs,
            ) + '\n'
            + 'Pretrain Model: {}'.format(args.pretrained_model_path)
        )

    seed = random.randint(1, 10000) if rank == 0 else 0
    if args.distributed:
        seed_tensor = torch.tensor(seed, device=args.device)
        dist.broadcast(seed_tensor, src=0)
        seed = int(seed_tensor.item())
    if rank == 0:
        logger.info('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    args.seed = seed

    if torch.cuda.is_available():
        if args.distributed:
            torch.cuda.set_device(local_rank)
        elif args.device.type == 'cuda' and args.device.index is not None:
            torch.cuda.set_device(args.device.index)
        torch.cuda.empty_cache()

    network = ProxUnroll(
        solver=args.solver,
        color_channel=args.color_channel,
        dim=args.dim,
        mid_blocks=args.mid_blocks,
        enc_blocks=args.enc_blocks,
        dec_blocks=args.dec_blocks,
    ).to(args.device)

    if args.torchcompile:
        assert hasattr(torch, 'compile'), 'torch.compile() is required for --torchcompile.'
        network = torch.compile(network, backend=args.torchcompile)

    optimizer = optim.Adam(network.parameters(), lr=args.lr)
    pretrained_dict = None
    if args.pretrained_model_path is not None:
        pretrained_dict = torch.load(args.pretrained_model_path, map_location=args.device)
        args.pretrain_epoch = pretrained_dict.get('pretrain_epoch', 0)
    elif rank == 0:
        logger.info('No pretrained model.')

    last_epoch = args.pretrain_epoch - 1 if args.pretrain_epoch > 0 else -1
    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_min, last_epoch=last_epoch,
    )
    if pretrained_dict is not None:
        load_checkpoint(network, pretrained_dict, logger, optimizer, scheduler)

    if args.distributed:
        pretrain_epoch = torch.tensor(args.pretrain_epoch, device=args.device)
        dist.broadcast(pretrain_epoch, src=0)
        args.pretrain_epoch = int(pretrain_epoch.item())

    if args.distributed:
        network = DDP(network, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    train(
        args,
        network,
        optimizer,
        scheduler,
        logger,
        checkpoint_dir,
        metrics_dir,
        train_vis_dir,
        test_vis_dir,
    )
