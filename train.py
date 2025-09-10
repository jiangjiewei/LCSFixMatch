import argparse
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


from dataset.local import DATASET_GETTERS
from utils import AverageMeter, accuracy

logger = logging.getLogger(__name__)
best_acc = 0


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=7./16., last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))
    return LambdaLR(optimizer, _lr_lambda, last_epoch)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DenseNet121_SE(nn.Module):
    def __init__(self, num_classes=4): 
        super(DenseNet121_SE, self).__init__()
        self.densenet = models.densenet121(pretrained=True)
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_ftrs, num_classes)
        self.se = SEBlock(num_ftrs)
        
    def forward(self, x):
        features = self.densenet.features(x)
        features = self.se(features) 
        out = nn.functional.relu(features, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.densenet.classifier(out)
        return out


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='3', type=int,
                        help='id for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=12,
                        help='number of workers')
    parser.add_argument('--dataset', default='custom', type=str,
                        choices=['cifar10', 'cifar100', 'custom'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=1440,
                        help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext'],
                        help='model architecture')
    parser.add_argument('--total-steps', default=10000, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=100, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=16, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=4, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=0.45, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.70, type=float,
                        help='pseudo label threshold (original FixMatch)')
    parser.add_argument('--out', default='result/result_4_0.70_0.20_CBAM',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    
    parser.add_argument('--init-tau-k', default=0.8, type=float)
    parser.add_argument('--alpha', default=0.95, type=float)
    parser.add_argument('--min-acc-gain', default=0.1, type=float)
    parser.add_argument('--acc-fluctuation', default=0.2, type=float)
    parser.add_argument('--min-tau-k', default=0.1, type=float)
    parser.add_argument('--max-tau-k', default=1.0, type=float)

    args = parser.parse_args()
    global best_acc

    def create_model(args):
        model = DenseNet121_SE(num_classes=4)
        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters())/1e6))
        return model

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1
    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}",)
    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)

    if args.dataset == 'custom':
        args.num_classes = 4

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](args, )

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size * args.mu,
        num_workers=args.num_workers)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

  
    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)
    else:
        ema_model = None 

    args.start_epoch = 0
    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(args.resume), "Error: no checkpoint found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume, map_location=args.device)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema and 'ema_state_dict' in checkpoint:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        if 'tau_k' in checkpoint:
            args.init_tau_k = checkpoint['tau_k']

    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(f"  Total train batch size = {args.batch_size * args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler)


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader, model, optimizer, ema_model, scheduler):
    if args.amp:
        from apex import amp
    global best_acc
    test_accs = []
    end = time.time()
    
    tau_k = args.init_tau_k  
    tau_h = 0.5 


    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()  
        entropy_mask_probs = AverageMeter() 
        kl_mask_probs = AverageMeter()  

        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step), disable=args.local_rank not in [-1, 0])
        
        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = next(labeled_iter)
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = next(labeled_iter)
            
            try:
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            unlabeled_batch_size = inputs_u_w.shape[0]

            inputs_x = inputs_x.to(args.device)
            inputs_u_w = inputs_u_w.to(args.device)
            inputs_u_s = inputs_u_s.to(args.device)
            targets_x = targets_x.to(args.device)

            logits_u_w = model(inputs_u_w)
            probs_u_w = torch.softmax(logits_u_w, dim=-1) 
            entropy = -torch.sum(probs_u_w * torch.log(probs_u_w + 1e-10), dim=-1)
            entropy_mask = entropy <= tau_h  
            entropy_mask_probs.update(entropy_mask.float().mean().item())

            filtered_u_w = inputs_u_w[entropy_mask]
            filtered_u_s = inputs_u_s[entropy_mask]
            filtered_probs_u_w = probs_u_w[entropy_mask]

            if filtered_u_w.shape[0] == 0:
                Lu = torch.tensor(0.0, device=args.device)  
            else:
                logits_u_s = model(filtered_u_s)
                probs_u_s = torch.softmax(logits_u_s, dim=-1)
                
               
                kl_div = torch.sum(
                    filtered_probs_u_w * torch.log((filtered_probs_u_w + 1e-10) / (probs_u_s + 1e-10)), 
                    dim=-1
                )
                kl_mask = kl_div <= tau_k  
                kl_mask_probs.update(kl_mask.float().mean().item())

               
                final_u_s = filtered_u_s[kl_mask]
                final_probs_u_w = filtered_probs_u_w[kl_mask]

                if final_u_s.shape[0] == 0:
                    Lu = torch.tensor(0.0, device=args.device)
                else:
                    _, targets_u = torch.max(final_probs_u_w, dim=-1)
                    Lu = F.cross_entropy(model(final_u_s), targets_u, reduction='mean')

            logits_x = model(inputs_x)
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            loss = Lx + args.lambda_u * Lu

            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())

            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model) 
            model.zero_grad()  

            total_mask_ratio = (entropy_mask.float().mean() * kl_mask.float().mean()).item() if unlabeled_batch_size > 0 else 0.0
            mask_probs.update(total_mask_ratio)

            batch_time.update(time.time() - end)
            end = time.time()

            if not args.no_progress:
                p_bar.set_description(
                    f"Train Epoch: {epoch+1}/{args.epochs}. Iter: {batch_idx+1}/{args.eval_step}. "
                    f"LR: {scheduler.get_last_lr()[0]:.4f}. Data: {data_time.avg:.3f}s. Batch: {batch_time.avg:.3f}s. "
                    f"Loss: {losses.avg:.4f}. Loss_x: {losses_x.avg:.4f}. Loss_u: {losses_u.avg:.4f}. "
                    f"Entropy Mask: {entropy_mask_probs.avg:.2f}. KL Mask: {kl_mask_probs.avg:.2f}. "
                    f"Total Mask: {mask_probs.avg:.2f}. Tau_k: {tau_k:.4f}"
                )
                p_bar.update()
        if not args.no_progress:
            p_bar.close()

        if args.local_rank in [-1, 0]:
            test_model = ema_model.ema if args.use_ema else model
            test_loss, test_acc = test(args, test_loader, test_model, epoch)

            if epoch > 0:
                prev_acc = test_accs[-1] if test_accs else 0.0
                if test_acc - prev_acc < args.min_acc_gain:
                    tau_k *= args.alpha  
            
                elif test_acc < prev_acc - args.acc_fluctuation:
                    tau_k /= args.alpha  
                tau_k = max(args.min_tau_k, min(args.max_tau_k, tau_k))

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(ema_model.ema, "module") else ema_model.ema

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'tau_k': tau_k,  
            }, is_best, args.out)

            test_accs.append(test_acc)
            logger.info(f'Best top-1 acc: {best_acc:.2f}')
            logger.info(f'Mean top-1 acc (last 20): {np.mean(test_accs[-20:]):.2f}\n')

    if args.local_rank in [-1, 0]:
        args.writer.close()


def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader, disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        model.eval()
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)

            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 2))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])

            batch_time.update(time.time() - end)
            end = time.time()

            if not args.no_progress:
                test_loader.set_description(
                    f"Test Iter: {batch_idx+1}/{len(test_loader)}. Data: {data_time.avg:.3f}s. "
                    f"Batch: {batch_time.avg:.3f}s. Loss: {losses.avg:.4f}. "
                    f"top1: {top1.avg:.2f}. top5: {top5.avg:.2f}."
                )
        if not args.no_progress:
            test_loader.close()

    logger.info(f"Epoch {epoch+1} Test: top-1 acc: {top1.avg:.2f}, top-5 acc: {top5.avg:.2f}")
    return losses.avg, top1.avg


if __name__ == '__main__':
    main()
