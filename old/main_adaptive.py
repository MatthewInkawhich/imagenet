import argparse
import os
import random
import shutil
import time
import warnings
import yaml
import copy
import itertools
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from models import strider

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
#parser.add_argument('data', metavar='DIR',
#                    help='path to dataset')
parser.add_argument('config', metavar='FILE',
                    help='path to config file')
#parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
#                    choices=model_names,
#                    help='model architecture: ' +
#                        ' | '.join(model_names) +
#                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
# New adaptive configs
parser.add_argument('--start-cycle', default=0, type=int, metavar='N',
                    help='manual cycle number (useful on restarts)')
parser.add_argument('--cycles', default=1, type=int, metavar='N',
                    help='number of s1 s2 cycles')
parser.add_argument('--stage1-epochs-per-cycle', default=90, type=int, metavar='N',
                    help='number of stage1 epochs per cycle')
parser.add_argument('--stage2-epochs-per-cycle', default=45, type=int, metavar='N',
                    help='number of stage2 epochs per cycle')
parser.add_argument('--eps-start', default=1.0, type=float, metavar='N',
                    help='starting epsilon value')
parser.add_argument('--eps-end', default=0.05, type=float, metavar='N',
                    help='ending epsilon value')
parser.add_argument('--eps-decay-factor', default=0.75, type=float, metavar='N',
                    help='percentage of total epochs until epsilon decays to eps-end')
parser.add_argument('--stage1-loss-queue-length', default=50, type=int, metavar='N',
                    help='length of the loss queue that is used to compute curr_avg_loss from stage1')
parser.add_argument('--evaluate-freq', default=10, type=int, metavar='N',
                    help='frequency to evaluate and save (epochs)')
parser.add_argument('--resume-stage2', default='', type=str, metavar='PATH',
                    help='path to latest stage1 checkpoint (default: none)')
parser.add_argument('--lr1', default=0.1, type=float,
                    metavar='LR', help='initial learning rate for stage 1')
parser.add_argument('--lr2', default=0.001, type=float,
                    metavar='LR', help='initial learning rate for stage 2')


parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--random-eval', action='store_true',
                    help='Use random strides when validating')
parser.add_argument('--batch-eval', action='store_true',
                    help='Use batched evaluation')
parser.add_argument('--oracle-eval', action='store_true',
                    help='Evaluate using all possible strides and choosing the best')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0


def main():
    args = parser.parse_args()

    # Load config file
    if not os.path.isfile(args.config):
        print("Error: Config file: {} does not exist".format(args.config))
        exit()
    with open(args.config, "r") as yamlfile:
        args.model_cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
    args.outdir = './out/' + args.config.split('configs/')[-1].split('.')[0]

    print("args.config:", args.config)
    print("args.outdir:", args.outdir)
    print("args.model_cfg:", args.model_cfg)


    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    # create model
    model_name = args.model_cfg['MODEL']
    if model_name in model_names:
        print("Using model from torchvision.models: {}".format(model_name))
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(model_name))
            model = models.__dict__[model_name](pretrained=True)
        else:
            print("=> creating model '{}'".format(model_name))
            model = models.__dict__[model_name]()
    elif model_name == 'strider':
        print("Using custom model: {}".format(model_name))
        model = strider.StriderClassifier(args.model_cfg['STRIDER'])
    else:
        print("Error: Model: {} not recognized!".format(model_name))
        exit()


    print(model)
    for n, p in model.named_parameters():
        print(n)
    print("params:", sum(p.numel() for p in model.parameters()))
    #exit()

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if model_name.startswith('alexnet') or model_name.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()


    # Define loss functions (criterions) for both training stages and optimizer
    # First, separate stage 1 and stage 2 params for optimizers
    print("args.lr1:", args.lr1)
    print("args.lr2:", args.lr2)
    params1 = []
    params2 = []
    for n, p in model.named_parameters():
        if '.ss.' in n:
            params2.append(p)
        else:
            params1.append(p)

    criterion1 = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion2 = nn.SmoothL1Loss().cuda(args.gpu)
    optimizer1 = torch.optim.SGD(params1, args.lr1,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    #optimizer2 = torch.optim.SGD(params2, args.lr2,
    #                            momentum=args.momentum,
    #                            weight_decay=args.weight_decay)
    optimizer2 = torch.optim.Adam(params2, args.lr2)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_cycle = checkpoint['cycle'] - 1
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer1.load_state_dict(checkpoint['optimizer1'])
            optimizer2.load_state_dict(checkpoint['optimizer2'])
            stage1_avg_loss = checkpoint['stage1_avg_loss']
            # Load current learning rates
            #args.lr1 = get_lr(optimizer1)
            #args.lr2 = get_lr(optimizer2)
            print("=> loaded checkpoint '{}' (cycle {})"
                  .format(args.resume, checkpoint['cycle']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.resume_stage2:
        if os.path.isfile(args.resume_stage2):
            print("=> loading checkpoint '{}'".format(args.resume_stage2))
            checkpoint = torch.load(args.resume_stage2)
            args.start_cycle = checkpoint['cycle'] - 1
            best_acc1 = checkpoint['best_acc1']
            stage1_avg_loss = checkpoint['stage1_avg_loss']
            # Only load non-SS layers
            non_ss_dict = {k: v for k, v in checkpoint['state_dict'].items() if '.ss.' not in k}
            model.load_state_dict(non_ss_dict, strict=False)
            # Load optimizer
            optimizer1.load_state_dict(checkpoint['optimizer1'])
            # Load current learning rate
            args.lr1 = get_lr(optimizer1)
            print("=> loaded checkpoint '{}' (cycle {})"
                  .format(args.resume_stage2, args.start_cycle))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume_stage2))
        

    # Enable benchmark mode
    cudnn.benchmark = True

    device_count = torch.cuda.device_count()
    print("device_count:", device_count)

    # Data loading code
    traindir = '/zero1/data1/ILSVRC2012/train/original'
    #traindir = './data/ILSVRC2012_val' 
    valdir = './data/ILSVRC2012_val' 

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    # Temp**
    #train_dataset = torch.utils.data.random_split(train_dataset, [20, len(train_dataset)-20])[0]
    #val_dataset = torch.utils.data.random_split(val_dataset, [20, len(val_dataset)-20])[0]
    #print("len(train_dataset):", len(train_dataset))
    #print("len(val_dataset):", len(val_dataset))
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    if args.batch_eval:
        val_batch_size = args.batch_size
    else:
        val_batch_size = device_count
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size, shuffle=False,  # batch size for each device should be 1
        num_workers=args.workers, pin_memory=True, drop_last=True)

    if args.evaluate:
        print("EVALUATE flag set, evaluating model now...")
        if args.oracle_eval:
            validate_oracle(val_loader, model, criterion1, args)
        else:
            validate(val_loader, model, criterion1, args, random=args.random_eval)
        return

    # Create epsilon greedy decay schedule
    stage1_total_epochs = args.cycles * args.stage1_epochs_per_cycle
    stage2_total_epochs = args.cycles * args.stage2_epochs_per_cycle
    if args.model_cfg['STRIDER']['RANDOM_STRIDE']:
        eps_schedule1 = torch.full([stage1_total_epochs], 100.0)
        eps_schedule2 = torch.full([stage2_total_epochs], 100.0)
    else:
        # Set epsilons for stage1
        eps_linspace = torch.linspace(args.eps_start, args.eps_end, int(stage1_total_epochs * args.eps_decay_factor))
        eps_schedule1 = torch.full([stage1_total_epochs], args.eps_end)
        eps_schedule1[:eps_linspace.shape[0]] = eps_linspace
        # Set epsilons for stage2
        eps_linspace = torch.linspace(args.eps_start, args.eps_end, int(stage2_total_epochs * args.eps_decay_factor))
        eps_schedule2 = torch.full([stage2_total_epochs], args.eps_end)
        eps_schedule2[:eps_linspace.shape[0]] = eps_linspace
    print("eps_schedule1:", eps_schedule1, eps_schedule1.shape)
    print("eps_schedule2:", eps_schedule2, eps_schedule2.shape)



    ##############
    # EPOCH LOOP
    ##############
    for c in range(args.start_cycle, args.cycles):
        print("Starting cycle:", c)
        for epoch in range(args.stage1_epochs_per_cycle):
            # Compute total stage1 epoch
            total_epoch = epoch + (c * args.stage1_epochs_per_cycle)
            # Update learning rate
            adjust_learning_rate(optimizer1, 1, total_epoch, args)
            # Update current epoch's epsilon
            epsilon = eps_schedule1[total_epoch].item()
            # Train STAGE1 for one epoch
            stage1_avg_loss = train_stage1(train_loader, model, criterion1, optimizer1, epoch, total_epoch, epsilon, args)
            # Evaluate if necessary
            if total_epoch != 0 and total_epoch % args.evaluate_freq == 0:
                evaluate_and_save(val_loader, model, optimizer1, optimizer2, criterion1, c, 1, total_epoch, model_name, stage1_avg_loss, args)

        if args.stage1_epochs_per_cycle > 0:
            print("Starting Post-Stage1 Eval for Cycle:", c)
            evaluate_and_save(val_loader, model, optimizer1, optimizer2, criterion1, c, 1, total_epoch, model_name, stage1_avg_loss, args, filename='C{}_post_S1.pth.tar'.format(c))


        for epoch in range(args.stage2_epochs_per_cycle):
            # Compute total stage2 epoch
            total_epoch = epoch + (c * args.stage2_epochs_per_cycle)
            # Update learning rate
            adjust_learning_rate(optimizer2, 2, total_epoch, args)
            # Update current epoch's epsilon
            epsilon = eps_schedule2[total_epoch].item()
            # Train STAGE2 for one epoch
            train_stage2(train_loader, model, criterion2, optimizer2, epoch, total_epoch, epsilon, stage1_avg_loss, args)
            # Evaluate if necessary
            if total_epoch != 0 and total_epoch % args.evaluate_freq == 0:
                evaluate_and_save(val_loader, model, optimizer1, optimizer2, criterion1, c, 2, total_epoch, model_name, stage1_avg_loss, args)

        if args.stage2_epochs_per_cycle > 0:
            print("Starting Post-Stage2 Eval for Cycle:", c)
            evaluate_and_save(val_loader, model, optimizer1, optimizer2, criterion1, c, 2, total_epoch, model_name, stage1_avg_loss, args, filename='C{}_post_S2.pth.tar'.format(c))


# One full epoch of training on task (stage 1)
# Returns average training task loss from last args.stage1_loss_queue_length
def train_stage1(train_loader, model, criterion, optimizer, epoch, total_epoch, epsilon, args):
    print("Starting Stage 1, Epoch:", epoch)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}] [{}] S1".format(epoch, total_epoch))

    # Initialize counts tensor
    num_ss_blocks = sum([1 if x[0] == 1 else 0 for x in args.model_cfg['STRIDER']['BODY_CONFIG']])
    num_stride_options = 7  # Hardcode this for now
    ss_choice_counts = torch.zeros(num_ss_blocks, num_stride_options)

    # switch to train mode
    model.train()
    end = time.time()

    # Create loss queue
    loss_queue = []

    # Iterate over training images
    for i, (images, target) in enumerate(train_loader):
        #old_sd = copy.deepcopy(model.state_dict())
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output, preds, choices = model(images, epsilon, 1, device='cuda')
        loss = criterion(output, target)

        # Track choice counts per SS block
        for ss_id in range(choices.shape[1]):
            for j in range(choices.shape[0]):
                ss_choice_counts[ss_id][choices[j][ss_id]] += 1

        #print("output:", output.shape)
        #print("preds:", preds, preds.shape)
        #print("choices:", choices, choices.shape)
        #print("loss:", loss, loss.shape)
        #exit()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # Add loss to loss queue
        loss_queue.append(loss.item())
        if len(loss_queue) > args.stage1_loss_queue_length:
            loss_queue.pop(0)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        #for n, p in model.named_parameters():
        #    if p.grad is None or p.grad.abs().sum() == 0:
        #        print(n, "NO/ZERO GRAD")
        #    else:
        #        print(n, p.grad.shape)
        #exit()
        optimizer.step()


        #new_sd = model.state_dict()
        #parameter_compare(old_sd, new_sd)
        #exit()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


    # Print SS choice counts
    print("SS Choice Counts:")
    for i in range(ss_choice_counts.shape[0]):
        print("SS Block:", i, ss_choice_counts[i])

    # After all iterations, compute average of last N losses
    loss_queue_avg = sum(loss_queue) / len(loss_queue)
    return loss_queue_avg



# One full epoch of training SS modules (stage 2)
def train_stage2(train_loader, model, criterion, optimizer, epoch, total_epoch, epsilon, stage1_avg_loss, args):
    print("Starting Stage 2, Epoch:", epoch)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}] [{}] S2".format(epoch, total_epoch))

    # Initialize counts tensor
    num_ss_blocks = sum([1 if x[0] == 1 else 0 for x in args.model_cfg['STRIDER']['BODY_CONFIG']])
    num_stride_options = 7  # Hardcode this for now
    ss_choice_counts = torch.zeros(num_ss_blocks, num_stride_options)

    # switch to train mode
    model.train()
    #model.eval()
    end = time.time()

    # Iterate over training samples
    for i, (images, target) in enumerate(train_loader):
        #old_sd = copy.deepcopy(model.state_dict())
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # Forward batch thru model
        output, ss_output, choices = model(images, epsilon, 2, device='cuda')

        # Generate SS target
        with torch.no_grad():
            # Compute task loss
            task_loss = torch.nn.functional.cross_entropy(output, target, reduction='none')
            # Compute ss_target
            ss_target = (task_loss.detach() - stage1_avg_loss) / stage1_avg_loss
            ss_target = torch.repeat_interleave(ss_target.unsqueeze(1), ss_output.shape[1], dim=1)
            # Note that the ss_output and ss_target should be of shape: [N, #adaptive blocks]

        # Compute SS loss
        ss_loss = criterion(ss_output, ss_target)

        # Track choice counts per SS block
        for ss_id in range(choices.shape[1]):
            for j in range(choices.shape[0]):
                ss_choice_counts[ss_id][choices[j][ss_id]] += 1

        #print("\noutput:", output.shape)
        #print("task loss:", task_loss, task_loss.shape)
        #print("choices:", choices, choices.shape)
        #print("ss_output:", ss_output, ss_output.shape, ss_output.dtype)
        #print("ss_target:", ss_target, ss_target.shape, ss_target.dtype)
        #print("ss_loss:", ss_loss, ss_loss.shape)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(ss_loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        ss_loss.backward()
        #for n, p in model.named_parameters():
        #    if p.grad is None or p.grad.abs().sum() == 0:
        #        print(n, "NO/ZERO GRAD")
        #    else:
        #        print(n, p.grad.shape)
        #exit()
        optimizer.step()

        #new_sd = model.state_dict()
        #parameter_compare(old_sd, new_sd)
        #print("AFTER:", new_sd['module.body.block15.conv3.weight'])
        #exit()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    # Print SS choice counts
    print("SS Choice Counts:")
    for i in range(ss_choice_counts.shape[0]):
        print("SS Block:", i, ss_choice_counts[i])



def validate(val_loader, model, criterion, args, random=False):
    print("Running evaluation...")
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # Initialize counts tensor
    num_ss_blocks = sum([1 if x[0] == 1 else 0 for x in args.model_cfg['STRIDER']['BODY_CONFIG']])
    num_stride_options = 7  # Hardcode this for now
    ss_choice_counts = torch.zeros(num_ss_blocks, num_stride_options)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            # if random=True set epsilon HIGH
            if random:
                output, preds, choices = model(images, 100, 2, device='cuda')
            # Else set epsilon to -1 so it is pure greedy selection and use stage 2 forward (doesn't matter)
            else:
                output, preds, choices = model(images, -1, 2, device='cuda')

            
            # OPTIONAL: Show image
            #print("choice:", choices.item())
            #if choices.item() == 6:
            #    im2show = images[0].permute(1, 2, 0)
            #    show_image(im2show)
            #    exit()

            loss = criterion(output, target)

            # Track choice counts per SS block
            for ss_id in range(choices.shape[1]):
                for j in range(choices.shape[0]):
                    ss_choice_counts[ss_id][choices[j][ss_id]] += 1

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        # Print SS choice counts
        print("SS Choice Counts:")
        for i in range(ss_choice_counts.shape[0]):
            print("SS Block:", i, ss_choice_counts[i])

    return top1.avg



def validate_oracle(val_loader, model, criterion, args):
    print("Running evaluation...")
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    body_config = args.model_cfg['STRIDER']['BODY_CONFIG']
    downsample_bounds = args.model_cfg['STRIDER']['DOWNSAMPLE_BOUNDS']
    # Initialize counts tensor
    num_ss_blocks = sum([1 if x[0] == 1 else 0 for x in body_config])
    num_stride_options = 7  # Hardcode this for now
    stride_options = [[1, 1], [1, 2], [2, 1], [2, 2], [1, .5], [.5, 1], [.5, .5]]
    ss_choice_counts = torch.zeros(num_ss_blocks, num_stride_options)

    # Create list of all possible stride options
    option_list = list(range(num_stride_options))
    all_combos = list(itertools.product(option_list, repeat=num_ss_blocks))
    good_combos = []

    # Trim stride options that are invalid due to bounds
    for i in range(len(all_combos)):
        good = True
        curr_downsample = [4, 4]  # [dH, dW] Stem downsamples H and W by 4x
        adaptive_idx = 0
        # Iterate over network configs to check downsample rate
        for layer_idx in range(len(body_config)):
            # If the curr layer is adaptive
            if body_config[layer_idx][0] == 1:
                stride = stride_options[all_combos[i][adaptive_idx]]
                curr_downsample = [s1*s2 for s1, s2 in zip(curr_downsample, stride)]
                adaptive_idx += 1    
            # If the curr layer is NOT adaptive
            else:
                stride_side = body_config[layer_idx][1][0]
                stride = [stride_side, stride_side]
                curr_downsample = [s1*s2 for s1, s2 in zip(curr_downsample, stride)]
            # Check if curr_downsample is now out of bounds
            curr_bounds = downsample_bounds[layer_idx]
            if curr_downsample[0] > curr_bounds[0] or curr_downsample[1] > curr_bounds[0] or curr_downsample[0] < curr_bounds[1] or curr_downsample[1] < curr_bounds[1]:
                good = False
                break   # Out of bounds, do NOT consider this stride combo
        if good:    
            good_combos.append(all_combos[i])

    print("good_combos:", good_combos)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            loss_summary = []
            output_summary = []
            # Iterate over all stride combinations
            for stride_combo in good_combos:
                # compute output
                output, preds, choices = model(images, -1, 2, device='cuda', manual_stride=stride_combo)
                loss = torch.nn.functional.cross_entropy(output, target, reduction='none')
                loss_summary.append(loss.unsqueeze(1))
                output_summary.append(output)

            # Find indices for stride options that lead to lowest loss for each sample in batch
            loss_summary = torch.cat(loss_summary, dim=1)
            min_loss_vals, min_loss_inds = torch.min(loss_summary, dim=1)
            # Compute average loss of best stride options
            loss = torch.mean(min_loss_vals)
            # Organize output tensor such that the output for each sample is from the optimal stride option
            output = []
            for b in range(target.shape[0]):
                best_stride_option = min_loss_inds[b]
                output.append(output_summary[best_stride_option][b].unsqueeze(0))
            output = torch.cat(output, dim=0)

            # Record stride selections
            for combo_ind in min_loss_inds:
                for ss_id in range(len(all_combos[combo_ind])):
                    ss_choice_counts[ss_id][all_combos[combo_ind][ss_id]] += 1

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        # Print SS choice counts
        print("SS Choice Counts:")
        for i in range(ss_choice_counts.shape[0]):
            print("SS Block:", i, ss_choice_counts[i])

    return top1.avg

def evaluate_and_save(val_loader, model, optimizer1, optimizer2, criterion1, cycle, stage, total_epoch, model_name, stage1_avg_loss, args, filename='checkpoint.pth.tar'):
    global best_acc1
    # Evaluate on validation set
    acc1 = validate(val_loader, model, criterion1, args)

    # remember best acc@1 and save checkpoint
    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            and args.rank % ngpus_per_node == 0):
        save_checkpoint({
            'cycle': cycle + 1,
            'stage': stage,
            'total_epoch': total_epoch + 1,
            'arch': model_name,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer1' : optimizer1.state_dict(),
            'optimizer2' : optimizer2.state_dict(),
            'stage1_avg_loss' : stage1_avg_loss,
        }, is_best, outdir=args.outdir, filename=filename)



def save_checkpoint(state, is_best, outdir='./out/test', filename='checkpoint.pth.tar'):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    filepath = os.path.join(outdir, filename)
    print("Saving checkpoint to: {} ".format(filepath))
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(outdir, 'model_best.pth.tar'))



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def get_lr(optimizer):
    """Returns current learning rate of an optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def adjust_learning_rate(optimizer, stage, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if stage == 1:
        lr = args.lr1 * (0.1 ** (epoch // 30))
    else:
        lr = args.lr2 * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def parameter_compare(old_sd, new_sd):
    for k, _ in old_sd.items():
        if torch.equal(old_sd[k], new_sd[k]):
            print(k, "same")
        else:
            print(k, "DIFFERENT!")


def show_image(x):
    plt.imshow(x)
    plt.show()



if __name__ == '__main__':
    main()
