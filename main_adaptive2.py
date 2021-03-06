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
import torch.nn.functional as F
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

from models import strider2

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('config', metavar='FILE',
                    help='path to config file')
parser.add_argument('--run-name', default='', type=str,
                    help='(optional) name of run within same config')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
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
parser.add_argument('--evaluate-freq', default=10, type=int, metavar='N',
                    help='frequency to evaluate and save (epochs)')
parser.add_argument('--resume-stage2', default='', type=str, metavar='PATH',
                    help='path to latest stage1 checkpoint (default: none)')
parser.add_argument('--resume-old', default='', type=str, metavar='PATH',
                    help='path to a checkpoint from strider1 model (default: none)')
parser.add_argument('--load-selector-truth', default='', type=str, metavar='PATH',
                    help='path to selector_truth file (default: none)')
parser.add_argument('--lr1', default=0.1, type=float,
                    metavar='LR', help='initial learning rate for stage 1')
parser.add_argument('--lr2', default=0.001, type=float,
                    metavar='LR', help='initial learning rate for stage 2')
parser.add_argument('--lr1-decay-every', default=30, type=int, metavar='N',
                    help='decay lr1 every _ epochs')
parser.add_argument('--lr2-decay-every', default=5, type=int, metavar='N',
                    help='decay lr2 every _ epochs')
parser.add_argument('--eta', default=1.0, type=float, metavar='N',
                    help='eta parameter used for calculating S2 cross entropy weights')
parser.add_argument('--gamma', default=2.0, type=float, metavar='N',
                    help='gamma parameter used in focal loss')


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
    if args.run_name:
        args.outdir += '/{}'.format(args.run_name)

    print("args.config:", args.config)
    print("args.run_name:", args.run_name)
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


    ############################################################################
    ### INITIALIZE MODEL
    ############################################################################
    # Collect valid stride option combos: Many combos are invalid as they take
    # the feature map out of downsample_bounds
    valid_combos = get_valid_stride_combos(args)
    # TEMP: for static model
    #valid_combos = [(3, 0, 0, 0)]
    print("valid_combos:", valid_combos, len(valid_combos))

    model_name = args.model_cfg['MODEL']
    if model_name == 'strider':
        print("Using custom model: {}".format(model_name))
        model = strider2.StriderClassifier(args.model_cfg['STRIDER'])
        selector = strider2.StrideSelectorModule(len(valid_combos))
    else:
        print("Error: Model: {} not recognized!".format(model_name))
        exit()

    print("\n\nmodel:", model)
    for n, p in model.named_parameters():
        print(n)
    model_param_count = sum(p.numel() for p in model.parameters())

    print("\n\nselector:", selector)
    for n, p in selector.named_parameters():
        print(n)
    selector_param_count = sum(p.numel() for p in selector.parameters())

    print("\nmodel:", model_param_count)
    print("selector:", selector_param_count)

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
            selector = torch.nn.DataParallel(selector).cuda()


    ############################################################################
    ### INITIALIZE CRITERIONS AND OPTIMIZERS
    ############################################################################
    # Define loss functions (criterions) for both training stages and optimizer
    # First, separate stage 1 and stage 2 params for optimizers
    print("args.lr1:", args.lr1)
    print("args.lr2:", args.lr2)
    criterion1 = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer1 = torch.optim.SGD(model.parameters(), args.lr1,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    optimizer2 = torch.optim.Adam(selector.parameters(), args.lr2)


    ############################################################################
    ### OPTIONALLY LOAD FROM CHECKPOINT
    ############################################################################
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
            model.load_state_dict(checkpoint['model_sd'])
            selector.load_state_dict(checkpoint['selector_sd'])
            # Load optimizer
            #optimizer1.load_state_dict(checkpoint['optimizer1'])
            #optimizer2.load_state_dict(checkpoint['optimizer2'])
            # Load current learning rates
            #args.lr1 = get_lr(optimizer1)
            #args.lr2 = get_lr(optimizer2)
            print("=> loaded checkpoint '{}' (cycle {})"
                  .format(args.resume, checkpoint['cycle']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            exit()

    if args.resume_stage2:
        if os.path.isfile(args.resume_stage2):
            print("=> loading checkpoint '{}'".format(args.resume_stage2))
            checkpoint = torch.load(args.resume_stage2)
            args.start_cycle = checkpoint['cycle'] - 1
            best_acc1 = checkpoint['best_acc1']
            # Only load model (not selector)
            model.load_state_dict(checkpoint['model_sd'])
            # Load optimizer
            #optimizer1.load_state_dict(checkpoint['optimizer1'])
            # Load current learning rate
            #args.lr1 = get_lr(optimizer1)
            print("=> loaded checkpoint '{}' (cycle {})"
                  .format(args.resume_stage2, args.start_cycle))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume_stage2))
            exit()

    if args.resume_old:
        if os.path.isfile(args.resume_old):
            print("=> loading checkpoint '{}'".format(args.resume_old))
            checkpoint = torch.load(args.resume_old)
            args.start_cycle = checkpoint['cycle'] - 1
            best_acc1 = checkpoint['best_acc1']
            # Only load non-SS layers
            non_ss_dict = {k: v for k, v in checkpoint['state_dict'].items() if '.ss.' not in k}
            model.load_state_dict(non_ss_dict, strict=False)
            # Load optimizer
            #optimizer1.load_state_dict(checkpoint['optimizer1'])
            # Load current learning rate
            #args.lr1 = get_lr(optimizer1)
            print("=> loaded checkpoint '{}' (cycle {})"
                  .format(args.resume_old, args.start_cycle))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume_stage2))
            exit()
        
    # Optionally load the selector_truth for the first stage2 train
    selector_truth = []
    if args.load_selector_truth:
        if os.path.isfile(args.load_selector_truth):
            print("=> loading selector truth lookup '{}'".format(args.load_selector_truth))
            selector_truth = torch.load(args.load_selector_truth)
        else:
            print("=> no selector_truth found at '{}'".format(args.load_selector_truth))
            exit()


    # Enable benchmark mode
    cudnn.benchmark = True

    # Obtain device count
    device_count = torch.cuda.device_count()
    print("device_count:", device_count)


    ############################################################################
    ### BUILD DATASETS AND DATALOADERS
    ############################################################################
    traindir = '/zero1/data1/ILSVRC2012/train/original'
    #traindir = './data/ILSVRC2012_val' 
    valdir = './data/ILSVRC2012_val' 

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    ImageFolderWithIndices = add_indices_to_dataset(datasets.ImageFolder)

    train_dataset_s1 = ImageFolderWithIndices(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_dataset_s2 = ImageFolderWithIndices(
        traindir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = ImageFolderWithIndices(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    # Temp**
    #subset_size = 1000
    #train_dataset_s1 = torch.utils.data.random_split(train_dataset_s1, [subset_size, len(train_dataset_s1)-subset_size])[0]
    #train_dataset_s2 = torch.utils.data.random_split(train_dataset_s2, [subset_size, len(train_dataset_s2)-subset_size])[0]
    #val_dataset = torch.utils.data.random_split(val_dataset, [subset_size, len(val_dataset)-subset_size])[0]
    print("len(train_dataset_s1):", len(train_dataset_s1))
    print("len(train_dataset_s2):", len(train_dataset_s2))
    print("len(val_dataset):", len(val_dataset))
    
    if args.distributed:
        train_sampler_s1 = torch.utils.data.distributed.DistributedSampler(train_dataset_s1)
        train_sampler_s2 = torch.utils.data.distributed.DistributedSampler(train_dataset_s2)
    else:
        train_sampler_s1 = None
        train_sampler_s2 = None

    train_loader_s1 = torch.utils.data.DataLoader(
        train_dataset_s1, batch_size=args.batch_size, shuffle=(train_sampler_s1 is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler_s1, drop_last=False)
    train_loader_stcollect = torch.utils.data.DataLoader(
        train_dataset_s2, batch_size=args.batch_size, shuffle=(train_sampler_s2 is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler_s2, drop_last=False)
    #train_loader_s2 = torch.utils.data.DataLoader(
    #    train_dataset_s2, batch_size=args.batch_size, shuffle=(train_sampler_s2 is None),
    #    num_workers=args.workers, pin_memory=True, sampler=train_sampler_s2, drop_last=False)

    if args.batch_eval:
        val_batch_size = args.batch_size
    else:
        val_batch_size = device_count
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size, shuffle=False,  # batch size for each device should be 1
        num_workers=args.workers, pin_memory=True, drop_last=False)


    ############################################################################
    ### EVALUATE
    ############################################################################
    if args.evaluate:
        print("EVALUATE flag set, evaluating model now...")
        if args.oracle_eval:
            validate_oracle(val_loader, model, selector, criterion1, valid_combos, args)
        else:
            validate(val_loader, model, selector, criterion1, valid_combos, args, random=args.random_eval)
        return


    ############################################################################
    ### CREATE EPSILON GREEDY DECAY SCHEDULE
    ############################################################################
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


    ############################################################################
    ### EPOCH LOOP
    ############################################################################
    # Repeat for args.cycles
    for c in range(args.start_cycle, args.cycles):
        print("Starting cycle:", c)
        # Train Stage1
        for epoch in range(args.stage1_epochs_per_cycle):
            # Compute total stage1 epoch
            total_epoch = epoch + (c * args.stage1_epochs_per_cycle)
            # Update learning rate
            adjust_learning_rate(optimizer1, 1, total_epoch, args)
            # Update current epoch's epsilon
            epsilon = eps_schedule1[total_epoch].item()
            # Train STAGE1 for one epoch
            train_stage1(train_loader_s1, model, selector, criterion1, optimizer1, epoch, total_epoch, epsilon, valid_combos, args)
            # Evaluate if necessary
            if total_epoch != 0 and total_epoch % args.evaluate_freq == 0:
                evaluate_and_save(val_loader, model, selector, optimizer1, optimizer2, criterion1, c, total_epoch, model_name, valid_combos, args)

        # Evaluate post Stage1
        if args.stage1_epochs_per_cycle > 0:
            print("Starting Post-Stage1 Eval for Cycle:", c)
            evaluate_and_save(val_loader, model, selector, optimizer1, optimizer2, criterion1, c, total_epoch, model_name, valid_combos, args, filename='C{}_post_S1.pth.tar'.format(c))

        # Get selector truth (if this is not the first S2 train after loading with load_selector_truth)
        if not selector_truth:
            selector_truth = collect_selector_truth(train_loader_stcollect, len(train_dataset_s2), model, valid_combos, args)
            save_selector_truth(selector_truth, args.outdir, "selector_truth_train.pth.tar")

        #train_loader_s2 = generate_weighted_loader(train_dataset_s2, selector_truth, len(valid_combos), args)
        train_loader_s2 = train_loader_stcollect

        # Train Stage2
        for epoch in range(args.stage2_epochs_per_cycle):
            # Compute total stage2 epoch
            total_epoch = epoch + (c * args.stage2_epochs_per_cycle)
            # Update learning rate
            adjust_learning_rate(optimizer2, 2, total_epoch, args)
            # Update current epoch's epsilon
            epsilon = eps_schedule2[total_epoch].item()
            # Train STAGE2 for one epoch
            train_stage2(train_loader_s2, selector, optimizer2, epoch, total_epoch, epsilon, selector_truth, valid_combos, args)
            # Evaluate if necessary
            if total_epoch != 0 and total_epoch % args.evaluate_freq == 0:
                evaluate_and_save(val_loader, model, selector, optimizer1, optimizer2, criterion1, c, total_epoch, model_name, valid_combos, args)

        if args.stage2_epochs_per_cycle > 0:
            print("Starting Post-Stage2 Eval for Cycle:", c)
            evaluate_and_save(val_loader, model, selector, optimizer1, optimizer2, criterion1, c, total_epoch, model_name, valid_combos, args, filename='C{}_post_S2.pth.tar'.format(c))

        # To ensure we don't load truths from file again, set selector_truth to empty
        selector_truth = []



############################################################################
### STAGE 1
############################################################################
# One full epoch of training on task (stage 1)
def train_stage1(train_loader, model, selector, criterion, optimizer, epoch, total_epoch, epsilon, valid_combos, args):
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
    num_stride_options = len(args.model_cfg['STRIDER']['STRIDE_OPTIONS'])
    ss_choice_counts = torch.zeros(num_ss_blocks, num_stride_options)

    # switch to train mode
    model.train()
    selector.eval()
    end = time.time()

    # Iterate over training images
    for i, (images, target, indices) in enumerate(train_loader):
        #print("images:", images.shape)
        #print("target:", target, target.shape)
        #print("indices:", indices, indices.shape)
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        stride_combo = select_stride_combo(images, selector, epsilon, valid_combos)
        output = model(images, stride_combo)
        loss = criterion(output, target)
        #print("output:", output.shape)
        #print("loss:", loss)

        # Track choice counts per SS block
        for ss_id in range(len(stride_combo)):
            ss_choice_counts[ss_id][stride_combo[ss_id]] += images.shape[0]

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    # Print SS choice counts
    print("SS Choice Counts:")
    for i in range(ss_choice_counts.shape[0]):
        print("SS Block:", i, ss_choice_counts[i])



############################################################################
### COLLECT SELECTOR TRUTH
############################################################################
def collect_selector_truth(data_loader, dataset_length, model, valid_combos, args):
    print("Starting Selector Truth Collection:")
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(data_loader),
        [batch_time],
        prefix="STCollect")

    # Set to eval mode so we don't change BN averages
    model.eval()
    
    # Construct empty selector_truth
    selector_truth = []
    for i in range(dataset_length):
        selector_truth.append([[], [], [], [], []])

    with torch.no_grad():
        end = time.time()
        for i, (images, target, indices) in enumerate(data_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            # Iterate over all valid stride combinations
            for combo_idx in range(len(valid_combos)):
                stride_combo = valid_combos[combo_idx]
                # Compute output
                output = model(images, stride_combo)
                soft_output = F.softmax(output, dim=1)
                # Compute loss for each input in batch
                loss = torch.nn.functional.cross_entropy(output, target, reduction='none')
                # Compute correct/incorrect
                preds = output.argmax(dim=1)
                corrects = torch.eq(preds, target).int()

                #print("\n\noutput:", output, output.shape)
                #print("soft_output:", soft_output, soft_output.shape)
                #print("preds:", preds, preds.shape)
                #print("target:", target, target.shape)
                #print("corrects:", corrects, corrects.shape)
                # Record current corrects and losses for each sample in the batch
                for batch_sample_idx in range(len(indices)):
                    sample_idx = indices[batch_sample_idx].item()
                    task_target = target[batch_sample_idx].item()
                    pt = soft_output[batch_sample_idx][task_target].item()
                    #print(sample_idx, pt)
                    selector_truth[sample_idx][0].append(loss[batch_sample_idx].item())
                    selector_truth[sample_idx][1].append(pt)
                    selector_truth[sample_idx][2].append(corrects[batch_sample_idx].item())
                    selector_truth[sample_idx][3].append(task_target)
                    selector_truth[sample_idx][4].append(output[batch_sample_idx].tolist())
            
            # Measure elapsed time and show progress
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)

#    selector_truth = [-1] * dataset_length
#    best_loss = [10000.0] * dataset_length
#    with torch.no_grad():
#        end = time.time()
#        for i, (images, target, indices) in enumerate(data_loader):
#            if args.gpu is not None:
#                images = images.cuda(args.gpu, non_blocking=True)
#            target = target.cuda(args.gpu, non_blocking=True)
#            # Iterate over all valid stride combinations
#            for combo_idx in range(len(valid_combos)):
#                stride_combo = valid_combos[combo_idx]
#                # Compute output
#                output = model(images, stride_combo)
#                # Compute loss for each input in batch
#                loss = torch.nn.functional.cross_entropy(output, target, reduction='none')
#                # Check against bests that we already have and update if necessary
#                for batch_sample_idx in range(len(loss)):
#                    sample_idx = indices[batch_sample_idx].item()
#                    if loss[batch_sample_idx] < best_loss[sample_idx]:
#                        # Update best_loss
#                        best_loss[sample_idx] = loss[batch_sample_idx].item()
#                        # Update selector_truth
#                        selector_truth[sample_idx] = combo_idx
#
#            # Measure elapsed time and show progress
#            batch_time.update(time.time() - end)
#            end = time.time()
#            if i % args.print_freq == 0:
#                progress.display(i)

    return selector_truth


############################################################################
### STAGE 2
############################################################################
# One full epoch of training SS modules (stage 2)
def train_stage2(train_loader, selector, optimizer, epoch, total_epoch, epsilon, selector_truth, valid_combos, args):
    print("Starting Stage 2, Epoch:", epoch)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}] [{}] S2".format(epoch, total_epoch))

    # Create truth_counts tensor
    #truth_counts = torch.zeros(len(valid_combos))
    #for truth in selector_truth:
    #    truth_counts[truth] += 1
    #print("truth_counts:", truth_counts, truth_counts/truth_counts.sum())

    # Initialize criterion object
    # Non-weighted
    #criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # Standard inverse
    #weight = (truth_counts / truth_counts.sum()) ** -1.0
    # Normalized
    #weight = (-(truth_counts / truth_counts.sum()) + 1.0) ** args.eta
    #weight = truth_counts.min() / truth_counts
    #criterion = nn.CrossEntropyLoss(weight=weight).cuda(args.gpu)

    # Focal
    #weight = torch.ones(len(valid_combos))
    #criterion = FocalLoss(weight=weight, gamma=args.gamma, reduction='mean').cuda(args.gpu)

    #print("weight:", weight)

    # switch to train mode
    selector.train()
    end = time.time()

    # Iterate over training samples
    for i, (images, _, indices) in enumerate(train_loader): 
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        # Collect truth for SSM
        #target = torch.zeros_like(indices).cuda(args.gpu)
        #for batch_sample_idx in range(len(indices)):
        #    sample_idx = indices[batch_sample_idx].item()
        #    target[batch_sample_idx] = selector_truth[sample_idx]
        soft_target = generate_soft_targets(indices, selector_truth, args) 


        #count = [0] * len(valid_combos)
        #for t in target:
        #    count[t] += 1
        #print("count:", count)
        #continue

        # Forward batch thru selector
        output = selector(images)
        # Compute loss
        #loss = criterion(output, target)
        loss = xent_with_soft_targets(output, soft_target)

        # Generate hard target
        hard_target = torch.argmax(soft_target, dim=1)
        # measure accuracy and record loss
        acc1, _ = accuracy(output, hard_target, topk=(1, 2))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)



############################################################################
### VALIDATE 
############################################################################
def validate(val_loader, model, selector, criterion, valid_combos, args, random=False):
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
    num_stride_options = len(args.model_cfg['STRIDER']['STRIDE_OPTIONS'])
    ss_choice_counts = torch.zeros(num_ss_blocks, num_stride_options)

    # switch to evaluate mode
    model.eval()
    selector.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target, indices) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # Select stride combo
            # If random=True set epsilon HIGH
            if random:
                stride_combo = select_stride_combo(images, selector, 100, valid_combos)
            # Else set epsilon to -1 so it is pure greedy selection
            else:
                stride_combo = select_stride_combo(images, selector, -1, valid_combos)

            # Forward images thru task model
            output = model(images, stride_combo)
            
            # OPTIONAL: Show image
            #print("choice:", choices.item())
            #if choices.item() == 6:
            #    im2show = images[0].permute(1, 2, 0)
            #    show_image(im2show)
            #    exit()

            loss = criterion(output, target)

            # Track choice counts per SS block
            for ss_id in range(len(stride_combo)):
                ss_choice_counts[ss_id][stride_combo[ss_id]] += images.shape[0]

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



############################################################################
### VALIDATE W/ ORACLE
############################################################################
def validate_oracle(val_loader, model, selector, criterion, valid_combos, args):
    print("Running evaluation...")
    batch_time = AverageMeter('Time', ':6.3f')
    losses_best = AverageMeter('Loss', ':.4e')
    top1_best = AverageMeter('Acc@1', ':6.2f')
    top5_best = AverageMeter('Acc@5', ':6.2f')
    progress_best = ProgressMeter(
        len(val_loader),
        [batch_time, losses_best, top1_best, top5_best],
        prefix='BEST:  ')
    losses_worst = AverageMeter('Loss', ':.4e')
    top1_worst = AverageMeter('Acc@1', ':6.2f')
    top5_worst = AverageMeter('Acc@5', ':6.2f')
    progress_worst = ProgressMeter(
        len(val_loader),
        [batch_time, losses_worst, top1_worst, top5_worst],
        prefix='WORST: ')

    # Initialize counts tensor
    num_ss_blocks = sum([1 if x[0] == 1 else 0 for x in args.model_cfg['STRIDER']['BODY_CONFIG']])
    num_stride_options = len(args.model_cfg['STRIDER']['STRIDE_OPTIONS'])
    ss_choice_counts_best = torch.zeros(num_ss_blocks, num_stride_options)
    ss_choice_counts_worst = torch.zeros(num_ss_blocks, num_stride_options)

    # switch to evaluate mode
    model.eval()
    selector.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target, indices) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            loss_summary = []
            output_summary = []
            # Iterate over all stride combinations
            for stride_combo in valid_combos:
                # compute output
                output = model(images, stride_combo)
                loss = torch.nn.functional.cross_entropy(output, target, reduction='none')
                loss_summary.append(loss.unsqueeze(1))
                output_summary.append(output)

            loss_summary = torch.cat(loss_summary, dim=1)
            # Find indices for stride options that lead to lowest (and highest) loss for each sample in batch
            min_loss_vals, min_loss_inds = torch.min(loss_summary, dim=1)
            max_loss_vals, max_loss_inds = torch.max(loss_summary, dim=1)
            # Compute average loss of best stride options
            loss_best = torch.mean(min_loss_vals)
            loss_worst = torch.mean(max_loss_vals)
            # Organize output tensor such that the output for each sample is from the optimal stride option
            output_best = []
            output_worst = []
            for b in range(target.shape[0]):
                best_stride_option = min_loss_inds[b]
                worst_stride_option = max_loss_inds[b]
                output_best.append(output_summary[best_stride_option][b].unsqueeze(0))
                output_worst.append(output_summary[worst_stride_option][b].unsqueeze(0))
            output_best = torch.cat(output_best, dim=0)
            output_worst = torch.cat(output_worst, dim=0)

            # Record stride selections
            for combo_ind in min_loss_inds:
                for ss_id in range(len(valid_combos[combo_ind])):
                    ss_choice_counts_best[ss_id][valid_combos[combo_ind][ss_id]] += 1
            for combo_ind in max_loss_inds:
                for ss_id in range(len(valid_combos[combo_ind])):
                    ss_choice_counts_worst[ss_id][valid_combos[combo_ind][ss_id]] += 1

            # measure accuracy and record loss
            acc1_best, acc5_best = accuracy(output_best, target, topk=(1, 5))
            losses_best.update(loss_best.item(), images.size(0))
            top1_best.update(acc1_best[0], images.size(0))
            top5_best.update(acc5_best[0], images.size(0))

            acc1_worst, acc5_worst = accuracy(output_worst, target, topk=(1, 5))
            losses_worst.update(loss_worst.item(), images.size(0))
            top1_worst.update(acc1_worst[0], images.size(0))
            top5_worst.update(acc5_worst[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress_worst.display(i)
                progress_best.display(i)

        # TODO: this should also be done with the ProgressMeter
        print('WORST: * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1_worst, top5=top5_worst, loss=losses_worst))
        print('BEST:  * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1_best, top5=top5_best, loss=losses_best))

        # Print SS choice counts
        print("\nSS Choice Counts (WORST):")
        for i in range(ss_choice_counts_worst.shape[0]):
            print("SS Block:", i, ss_choice_counts_worst[i])
        print("\nSS Choice Counts (BEST):")
        for i in range(ss_choice_counts_best.shape[0]):
            print("SS Block:", i, ss_choice_counts_best[i])



############################################################################
### GET VALID STRIDE OPTION COMBOS
############################################################################
def get_valid_stride_combos(args):
    body_config = args.model_cfg['STRIDER']['BODY_CONFIG']
    stride_options = args.model_cfg['STRIDER']['STRIDE_OPTIONS']
    downsample_bounds = args.model_cfg['STRIDER']['DOWNSAMPLE_BOUNDS']
    # Initialize counts tensor
    num_ss_blocks = sum([1 if x[0] == 1 else 0 for x in body_config])
    num_stride_options = len(stride_options)
    stride_options_scales = [[1/x[1][0], 1/x[1][1]] if x[0] else x[1] for x in stride_options]
    ss_choice_counts = torch.zeros(num_ss_blocks, num_stride_options)

    # Create list of all possible stride options
    option_list = list(range(num_stride_options))
    all_combos = list(itertools.product(option_list, repeat=num_ss_blocks))
    valid_combos = []

    # Trim stride options that are invalid due to bounds
    for i in range(len(all_combos)):
        valid = True
        curr_downsample = [4, 4]  # [dH, dW] Stem downsamples H and W by 4x
        adaptive_idx = 0
        # Iterate over network configs to check downsample rate
        for layer_idx in range(len(body_config)):
            # If the curr layer is adaptive
            if body_config[layer_idx][0] == 1:
                stride = stride_options_scales[all_combos[i][adaptive_idx]]
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
                valid = False
                break   # Out of bounds, do NOT consider this stride combo
        if valid:    
            valid_combos.append(all_combos[i])

    return valid_combos



############################################################################
### SELECT STRIDE COMBO
############################################################################
def select_stride_combo(images, selector, epsilon, valid_combos):
    # Select stride combo with epsilon greedy exploration
    sample = random.random()
    # If greedy, select stride combo using majority vote over batch
    if sample > epsilon:
        with torch.no_grad():
            selector_out = selector(images)
        selector_out_soft = F.softmax(selector_out, dim=1)
        selector_out_sum = torch.sum(selector_out_soft, dim=0)
        sorted_stride_combos = torch.argsort(selector_out_sum, descending=True)
        choice_idx = sorted_stride_combos[0].item()
        choice = valid_combos[choice_idx]
    # Else, randomly select a valid combo
    else:
        choice = random.choice(valid_combos)

    return choice
    


############################################################################
### EVALUATE AND SAVE
############################################################################
def evaluate_and_save(val_loader, model, selector, optimizer1, optimizer2, criterion1, cycle, total_epoch, model_name, valid_combos, args, filename='checkpoint.pth.tar'):
    global best_acc1
    # Evaluate on validation set
    acc1 = validate(val_loader, model, selector, criterion1, valid_combos, args)

    # remember best acc@1 and save checkpoint
    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            and args.rank % ngpus_per_node == 0):
        save_checkpoint({
            'cycle': cycle + 1,
            'total_epoch': total_epoch + 1,
            'arch': model_name,
            'model_sd': model.state_dict(),
            'selector_sd': selector.state_dict(),
            'best_acc1': best_acc1,
            'optimizer1' : optimizer1.state_dict(),
            'optimizer2' : optimizer2.state_dict(),
        }, is_best, outdir=args.outdir, filename=filename)



############################################################################
### SAVE CHECKPOINT
############################################################################
def save_checkpoint(state, is_best, outdir='./out/test', filename='checkpoint.pth.tar'):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    filepath = os.path.join(outdir, filename)
    print("Saving checkpoint to: {} ".format(filepath))
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(outdir, 'model_best.pth.tar'))



############################################################################
### SAVE CHECKPOINT
############################################################################
def save_selector_truth(selector_truth, outdir='./out/test', filename='selector_truth.pth.tar'):
    filepath = os.path.join(outdir, filename)
    print("Saving selector_truth to: {} ".format(filepath))
    torch.save(selector_truth, filepath)



############################################################################
### PERFORMANCE METER CLASSES
############################################################################
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



############################################################################
### Alternative loss functions
############################################################################
class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target,reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


def xent_with_soft_targets(logit_preds, targets):
    logsmax = F.log_softmax(logit_preds, dim=1)
    batch_loss = targets * logsmax
    batch_loss =  -1*batch_loss.sum(dim=1)
    return batch_loss.mean()



############################################################################
### HELPERS
############################################################################
def generate_soft_targets(indices, selector_truth, args):
    soft_targets = []
    with torch.no_grad():
        for batch_sample_idx in range(len(indices)):
            sample_idx = indices[batch_sample_idx].item()
            corrects = torch.tensor(selector_truth[sample_idx][2], dtype=torch.float32).cuda(args.gpu)
            if corrects.sum() > 0:
                curr_st = corrects / corrects.sum()
            else:
                curr_st = F.softmax(corrects, dim=0)
            curr_st.unsqueeze_(0)
            soft_targets.append(curr_st)
            #print("corrects:", corrects, corrects.shape)
            #print("curr_st:", curr_st, curr_st.shape)

        soft_targets = torch.cat(soft_targets)
        #print("soft_targets:", soft_targets, soft_targets.shape)
        #exit()
    return soft_targets

#def generate_soft_targets(indices, selector_truth, args):
#    soft_targets = []
#    with torch.no_grad():
#        for batch_sample_idx in range(len(indices)):
#            sample_idx = indices[batch_sample_idx].item()
#            losses = torch.tensor(selector_truth[sample_idx][0]).cuda(args.gpu)
#            losses.unsqueeze_(0)
#            curr_st = F.softmax(-(losses * args.eta), dim=1)
#            soft_targets.append(curr_st)
#            #print("losses:", losses, losses.shape)
#            #print("curr_st:", curr_st, curr_st.shape)
#
#        soft_targets = torch.cat(soft_targets)
#        print("soft_targets:", soft_targets, soft_targets.shape)
#        exit()
#    return soft_targets


def generate_weighted_loader(dataset, truth, nclasses, args):
    # First, organize truth properly
    cls_truth = []
    for sample_idx in range(len(truth)):
        curr_losses = torch.tensor(truth[sample_idx][0])
        curr_truth = torch.argmin(curr_losses).item()
        cls_truth.append(curr_truth)
        #print("curr_losses:", curr_losses)
        #print("curr_truth:", curr_truth)


    #exit()
    # Generate weight
    count = [0] * nclasses
    for t in cls_truth:
        count[t] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(cls_truth)
    for idx, val in enumerate(cls_truth):
        weight[idx] = weight_per_class[val]

    # Create the sampler
    weight = torch.DoubleTensor(weight)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weight, len(weight))
    # Create the loader
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers, pin_memory=True)
    return loader


def add_indices_to_dataset(cls):
    """
    Modifier the given Dataset class to return data, target, index instead
    of just data, target.
    source: https://discuss.pytorch.org/t/how-to-retrieve-the-sample-indices-of-a-mini-batch/7948/19
    """
    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })


def get_lr(optimizer):
    """Returns current learning rate of an optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def adjust_learning_rate(optimizer, stage, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if stage == 1:
        lr = args.lr1 * (0.1 ** (epoch // args.lr1_decay_every))
    else:
        lr = args.lr2 * (0.1 ** (epoch // args.lr2_decay_every))
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


def unravel_index(index, shape):
    """See numpy's unravel_index function"""
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


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
