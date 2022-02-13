# Copyright (C) 2019 Amir Alansary <amiralansary@gmail.com>
# License: GPL-3.0
# This file is based on the official pytorch ImageNet example below
# https://github.com/pytorch/examples/blob/master/imagenet/main.py

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import random
import shutil
import time
import warnings
import numpy as np
import pandas as pd
from torchsummary import summary

from utils.metrics import accuracy, accuracy_for_multi_label

# from torch.optim import RAdam
from torch.optim import AdamW
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from utils.config import MyRobertaConfig,MyRobertaClassificationConfig
from transformers import RobertaConfig, BertConfig, BertForSequenceClassification

from roberta import MyRobertaForSequenceClassification
from bert import MyBertForSequenceClassification
# from modeling.roberta import RobertaForSequenceClassification
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, RobertaModel,RobertaForSequenceClassification

from prepare import get_clinc_datasets, get_num_labels
from Dataset import multi_intent_dataset, get_ATIS_num_labels, get_SNIPS_num_labels
import matplotlib.pyplot as plt
from utils.building_utils import load_model


from utils.model_config import Config
from label_aware import MulCon
from Dataset import get_labels_vocab
from transformers import BertTokenizer



parser = argparse.ArgumentParser(description='Multi-intent Detection')
parser.add_argument('-s', '--save_dir', metavar='SAVE_DIR',
                    help='path to the save directory', default='models')
parser.add_argument('-p', '--data-path', metavar='TRAIN_FILES', default=None,
                    help='path to the csv file that contain training data')
parser.add_argument('-c', '--classes', default=5, type=int,
                    metavar='CLASSES', help='number of classes')
parser.add_argument('-a', '--arch', metavar='ARCH', default='roberta')
parser.add_argument('-j', '--workers', default=os.cpu_count(),
                    type=int, metavar='N',
                    help='number of data loading workers (default: max)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--optim', default='adam', type=str, metavar='OPTIM',
                    help='select optimizer [sgd, adam]',
                    dest='optim')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_scheduler', default=None, type=str, metavar='LR_SCH',
                    help='learning scheduler [reduce, cyclic, cosine]',
                    dest='lr_scheduler')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-f', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
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
                         'fastest way to use PyTorch for either single node or'
                         ' multi node data parallel training')
parser.add_argument('--suffix', default='', type=str, metavar='SUFFIX',
                    help='add suffix to model save', dest='suffix')
parser.add_argument('--save_results', default='validation_results.csv', type=str,
                    help='Save validation results in a csv file')
parser.add_argument('--pos_tag', dest='pos_tag', action='store_true',
                    help='use pos_tag_embedding')
parser.add_argument('--multi_intent', dest='multi_intent', action='store_true',
                    help='use multi_intent')
###############################################################################
best_acc1 = 0

def main():
    args = parser.parse_args()

    # seed_everything
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
    else:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is None:
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

    ###########################################################################
    # create model
    ###########################################################################

    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        # args.arch: roberta-base
        # tokenizer = AutoTokenizer.from_pretrained(args.arch)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        if args.pos_tag:
            # config = RobertaConfig.from_pretrained('roberta-base')
            # model = MyRobertaForSequenceClassification(config)
            # model = load_model(model, "./pytorch_model_roberta.bin")
            config = BertConfig.from_pretrained('bert-base-uncased')
            model = MyBertForSequenceClassification(config)
            model = load_model(model, "./pytorch_model_bert.bin")
        elif args.arch.startswith('MulCon'):
            config = Config()
            labels = get_labels_vocab(config, args.data_path)
            model = MulCon(config, labels)
            print(labels)
            print(config.num_classes)
        else:
            # model = RobertaForSequenceClassification.from_pretrained(args.arch)
            model = BertForSequenceClassification.from_pretrained(args.arch)

    else:
        print("=> creating model '{}'".format(args.arch))
        config = RobertaConfig.from_pretrained('roberta-base')
        tokenizer = AutoTokenizer.from_pretrained(args.arch)
        model = AutoModel(config)

    ###########################################################################
    # save directory
    ###########################################################################
    save_dir = os.path.join(os.getcwd(), args.save_dir)
    save_dir += ('/arch[{}]_optim[{}]_lr[{}]_lrsch[{}]_batch[{}]').format(
        args.arch,
        args.optim,
        args.lr,
        args.lr_scheduler,
        args.batch_size)
    if args.suffix:
        save_dir += '_{}'.format(args.suffix)
    save_dir = save_dir[:]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ###########################################################################
    # Initialize and Reshape the Networks
    ###########################################################################
    if args.arch.startswith('roberta'):
        if args.multi_intent:
            model.config.problem_type = "multi_label_classification"
            model.num_labels = get_ATIS_num_labels()
            num_hidden = model.classifier.out_proj.in_features
            model.classifier.out_proj = nn.Linear(num_hidden, get_ATIS_num_labels())
        else:
            model.config.problem_type = "single_label_classification"
            model.num_labels = get_num_labels()
            num_hidden = model.classifier.out_proj.in_features
            model.classifier.out_proj = nn.Linear(num_hidden, get_num_labels())

    if args.arch.startswith('bert'):
        if args.multi_intent:
            model.config.problem_type = "multi_label_classification"
            model.num_labels = get_ATIS_num_labels()
            num_hidden = model.classifier.in_features
            model.classifier = nn.Linear(num_hidden, get_ATIS_num_labels())
        else:
            model.config.problem_type = "single_label_classification"
            model.num_labels = get_num_labels()
            num_hidden = model.classifier.in_features
            model.classifier = nn.Linear(num_hidden, get_num_labels())

    ###########################################################################
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
        print("Use Model on GPU")
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print("Data Parallel")
        model = torch.nn.DataParallel(model).cuda()

    ###########################################################################
    # Loss and optimizer
    ###########################################################################
    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    # weighted loss if necessary
    # weights = [0.27218907, 2.8756447,  1.32751323, 8.04719359, 9.92259887]
    # class_weights = torch.FloatTensor(weights).cuda(args.gpu)
    # criterion = nn.CrossEntropyLoss(weight=class_weights).cuda(args.gpu)

    # define optimizer
    if args.optim == 'sgd':
        print("=> using '{}' optimizer".format(args.optim))
        optimizer = torch.optim.SGD(model.parameters(),
                                    args.lr,
                                    momentum=0.9,
                                    weight_decay=1e-5,
                                    nesterov=True)
    elif args.optim == "radam":
        print("=> using '{}' optimizer".format(args.optim))
        optimizer = RAdam(model.parameters(), args.lr, betas=(0.9, 0.999), eps=1e-08,
                          weight_decay=1e-5)
    elif args.optim == "adamw":
        print("=> using '{}' optimizer".format(args.optim))
        optimizer = AdamW(model.parameters(), args.lr, betas=(0.9, 0.999), eps=1e-08,
                          weight_decay=0.01)
    else:  # default is adam
        print("=> using '{}' optimizer".format(args.optim))
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                     betas=(0.9, 0.999), eps=1e-08,
                                     weight_decay=1e-5,
                                     amsgrad=False)

    ###########################################################################
    # Resume training and load a checkpoint
    ###########################################################################
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
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    ###########################################################################
    # Data Transforms
    ###########################################################################
    # Todo: Pos Tag Processing

    ###########################################################################
    # Learning rate scheduler
    ###########################################################################
    print("=> using '{}' initial learning rate (lr)".format(args.lr))
    # define learning rate scheduler

    scheduler = args.lr_scheduler
    if args.lr_scheduler == 'reduce':
        print("=> using '{}' lr_scheduler".format(args.lr_scheduler))
        # Reduce learning rate when a metric has stopped improving.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=0.8,
                                                               patience=2)
    elif args.lr_scheduler == 'cyclic':
        print("=> using '{}' lr_scheduler".format(
            args.lr_scheduler))  # optimizer must support momentum with `cycle_momentum` option enabled
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                      base_lr=0.00005,
                                                      max_lr=0.005,
                                                      cycle_momentum=True)
    elif args.lr_scheduler == 'cosine':
        print("=> using '{}' lr_scheduler".format(args.lr_scheduler))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=10,
                                                               eta_min=0)
    elif args.lr_scheduler == 'step':
        print("=> using '{}' lr_scheduler".format(args.lr_scheduler))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)

    ###########################################################################
    # load validation data and apply transforms
    ###########################################################################
    # train_dataset, valid_dataset, test_dataset = get_clinc_datasets(tokenizer)

    train_dataset = multi_intent_dataset(args.data_path, "train", tokenizer)
    valid_dataset = multi_intent_dataset(args.data_path, "dev", tokenizer)

    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    test_dataset = multi_intent_dataset(args.data_path, "test", tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        # retrieve correct save path from saved model
        test_dataset = multi_intent_dataset(args.data_path, "test", tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)
        save_dir = os.path.split(args.resume)[0]
        validate(test_loader, model, criterion, save_dir, args)
        return

    ###########################################################################
    # load train data and apply transforms
    ###########################################################################
    # if args.train_files:
    #     train_dataset = BalancedImageLabelDataset(csv=args.train_files,
    #                                       transform=data_transforms['train'],
    #                                       label_names=LABELS, shuffle=True, balanced=True, scale=2) # Todo
    # train_dataset = ImageLabelDataset(csv=args.train_files,
    #                                   transform=data_transforms['train'],
    #                                   label_names=LABELS) # Todo

    print("Training dataset size: ", len(train_dataset))

    ## [TODO]: Evaluate Balance of Datasets

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    # [TODO] weighted sampling for class imbalance
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    #     num_workers=args.workers, pin_memory=True, sampler=ImbalancedDatasetSampler(train_dataset))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size, shuffle=(train_sampler is None),
                                               num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    ###########################################################################
    # Train the model
    ###########################################################################

    for epoch in range(args.start_epoch, args.epochs):
        # train sampler
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch, args)
        print_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer,
              scheduler, epoch, args)

        # evaluate on validation set
        acc1, loss = validate(val_loader, model, criterion, save_dir, args)

        print("Test Eval")
        validate(test_loader, model, criterion, save_dir, args)

        # update learning rate based on lr_scheduler
        if (args.lr_scheduler == 'reduce'):
            scheduler.step(loss)
        elif (args.lr_scheduler == 'cosine'):
            scheduler.step()
        elif (args.lr_scheduler == 'step'):
            scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            print("Saving model [{}]...".format(save_dir))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_checkpoint({'epoch': epoch + 1,
                             'arch': args.arch,
                             'state_dict': model.state_dict(),
                             'best_acc1': best_acc1,
                             'optimizer': optimizer.state_dict(), },
                            is_best,
                            save_dir=save_dir)
            print(30 * '=')


###############################################################################
###############################################################################
def train(train_loader, model, criterion, optimizer, scheduler, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('Acc@2', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top2],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    for i, batch in enumerate(train_loader):

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["intent"]
        # pos_tag_ids = batch["pos_tag_ids"]

        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input_ids = input_ids.cuda(args.gpu)
            attention_mask = attention_mask.cuda(args.gpu)
            labels = labels.cuda(args.gpu)
            # pos_tag_ids = pos_tag_ids.cuda(args.gpu, non_blocking=True)

        # compute output
#         output = model(input_ids=input_ids, attention_mask=attention_mask, pos_tag_ids = pos_tag_ids, labels=labels)
#         output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        output = model(input_ids, attention_mask, labels)

#         logits = output.logits
#         loss = output.loss
        logits = output[1]
        # loss = output[0]
        loss = output[0] + output[2]

        # loss = criterion(logits, labels)
        # pred = torch.argmax(logits, dim=-1)
        # loss = output[0]
        # logits = output[1]
        # print(logits.size())
        # print(labels.size())
        # measure accuracy and record loss
        acc1, acc2 = accuracy_for_multi_label(logits, labels, topk=(1, 2))
        # acc1, acc2 = accuracy(logits, labels, topk=(1, 2))
        losses.update(loss.item(), input_ids.size(0))
        top1.update(acc1[0], input_ids.size(0))
        top2.update(acc2[0], input_ids.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update learning rate
        if args.lr_scheduler == 'cyclic':
            scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


###############################################################################
###############################################################################
def validate(val_loader, model, criterion, save_dir, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('Acc@2', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top2],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    # create dataframe to save results in csv file
    if args.save_results:
        results_df = pd.DataFrame(columns=['label', 'predict', 'predict_top2'])

    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(val_loader):

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["intent"]
            # pos_tag_ids = batch["pos_tag_ids"]

            if args.gpu is not None:
                input_ids = input_ids.cuda(args.gpu, non_blocking=True)
                attention_mask = attention_mask.cuda(args.gpu, non_blocking=True)
                labels = labels.cuda(args.gpu, non_blocking=True)
                # pos_tag_ids = pos_tag_ids.cuda(args.gpu, non_blocking=True)

            # compute output
#             output = model(input_ids=input_ids, attention_mask=attention_mask, pos_tag_ids=pos_tag_ids,labels=labels)
#             output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            output = model(input_ids, attention_mask, labels)

            #             loss = output.loss
#             logits = output.logits
            logits = output[1]
            # loss = output[0]
            loss = output[0] + output[2]

            # Get top2 predictions
            _, pred = logits.topk(2, 1, True, True)

            # measure accuracy and record loss
            acc1, acc2 = accuracy_for_multi_label(logits, labels, topk=(1, 2))
            # acc1, acc2 = accuracy(logits, labels, topk=(1, 2))

            losses.update(loss.item(), input_ids.size(0))
            top1.update(acc1[0], input_ids.size(0))
            top2.update(acc2[0], input_ids.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # update dataframe with new results
            for b in range(len(batch['intent'])):
                results_df = results_df.append(
                    dict(label=batch['intent'][b].cpu().numpy(), predict=pred[b, 0].cpu().numpy(),
                         predict_top2=pred[b].cpu().numpy()), ignore_index=True)

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@2 {top2.avg:.3f}'
              .format(top1=top1, top2=top2))

    if args.save_results:
        # Save validation results
        if args.evaluate:
            args.save_results = "eval_" + args.save_results
        results_file = os.path.join(save_dir, args.save_results)
        results_df.to_csv(results_file, index=False)
        # print(results_df)

    return top1.avg, loss.item()


###############################################################################
###############################################################################
def predict(seq, model, args, tokenizer):
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        seq = tokenizer(seq, return_tensors="pt")
        if args.gpu is not None:
            seq = seq.cuda(args.gpu, non_blocking=True)
        output = model(**seq)
        logits = output.logits
        _, pred = logits.topk(1, 1, True, True)
        return logits.cpu().numpy(), pred.t().cpu().numpy()


###############################################################################
###############################################################################
def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    if is_best:
        filename = os.path.join(save_dir, 'model_best.pth.tar')
    else:
        filename = os.path.join(save_dir, filename)

    torch.save(state, filename)


###############################################################################
###############################################################################
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


###############################################################################
###############################################################################
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


###############################################################################
###############################################################################
def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


###############################################################################
###############################################################################
def print_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        print("Epoch: [{}] Current learning rate (lr) = {}".format(
            epoch, param_group['lr']))


###############################################################################
###############################################################################
if __name__ == '__main__':
    main()
