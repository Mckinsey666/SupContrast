from __future__ import print_function

import os
import sys
import argparse
import time
import math
import numpy as np

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms, datasets
from torchvision.utils import save_image

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, set_optimizer_param, save_model, save_param
from util import Denormalize

from aug import CIFAR10_Transform, relaxed_bernoulli
from custom_data import CIFAR10_PretextRotate

from networks.resnet_big import SupConResNet, LinearClassifier
from networks.aug import AugSample
from losses import SupConLoss

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.autograd.set_detect_anomaly(True)

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--aug_num', type=int, default=4,
                        help='number of augmentations')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'], help='dataset')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
                        

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 50
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    """

    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if opt.dataset == 'cifar10':
        """
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)
        """
        train_dataset = CIFAR10_PretextRotate(root=opt.data_folder,
                                              transform=train_transform)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader


def set_model(opt):
    model = SupConResNet(name=opt.model)

    contrast_criterion = SupConLoss(temperature=opt.temp)

    classifier = LinearClassifier(name=opt.model, num_classes=4) # rotation pretext = 4 labels
    cls_criterion = torch.nn.CrossEntropyLoss()

    aug_sample = AugSample(opt.aug_num)
    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        contrast_criterion = contrast_criterion.cuda()
        classifier = classifier.cuda()
        cls_criterion = cls_criterion.cuda()
        aug_sample = aug_sample.cuda()
        cudnn.benchmark = True

    transform_module = CIFAR10_Transform()

    return model, contrast_criterion, classifier, cls_criterion, aug_sample, transform_module


def train(train_loader, model, contrast_criterion, classifier, cls_criterion, aug_sample, optimizers, epoch, opt, transform):
    """one epoch training"""
     # 128 is maximum batch size that fits memory
    denorm = Denormalize(transform.mean, transform.std)


    model.train()
    classifier.train()
    aug_sample.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    info_losses = AverageMeter()
    cls_losses = AverageMeter()

    end = time.time()
    
    for idx, (images, labels, pretext_labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.to(device) # images is batch tensor

        labels = labels.to(device)#cuda(non_blocking=True)
        pretext_labels = pretext_labels.to(device)
        bsz = labels.shape[0]

        # warm-up learning rate for all optimizers
        for key in optimizers:
            warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizers[key])


        # Train feature extractor
        s = aug_sample().to(device)
        v1 = transform(images, s)
        v2 = transform(images, s)
        v = torch.cat([v1, v2], dim = 0).to(device)

        features = model(v)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = contrast_criterion(features)
        info_losses.update(loss.item(), bsz)

        optimizers['model'].zero_grad()
        loss.backward() # retain variables
        optimizers['model'].step()
            
        # train g, c
        s = aug_sample().to(device)
        v1 = transform(images, s)
        v2 = transform(images, s)
        v = torch.cat([v1, v2], dim = 0).to(device)

        features = model(v)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        info_loss = contrast_criterion(features)

        hidden_features = model.hidden_feat
        predict_pretext_labels = classifier(hidden_features)

        pretext_labels = pretext_labels.repeat(2)
        cls_loss = cls_criterion(predict_pretext_labels, pretext_labels)

        cls_losses.update(cls_loss.item(), bsz)

        loss = -info_loss + cls_loss

        optimizers['p'].zero_grad()
        optimizers['cls'].zero_grad()
        loss.backward()
        aug_sample.prob.grad *= 1e3
        optimizers['p'].step()
        print(aug_sample.prob.grad)
        optimizers['cls'].step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'info_loss {info_loss.val:.3f} ({info_loss.avg:.3f})\t'
                  'cls_loss {cls_loss.val:.3f} ({cls_loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), info_loss=info_losses, 
                   cls_loss=cls_losses))
            #print('P: [{}]'.format(torch.sigmoid(aug_params['p'])))
            sys.stdout.flush()

    return info_losses.avg, cls_losses.avg


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    opt = parse_option()
    p_evolve = []
    info_log = []
    cls_log = []

    # build data loader and transform module
    train_loader = set_loader(opt)

    # build model, criterion, and transform module
    model, contrast_criterion, classifier, cls_criterion, aug_sample, transform = set_model(opt)

    # build optimizers
    optimizers = {}
    optimizers['model'] = set_optimizer(opt, model)
    optimizers['cls'] = set_optimizer(opt, classifier)
    optimizers['p'] = optim.SGD([aug_sample.prob], lr=1)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        for key in optimizers:
            if key == 'p':
                continue
            adjust_learning_rate(opt, optimizers[key], epoch)
        
        # anneal temperature each epoch
        anneal_temp = (epoch - opt.epochs) * 1.0 / (1 - opt.epochs)
        aug_sample.temperature = anneal_temp

        # train for one epoch
        time1 = time.time()
        info_loss, cls_loss = train(
            train_loader, 
            model, contrast_criterion, 
            classifier, cls_criterion, 
            aug_sample, optimizers, epoch, opt, transform)

        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        print("P: {}".format(aug_sample.prob.data))

        p_evolve.append(aug_sample.prob.data.clone().cpu().unsqueeze(-1).numpy())
        #print(np.concatenate(p_evolve, axis=1))
        np.save('p_comp.npy', np.concatenate(p_evolve, axis=1))
        info_log.append(info_loss)
        np.save('info_loss.npy', np.array(info_log))
        cls_log.append(cls_loss)
        np.save('cls_loss.npy', np.array(cls_log))

        # tensorboard logger
        logger.log_value('info_loss', info_loss, epoch)
        logger.log_value('cls_loss', cls_loss, epoch)
        #logger.log_value('learning_rate', optimizers['model'].param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizers['model'], opt, epoch, save_file)

    # save the last model and params
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizers['model'], opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
