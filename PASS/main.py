# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import sys
sys.path.append('/home/andy/PASS-reID-main/PASS_cluster_contrast_reid')
import argparse
import os
import os.path as osp
import random
import numpy as np
import time
import utils
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets as torch_datasets
import torch.distributed as dist

import vision_transformer as vits
from vision_transformer import DINOHead
from clustercontrast import datasets
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data import IterLoader
from clustercontrast.utils.data.sampler import RandomMultipleGallerySampler
from clustercontrast.utils.data.preprocessor import Preprocessor
from clustercontrast.utils.logging import Logger
from clustercontrast.evaluators import Evaluator, extract_features
from clustercontrast import models
from clustercontrast.trainers import ClusterContrastTrainer

def get_data(name, data_dir):
    #root = './data'
    dataset = datasets.create(name, data_dir)
    return dataset

def get_test_loader(args, dataset, height, width, batch_size, workers, testset=None):

    test_transformer = DataAugmentationDINO(
        (height,width),
        (args.crop_height,args.crop_width),
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

class DataAugmentationDINO(object):
    def __init__(self, size, crop_size, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        #print(local_crops_scale)
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(size=crop_size, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        #print('original img', image.size)
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        width, height = image.size
        image_test = transforms.functional.crop(image, 0, 0, int(0.5*height), width)
        #print('crop img', image_test.size)
        for _ in range(self.local_crops_number//3):
            crops.append(self.local_transfo(transforms.functional.crop(image, 0, 0, int(0.5*height), width)))
        for _ in range(self.local_crops_number//3):
            crops.append(self.local_transfo(transforms.functional.crop(image, int(0.25*height), 0, int(0.5*height), width)))
        for _ in range(self.local_crops_number//3):
            crops.append(self.local_transfo(transforms.functional.crop(image, int(0.5*height), 0, int(0.5*height), width)))
        return crops

def create_model(args):
    if 'resnet' in args.arch:
        model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                num_classes=0, pooling_type=args.pooling_type,pretrained_path=args.pretrained_path)
    else:
        model = models.create(args.arch,img_size=(args.height,args.width),drop_path_rate=args.drop_path_rate
                , pretrained_path = args.pretrained_path,hw_ratio=args.hw_ratio, conv_stem=args.conv_stem, feat_fusion=args.feat_fusion, multi_neck=args.multi_neck)
    # use CUDA

    model.cuda()
    # model = nn.DataParallel(model)
    return model

def main():
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    start_time = time.monotonic()
    cudnn.benchmark = True
    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    transform = DataAugmentationDINO(
        (args.height, args.width),
        (args.crop_height, args.crop_width),
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )
    reid_dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(args, reid_dataset, args.height, args.width, args.batch_size, args.workers)


    # dataset = torch_datasets.ImageFolder(args.pet_data_dir, transform=transform)
    # data_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=args.batch_size,
    #     num_workers=args.workers,
    #     pin_memory=True,
    #     drop_last=True,
    # )

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a vision transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            img_size=(args.height, args.width),
            patch_size=args.patch_size,
            drop_path_rate=0.1,  # stochastic depth
            pretrained_path = args.pretrained_path,
            hw_ratio = args.hw_ratio
        )
        teacher = vits.__dict__[args.arch](
            img_size=(args.height, args.width),
            patch_size=args.patch_size,
            pretrained_path = args.pretrained_path,
            hw_ratio = args.hw_ratio
        )
        embed_dim = student.embed_dim
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    student = student.cuda()

    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    teacher = teacher.cuda()

    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()



    for epoch in range(args.start_epochs, args.epochs):
        for it, data in enumerate(test_loader):
            images = data[0]
            images = [im.cuda(non_blocking=True) for im in images]
            teacher_output_g = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output_g, student_output_pt1, student_output_pt2, student_output_pt3 = student(images)
            loss = dino_loss(teacher_output_g, student_output_g, student_output_pt1, student_output_pt2, student_output_pt3,
                             epoch)
            print(it)


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center_cls", torch.zeros(1, out_dim))
        self.register_buffer("center_pt1", torch.zeros(1, out_dim))
        self.register_buffer("center_pt2", torch.zeros(1, out_dim))
        self.register_buffer("center_pt3", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, teacher_output_g, student_output_g, student_output_pt1, student_output_pt2, student_output_pt3,
                epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """

        teacher_output_g_cls, teacher_output_g_pt1, teacher_output_g_pt2, teacher_output_g_pt3 = teacher_output_g
        student_output_g_cls, student_output_g_pt1, student_output_g_pt2, student_output_g_pt3 = student_output_g
        student_output_p1_cls, student_output_p1_pt = student_output_pt1
        student_output_p2_cls, student_output_p2_pt = student_output_pt2
        student_output_p3_cls, student_output_p3_pt = student_output_pt3

        student_output_g_cls = student_output_g_cls / self.student_temp
        student_output_g_pt1 = student_output_g_pt1 / self.student_temp
        student_output_g_pt2 = student_output_g_pt2 / self.student_temp
        student_output_g_pt3 = student_output_g_pt3 / self.student_temp
        student_output_p1_cls = student_output_p1_cls / self.student_temp
        student_output_p1_pt = student_output_p1_pt / self.student_temp
        student_output_p2_cls = student_output_p2_cls / self.student_temp
        student_output_p2_pt = student_output_p2_pt / self.student_temp
        student_output_p3_cls = student_output_p3_cls / self.student_temp
        student_output_p3_pt = student_output_p3_pt / self.student_temp

        student_output_g_cls = student_output_g_cls.chunk(2)
        student_output_g_pt1 = student_output_g_pt1.chunk(2)
        student_output_g_pt2 = student_output_g_pt2.chunk(2)
        student_output_g_pt3 = student_output_g_pt3.chunk(2)
        student_output_p1_cls = student_output_p1_cls.chunk(3)
        student_output_p1_pt = student_output_p1_pt.chunk(3)
        student_output_p2_cls = student_output_p2_cls.chunk(3)
        student_output_p2_pt = student_output_p2_pt.chunk(3)
        student_output_p3_cls = student_output_p3_cls.chunk(3)
        student_output_p3_pt = student_output_p3_pt.chunk(3)

        student_output_cls_list = student_output_g_cls + student_output_p1_cls + student_output_p2_cls + student_output_p3_cls
        student_output_pt1_list = student_output_g_pt1 + student_output_p1_pt
        student_output_pt2_list = student_output_g_pt2 + student_output_p2_pt
        student_output_pt3_list = student_output_g_pt3 + student_output_p3_pt

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_output_g_cls = F.softmax((teacher_output_g_cls - self.center_cls) / temp, dim=-1)
        teacher_output_g_pt1 = F.softmax((teacher_output_g_pt1 - self.center_pt1) / temp, dim=-1)
        teacher_output_g_pt2 = F.softmax((teacher_output_g_pt2 - self.center_pt2) / temp, dim=-1)
        teacher_output_g_pt3 = F.softmax((teacher_output_g_pt3 - self.center_pt3) / temp, dim=-1)

        teacher_output_g_cls = teacher_output_g_cls.detach().chunk(2)
        teacher_output_g_pt1 = teacher_output_g_pt1.detach().chunk(2)
        teacher_output_g_pt2 = teacher_output_g_pt2.detach().chunk(2)
        teacher_output_g_pt3 = teacher_output_g_pt3.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_output_g_cls):
            for v in range(len(student_output_cls_list)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_output_cls_list[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        for iq, q in enumerate(teacher_output_g_pt1):
            for v in range(len(student_output_pt1_list)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_output_pt1_list[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        for iq, q in enumerate(teacher_output_g_pt2):
            for v in range(len(student_output_pt2_list)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_output_pt2_list[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        for iq, q in enumerate(teacher_output_g_pt3):
            for v in range(len(student_output_pt3_list)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_output_pt3_list[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output_g)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output_g):
        """
        Update center used for teacher output.
        """
        teacher_output_g_cls, teacher_output_g_pt1, teacher_output_g_pt2, teacher_output_g_pt3 = teacher_output_g

        batch_center_cls = torch.sum(teacher_output_g_cls, dim=0, keepdim=True)
        dist.all_reduce(batch_center_cls)
        batch_center_cls = batch_center_cls / (len(teacher_output_g_cls) * dist.get_world_size())

        # ema update
        self.center_cls = self.center_cls * self.center_momentum + batch_center_cls * (1 - self.center_momentum)

        # part 1
        batch_center_pt1 = torch.sum(teacher_output_g_pt1, dim=0, keepdim=True)
        dist.all_reduce(batch_center_pt1)
        batch_center_pt1 = batch_center_pt1 / (len(teacher_output_g_pt1) * dist.get_world_size())

        # ema update
        self.center_pt1 = self.center_pt1 * self.center_momentum + batch_center_pt1 * (1 - self.center_momentum)

        # part 2
        batch_center_pt2 = torch.sum(teacher_output_g_pt2, dim=0, keepdim=True)
        dist.all_reduce(batch_center_pt2)
        batch_center_pt2 = batch_center_pt2 / (len(teacher_output_g_pt2) * dist.get_world_size())

        # ema update
        self.center_pt2 = self.center_pt2 * self.center_momentum + batch_center_pt2 * (1 - self.center_momentum)

        # part 3
        batch_center_pt3 = torch.sum(teacher_output_g_pt3, dim=0, keepdim=True)
        dist.all_reduce(batch_center_pt3)
        batch_center_pt3 = batch_center_pt3 / (len(teacher_output_g_pt3) * dist.get_world_size())

        # ema update
        self.center_pt3 = self.center_pt3 * self.center_momentum + batch_center_pt3 * (1 - self.center_momentum)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--start-epochs', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('-d', '--dataset', type=str, default='market1501',  # market1501, msmt17_v2
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--self-norm', default=True)

    # Multi-crop parameters
    parser.add_argument('--crop_height', default=128, type=int, help="""Height of crop image""")
    parser.add_argument('--crop_width', default=64, type=int, help="""Width of crop image""")
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
            Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
            recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=9, help="""Number of small
            local views to generate. Set this parameter to 0 to disable multi-crop training.
            When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.1, 0.8),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
            Used for small local view cropping of multi-crop.""")

    parser.add_argument('-a', '--arch', type=str, default='vit_small',
                        choices=models.names())
    parser.add_argument('--drop-path-rate', type=float, default=0.3)
    parser.add_argument('--hw-ratio', type=int, default=2)
    parser.add_argument('--conv-stem', action="store_true")
    parser.add_argument('--feat-fusion', type=str, default='cat')
    parser.add_argument('--multi-neck', action="store_true")
    parser.add_argument('--out_dim', default=8192, type=int, help="""Dimensionality of
            the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
                        help="""Whether or not to weight normalize the last layer of the DINO head.
            Not normalizing leads to better performance but can make the training unstable.
            In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
            of input square patches - default 16 (for 16x16 patches). Using smaller
            values leads to better performance but requires more memory. Applies only
            for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
            mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
            Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
            of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
            starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
                        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
            distributed training; see https://pytorch.org/docs/stable/distributed.html""")


    parser.add_argument('-pp', '--pretrained-path', type=str,
                        default='/home/andy/ICASSP_data/pretrain/PASS/pass_vit_small_full.pth')
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='/home/andy/ICASSP_data/data/')
    parser.add_argument('--pet-data-dir', type=str, metavar='PATH',
                        default='/home/andy/ICASSP_data/data/PetImages')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default='./log/cluster_contrast_reid/msmt17_v2/pass_vit_small_cat_singleneck')
    parser.add_argument('--output_dir', default='/home/andy/ICASSP_data/pretrain/PASS/', type=str, help='Path to save logs and checkpoints.')

    main()
