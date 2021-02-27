import argparse
import hashlib
import os
import time

import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from tensorboardX import SummaryWriter

from dataloaders import make_data_loader
from dataloaders.utils import decode_segmap, decode_seg_map_sequence
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args.resume, args.ft)
        if not args.resume or args.ft:
            self.saver.save_experiment_config(args)
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.expr_dir)
        self.summary_writer = SummaryWriter(self.saver.expr_dir)
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': False}
        self.train_loader, self.val_loader, _, self.num_classes = make_data_loader(args, **kwargs)

        # Define network
        model = DeepLab(backbone=args.backbone,
                        in_channels=args.in_channels,
                        output_stride=args.out_stride,
                        num_classes=self.num_classes,
                        sync_bn=args.sync_bn, 
                        freeze_bn=args.freeze_bn)

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = self.saver.get_previous_best()
        if args.resume:
            if not os.path.isfile(args.resume):
                raise RuntimeError("no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] if not args.ft else 0
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))


        # Define Evaluator
        self.evaluator = Evaluator(self.num_classes)
    
    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        iters_per_epoch = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.summary_writer.add_scalar('train/total_losses vs. iters', loss.item(), i + iters_per_epoch * epoch)

            # Show 10 * 3 inference results each epoch
            if iters_per_epoch <= 10 or (i + 1) % (iters_per_epoch // 10) == 0:
                global_step = i + iters_per_epoch * epoch
                self.summary_writer.add_images('train/original_images', image[:4] * 0.5 + 0.5, global_step) 
                self.summary_writer.add_images('train/groud_truth',
                        decode_seg_map_sequence(target.detach().cpu().numpy()[:4], dataset='suichang_round1'), global_step, dataformats='NCHW')
                self.summary_writer.add_images('train/predicted', 
                        decode_seg_map_sequence(torch.argmax(output, dim=1).detach().cpu().numpy()[:4], dataset='suichang_round1'), global_step, dataformats='NCHW')
        self.summary_writer.add_scalar('train/total_losses vs. epochs', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        self.saver.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filename='last.pth')

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.summary_writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.summary_writer.add_scalar('val/mIoU', mIoU, epoch)
        self.summary_writer.add_scalar('val/Acc', Acc, epoch)
        self.summary_writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.summary_writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        if new_pred > self.best_pred:
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, is_best=True, miou=new_pred)
            self.best_pred = new_pred

    def make_predict(self, args):
        self.model.eval()
        if args.do_eval:
            ckpt_path = os.path.join(self.saver.expr_dir, 'best.pth')
        else:
            ckpt_path = os.path.join(self.saver.expr_dir, 'last.pth')
        checkpoint = torch.load(ckpt_path)
        self.model.module.load_state_dict(checkpoint['state_dict'])
        results_dir = os.path.join(self.saver.expr_dir, 'results')
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)

        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.190, 0.235, 0.222, 0.527), (0.127, 0.122, 0.119, 0.199))
        ])
        test_image_names = os.listdir(args.test_dir)
        for i, test_image_name in enumerate(tqdm(test_image_names)):
            test_image = Image.open(os.path.join(args.test_dir, test_image_name))
            self.summary_writer.add_image('test/ground_truth', torch.tensor(np.asarray(test_image)), global_step=i, dataformats='HWC') 
            test_image = transform(test_image)
            
            test_image = torch.unsqueeze(test_image, 0)
            if args.cuda:
                test_image = test_image.cuda()
            output = self.model(test_image)
            output = torch.argmax(output, dim=1).squeeze() + 1
            output = output.data.cpu().numpy().astype(np.uint8)
            self.summary_writer.add_image('test/predicted', decode_segmap(output, dataset='tiny_suichang_round1'), global_step=i, dataformats='HWC')
            Image.fromarray(output).save(os.path.join(results_dir, test_image_name.replace('.tif', '.png')))

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='xception',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--in-channels', type=int, default=3,
                        help='input image channels, jpeg=3, tif=4')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes', 'suichang_round1', 'tiny_suichang_round1'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: False)')
    parser.add_argument('--workers', type=int, default=0,
                        metavar='N', help='dataloader threads, if workers > 0, some unknown errors will occur!!!')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--use-cuda', action='store_true', default=
                        True, help='whether using CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    
    # About validation  
    parser.add_argument('--do-eval', action='store_true', default=False,
                        help='whether evaluate model during training')
    parser.add_argument('--eval-interval', type=int, default=5,
                        help='evaluation interval (default: 1)')
    parser.add_argument('--test-dir', type=str, default=None,
                        help='test images directory')
    
    args = parser.parse_args()
    print(args)

    # Determine whether using cuda.
    args.cuda = torch.cuda.is_available() and args.use_cuda
    if args.cuda:
        print('Use CUDA support!')
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if trainer.args.do_eval and ((epoch + 1) % args.eval_interval == 0 or epoch + 1 == args.epochs):
            trainer.validation(epoch)

    if args.test_dir:
        trainer.make_predict(args)

    trainer.summary_writer.close()

if __name__ == "__main__":
   main()
