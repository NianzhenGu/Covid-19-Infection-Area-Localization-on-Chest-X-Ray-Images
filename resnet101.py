import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import argparse
import numpy as np
import os
from os.path import join
import time
import matplotlib.pyplot as plt
import dataset
from utils import calculateIoU
import sys
import segmentation_models_pytorch as smp
np.set_printoptions(threshold=sys.maxsize)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="Classifier train")
parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
parser.add_argument("--save_dir", type=str, default="Out_resnet50", help="folder to save the test results")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
parser.add_argument("--step", type=int, default=30, help="Learning rate decay every n epochs")
parser.add_argument("--reduce", type=float, default=0.5, help="Learning rate decay")
parser.add_argument("--max_epoch", type=int, default=101, help="Maximum epoch for training")
parser.add_argument("--num_cp", type=int, default=20, help="Save point")
parser.add_argument("--num_snapshot", type=int, default=1, help="Draw loss image")
parser.add_argument("--resume_epoch", type=int, default=0, help="resume from checkpoint epoch")
parser.add_argument("--trainData_path", type=str, default="/home/comp/e9251130/code/COMP4026_project/Infection Segmentation Data/Train", help="Dataset file for training")
parser.add_argument("--valData_path", type=str, default="/home/comp/e9251130/code/COMP4026_project/Infection Segmentation Data/Val", help="Dataset file for training")
#parser.add_argument("--testData_path", type=str, default="/home/comp/e9251130/code/VBB/data/bacteria-only", help="Dataset file for training")
parser.add_argument("--img_name", type=str, default='/images/*')
parser.add_argument("--mask_name", type=str, default='/infection masks/*')

opt = parser.parse_args()
print(opt)


def main():
     # generate save folder
    if not os.path.exists(opt.save_dir):
        os.mkdir(opt.save_dir)

    # Data loader
    print('===> Loading training & validating datasets')

    train_set = dataset.LoadData(opt.trainData_path, opt.img_name, opt.mask_name)
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle = True)

    # val_set = dataset.LoadData(opt.valData_path, opt.img_name, opt.mask_name)
    # val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle = True)
    print('loaded {} data from {} for training'.format(len(train_loader)*opt.batch_size, opt.trainData_path))
    #print('loaded {} data from {} for validating'.format(len(val_loader)*opt.batch_size, opt.valData_path))

    # Build model
    print("===> Building net")
    model = smp.Unet('resnet50', classes=1, in_channels=1, activation=None, encoder_weights='imagenet').to(device)

    # model dir
    model_dir = 'checkpoint_resnet50'  
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)


    print("===> setting optimizer")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.reduce)
    losslogger = defaultdict(list)
    IoUlogger = defaultdict(list)

    # optionally resume from a checkpoint
    if opt.resume_epoch:
        resume_path = join(model_dir, 'model_epoch_{}.pth'.format(opt.resume_epoch))
        if os.path.isfile(resume_path):
            print("==>loading checkpoint 'epoch{}'".format(resume_path))
            checkpoint = torch.load(resume_path)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            losslogger = checkpoint['losslogger']
            IoUlogger = checkpoint['IoUlogger']
        else:
            print("==> no model found at 'epoch{}'".format(opt.resume_epoch))

    # Training the model
    print("===> Training")

    for epoch in range(0, opt.max_epoch):
        print('Epoch: ', epoch)
        start_time = time.time()
        model.train()
        loss_count = 0

        for i, (img, gt_mask, _, _) in enumerate(train_loader):

            img = img.unsqueeze(1)
            pred_mask = model(img.to(device))
            loss = Loss(pred_mask, gt_mask.to(device))
            loss_count += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(scheduler.get_last_lr()[0])
        losslogger['epoch'].append(epoch)
        losslogger['loss'].append(loss_count.detach().cpu().numpy() / len(train_loader))


        #sys.exit()


        # print("===> evaluating")
        # model.eval()

        # total_IoU = 0
        # sig = nn.Sigmoid()

        # with torch.no_grad():
        #     for i , (img, gt_mask) in enumerate(val_loader):
        #         img = img.unsqueeze(1)
        #         pred_mask = model(img.to(device))
        #         pred_mask = sig(pred_mask)
        #         pred_mask = pred_mask.squeeze(1).detach().cpu().numpy()
        #         # np.savetxt('gt.txt', gt_mask.squeeze(1).detach().cpu().numpy()[0], delimiter=',')
        #         # np.savetxt('pred.txt', pred_mask[0], delimiter=',')
        #         IoU = calculateIoU(gt_mask.squeeze(1).detach().cpu().numpy(), pred_mask)

        #         total_IoU += IoU

        #     IoUlogger['epoch'].append(epoch)
        #     IoUlogger['IoU'].append(total_IoU / len(val_loader))

        # checkpoint
        if epoch % opt.num_cp == 0:
            model_save_path = join(model_dir, "model_epoch_{}.pth".format(epoch))
            state = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'losslogger': losslogger, 'IoUlogger': IoUlogger}
            torch.save(state, model_save_path)
            print("checkpoint saved to {}".format(model_save_path))

         # loss snapshot
        if epoch % opt.num_snapshot == 0:
            plt.figure()
            plt.title('loss')
            plt.plot(losslogger['epoch'], losslogger['loss'])
            plt.savefig(join(opt.save_dir, 'loss.jpg'))
            plt.close()

            # plt.figure()
            # plt.title('IoU')
            # plt.plot(IoUlogger['epoch'], IoUlogger['IoU'])
            # plt.savefig(join(opt.save_dir, 'IoU.jpg'))

            plt.cla()
            plt.close("all")

        print(epoch, ' epoch finished.')
        print('loss: ', losslogger['loss'][epoch])
        #print('IoU: ', IoUlogger['IoU'][epoch])
        print("--- %s seconds ---" % (time.time() - start_time))


def Loss(gt, pred):
    gt = gt.squeeze(1)
    pred = pred.squeeze(1)

    # print('gt:', gt.shape)
    # print('pred:', pred.shape)
    BCE = nn.BCEWithLogitsLoss()

    return BCE(gt, pred)



if __name__ == '__main__':
    main()