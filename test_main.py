import argparse
import torch
import torch.nn as nn
import dataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from utils import get
import math
import os
import numpy as np
import Sandwitch


parser = argparse.ArgumentParser(description="main")
parser.add_argument("--pre_trained_model_path", type=str, default="/home/comp/e9251130/code/COMP4026_project/checkpoint2/model_epoch_100.pth")
parser.add_argument("--testData_path", type=str, default="/home/comp/e9251130/code/COMP4026_project/Infection Segmentation Data/Normal", help="Dataset file for testing")
parser.add_argument("--img_name", type=str, default='/images/*')
parser.add_argument("--mask_name", type=str, default='/infection masks/*')
parser.add_argument("--save_dir", type=str, default='result_normal')
opt = parser.parse_args()
print(opt)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(opt.save_dir):
    os.makedirs(opt.save_dir)


def test():
    print('===> Loading pre-trained model')
    model = Sandwitch.u_shape().to(device)
    #model= smp.Unet('resnet50', classes=1, in_channels=1).to(device)
    resume_path = opt.pre_trained_model_path
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['model'], strict=True)
    print('loaded pretrained model'.format(resume_path))

    print('===> Loading training & validating datasets')
    test_set = dataset.LoadData(opt.testData_path, opt.img_name, opt.mask_name)
    test_loader = DataLoader(test_set, batch_size=1, shuffle = True)
    print('loaded {} data from {} for testing'.format(len(test_loader), opt.testData_path))

    IoU_list = []
    DSC_list = []
    acc_list = []
    recall_list = []
    precision_list = []
    f1_list = []
    path_list = []

    sig = nn.Sigmoid()
    count = 0
    model.eval()
    with torch.no_grad():
        for i, (img, gt_mask, imgpath, maskpath) in enumerate(test_loader):
            print(i)
            # print(imgpath)
            # print(maskpath)
            path_list.append(imgpath)
            img = img.unsqueeze(1)
            pred_mask = model(img.to(device))
            pred_mask = sig(pred_mask)
            pred_mask = pred_mask.squeeze(1).squeeze(0).detach().cpu().numpy()
            gt_mask = gt_mask.squeeze(0).detach().cpu().numpy()

            # print(gt_mask.shape)
            # print(pred_mask.shape)

            tn, fp, fn, tp = get(gt_mask, pred_mask)

            if fp == 0 and tp == 0:
                count +=1 

    #         IoU = tp/(tp+fp+fn)
    #         IoU_list.append(IoU)

    #         DSC = 2*tp/(2*tp+fp+fn) 
    #         DSC_list.append(DSC)

    #         acc = (tp+tn)/(tp+tn+fp+fn)
    #         acc_list.append(acc)

    #         precision = tp/(tp+fp)
    #         precision_list.append(precision)

    #         recall = tp/(tp+fn)
    #         recall_list.append(recall)

    #         f1 = 2*precision*recall/(precision+recall)
    #         f1_list.append(f1)

    # valid_precision = []
    # valid_f1 = []

    # for i in range(len(precision_list)):
    #     if math.isnan(precision_list[i]) == False:
    #         valid_precision.append(precision_list[i])
    #     if math.isnan(f1_list[i]) == False:
    #         valid_f1.append(f1_list[i])

    # print('Average IoU: ', sum(IoU_list)/len(IoU_list))
    # print('Average DSC: ', sum(DSC_list)/len(DSC_list))
    # print('Average Acc: ', sum(acc_list)/len(acc_list))
    # print('Average precision: ', sum(valid_precision)/len(valid_precision))
    # print('Average recall: ', sum(recall_list)/len(recall_list))
    # print('Average F1: ', sum(valid_f1)/len(valid_f1))

    # print('min IoU: ', min(IoU_list))
    # print('min DSC: ', min(DSC_list))
    # print('min Acc: ', min(acc_list))
    # print('min precision: ', min(precision_list))
    # print('min recall: ', min(recall_list))
    # print('min F1: ', min(f1_list))

    # print('max IoU: ', max(IoU_list))
    # print('max DSC: ', max(DSC_list))
    # print('max Acc: ', max(acc_list))
    # print('max precision: ', max(precision_list))
    # print('max recall: ', max(recall_list))
    # print('max F1: ', max(f1_list))
    
    
    # with open(os.path.join(opt.save_dir,'IoU.txt'), 'w') as f:
    #     f.write('IoU: ' + '\n')
    #     for i in range(len(IoU_list)):
    #         f.write(str(IoU_list[i])+'\n')
    #     f.write('\n')

    #     f.write('DSC: ' + '\n')
    #     for i in range(len(DSC_list)):
    #         f.write(str(DSC_list[i])+'\n')
    #     f.write('\n')

    #     f.write('Acc: ' + '\n')
    #     for i in range(len(acc_list)):
    #         f.write(str(acc_list[i])+'\n')
    #     f.write('\n')

    #     f.write('Precision: ' + '\n')
    #     for i in range(len(precision_list)):
    #         f.write(str(precision_list[i])+'\n')
    #     f.write('\n')

    #     f.write('Recall: ' + '\n')
    #     for i in range(len(recall_list)):
    #         f.write(str(recall_list[i])+'\n')
    #     f.write('\n')

    #     f.write('F1: ' + '\n')
    #     for i in range(len(f1_list)):
    #         f.write(str(f1_list[i])+'\n')
    #     f.write('\n')
    

    return IoU_list, path_list, count

def getWorstThree(list1, list2, N):
    IoU_list = []
    name_list = []
    for i in range(0, N):
        min1 = 1
        for j in range(len(list1)): 
            if list1[j] < min1: 
                min1 = list1[j]
        index = list1.index(min1) 
        IoU_list.append(min1)
        name_list.append(list2[index])
        list1.remove(min1)
        list2.remove(list2[index])

    return IoU_list, name_list

def getBestThree(list1, list2, N):
    IoU_list = []
    name_list = []
    for i in range(0, N):
        max = 0
        for j in range(len(list1)): 
            if list1[j] > max: 
                max = list1[j]
        index = list1.index(max) 
        IoU_list.append(max)
        name_list.append(list2[index])
        list1.remove(max)
        list2.remove(list2[index])

    return IoU_list, name_list
        

if __name__ == '__main__':
    IoU_list, path_list, count = test()

    print(count)

    # worstThree, name_worst = getWorstThree(IoU_list, path_list, 3)
    # bestThree, name_best = getBestThree(IoU_list, path_list, 30)

    # print('worstThree: ', worstThree)
    # print('index: ', name_worst)
    # print('bestThree: ', bestThree)
    # print('index: ', name_best[20:])




