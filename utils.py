''' 
IoU = TP/(TP+FP+FN)
Accuracy = (TP+TN)/(TP+TN+FP+FN)
Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
F1-score = 2*precision*recall/(precision+recall)
'''
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
from torchsummary import summary
import Sandwitch
import torch
import segmentation_models_pytorch as smp


def get(gt, pred):
    gt_list = []
    pred_list = []
    for i in range(256):
        for j in range(256):
            gt_list.append(int(gt[i][j]))
            if pred[i][j] > 0.5:
                pred_list.append(1)
            else:
                pred_list.append(0)
    #print(pred_list)
    tn, fp, fn, tp = confusion_matrix(gt_list, pred_list, labels=[0, 1]).ravel()
    #assert tn > 0 or fp > 0 or fn > 0 or tp > 0
    return tn, fp, fn, tp

def calculateIoU(gt, pred):
    gt = np.round_(gt)
    pred = np.round_(pred)
    # print(gt[0])
    # print(pred[0])

    # np.savetxt('gt', gt[0], delimiter=',')
    # np.savetxt('pred', pred[0], delimiter=',')

    # print(gt.shape)
    # print(pred.shape)

    tp = np.sum(np.logical_and(pred == 1, gt == 1))
    tn = np.sum(np.logical_and(pred == 0, gt == 0))
    fp = np.sum(np.logical_and(pred == 1, gt == 0))
    fn = np.sum(np.logical_and(pred == 0, gt == 1))

    # print(tn)
    # print(fp)
    # print(fn)
    # print(tp)

    return tp/(tp+fp+fn)

# def calculateAcc(gt, pred):
#     tn, fp, fn, tp = get(gt, pred)
#     return (tp+tn)/(tp+tn+fp+fn)

# def calculatePrecision(gt, pred):
#     tn, fp, fn, tp = get(gt, pred)
#     return tp/(tp+fp)

# def calculateRecall(gt, pred):
#     tn, fp, fn, tp = get(gt, pred)
#     return tp/(tp+fn)



def getSummary():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = Sandwitch.u_shape().to(device)
    # summary(model, (1, 256, 256), batch_size=32)

    model2 = smp.Unet('resnet101', classes=1, in_channels=1, activation=None).to(device)
    summary(model2, (1, 256, 256), batch_size=32)



if __name__ == '__main__':
    getSummary()