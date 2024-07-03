import argparse
import Sandwitch
import torch
import torch.nn as nn
import dataset
from torch.utils.data import DataLoader
from utils import get
import math
import cv2
import os

parser = argparse.ArgumentParser(description="main")
parser.add_argument("--pre_trained_model_path", type=str, default="/home/comp/e9251130/code/COMP4026_project/checkpoint2/model_epoch_100.pth")
parser.add_argument("--testData_path", type=str, default="/home/comp/e9251130/code/COMP4026_project/Infection Segmentation Data/Test", help="Dataset file for testing")
parser.add_argument("--img_name", type=str, default='/images/*')
parser.add_argument("--mask_name", type=str, default='/infection masks/*')
opt = parser.parse_args()
print(opt)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def draw(imgpath, maskpath, savedir):
    print('===> Loading pre-trained model')
    model = Sandwitch.u_shape().to(device)
    resume_path = opt.pre_trained_model_path
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['model'], strict=True)
    print('loaded pretrained model'.format(resume_path))

    sig = nn.Sigmoid()

    img = cv2.imread(imgpath, 0)
    img_tensor = torch.from_numpy(img/255.).float()
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.unsqueeze(0)

    mask = cv2.imread(maskpath, 0)
    mask_tensor = torch.from_numpy(mask/255.).float()

    print(img_tensor.shape)

    model.eval()
    with torch.no_grad():
        pred_mask = model(img_tensor.to(device))
        pred_mask = sig(pred_mask)
        pred_mask = pred_mask.squeeze(1).squeeze(0).detach().cpu().numpy()
        gt_mask = mask_tensor.detach().cpu().numpy()

        tn, fp, fn, tp = get(gt_mask, pred_mask)
        IoU = tp/(tp+fp+fn)

        for i in range(256):
            for j in range(256):
                if pred_mask[i][j] > 0.5:
                    pred_mask[i][j] = 255
                else:
                    pred_mask[i][j] = 0

        print(IoU)
        return pred_mask, mask


if __name__ == '__main__':

    images = ['/home/comp/e9251130/code/COMP4026_project/Infection Segmentation Data/Test/COVID-19/images/covid_1604.png']

    masks = ['/home/comp/e9251130/code/COMP4026_project/Infection Segmentation Data/Test/COVID-19/infection masks/covid_1604.png']
    savedir = 'test_s'
    for i in range(1):
        pred_mask, gt_mask = draw(images[i], masks[i], savedir)
        cv2.imwrite(os.path.join(savedir,'pred_%d.jpg'%(i+21)), pred_mask)
        cv2.imwrite(os.path.join(savedir,'gt_%d.jpg'%(i+21)), gt_mask)