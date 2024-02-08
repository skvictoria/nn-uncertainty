import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.utils import save_image

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from utils import *
from RISE import RISE

from PIL import Image
import pandas as pd

def normalize_image(image):
    # 이미지의 최소값과 최대값을 구함
    min_val = np.min(image)
    max_val = np.max(image)
    
    # 이미지 정규화: (image - min_val) / (max_val - min_val)
    # 최소값을 빼고 범위를 나누어 [0, 1] 범위로 조정
    normalized_image = (image - min_val) / (max_val - min_val) if max_val > min_val else image - min_val
    
    return normalized_image

def calculate_iou(image1, image2):
    image1_normalized = normalize_image(image1)
    image2_normalized = normalize_image(image2)

    image1_normalized = np.where(image1_normalized > np.mean(image1_normalized), 1, 0)
    image2_normalized = np.where(image2_normalized > np.mean(image2_normalized), 1, 0)

    # 교집합: 두 이미지 픽셀 값의 최소값을 사용
    intersection = np.minimum(image1_normalized, image2_normalized)
    
    # 합집합: 두 이미지 픽셀 값의 최대값을 사용
    union = np.maximum(image1_normalized, image2_normalized)
    
    # 합집합이 0인 픽셀을 제외하고 IOU 계산
    non_zero_union = union > 0
    iou_score = np.sum(intersection[non_zero_union]) / np.sum(union[non_zero_union])
    
    return iou_score

def calculate_snr(um):
    mu = np.mean(um)
    sigma = np.std(um)
    snr = mu / sigma
    return snr

# def calculate_log_likelihood(image1, image2):
#     # normalize images
#     image1_normalized = normalize_image(image1)
#     image2_normalized = normalize_image(image2)

#     image1_normalized = np.where(image1_normalized > np.mean(image1_normalized), 1, 0)
#     image2_normalized = np.where(image2_normalized > np.mean(image2_normalized), 1, 0)


#     epsilon = 1e-12
#     log_likelihood = np.sum(image1_normalized * np.log(image2_normalized + epsilon) + 
#                             (1 - image1_normalized) * np.log(1 - image2_normalized + epsilon))
#     return log_likelihood

def calculate_log_likelihood(image1, image2):
    # 이미지 정규화
    image1_normalized = normalize_image(image1)
    image2_normalized = normalize_image(image2)
    
    # 이미지의 픽셀 값 차이를 기반으로 로그 가능도 계산
    epsilon = 1e-12
    log_likelihood = -np.sum((image1_normalized - image2_normalized) ** 2)# / (2 * epsilon ** 2)
    
    return log_likelihood

def find_class_row(class_name, filename='/home/sophia/nn-uncertainty/synset_words.txt'):
    with open(filename, 'r') as file:
        for i, line in enumerate(file):
            if class_name in line:
                return i
    return -1

os.environ["CUDA_VISIBLE_DEVICES"]="1"

dataset = datasets.ImageFolder('/data/datasets/ImageNet/val/', preprocess)

# This example only works with batch size 1. For larger batches see RISEBatch in explanations.py.
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False,
    num_workers=8, pin_memory=True, sampler=RangeSampler(range(14000, 15000)))

## Load models
model = models.resnet50(True)
model = nn.Sequential(model, nn.Softmax(dim=1))
model = model.eval()
model = model.cuda()

for p in model.parameters():
    p.requires_grad = False
model = nn.DataParallel(model)
explainer = RISE(model, (224, 224), 250)
explainer.load_masks('/home/sophia/nn-uncertainty/evaluation/masks.npy')
explanations = np.load('/home/sophia/nn-uncertainty/evaluation/exp_14000-14999.npy', allow_pickle=True)

directory = '/home/sophia/nn-uncertainty'
files = [file for file in os.listdir(directory) if file.startswith('uncertain_')]

sorted_files = sorted(files)

list_result_false = []
list_result_true = []
batch_results_true = pd.DataFrame(columns=['ith', 'Accuracy', 'IOU', 'SNR', 'LogLikelihood'])
batch_results_false = pd.DataFrame(columns=['ith', 'Accuracy', 'IOU', 'SNR', 'LogLikelihood'])
i = 0
for single_npy, (img, data_classes) in zip(sorted_files, data_loader):
    original_prob, original_class = torch.max(model(img.cuda()), dim=1)
    original_prob, original_class = original_prob[0].item(), original_class[0].item()
    npy_uncertainty = np.load(os.path.join(directory, single_npy), allow_pickle=True)
    
    
    iou_score = calculate_iou(explanations[i], npy_uncertainty)
    snr_value = calculate_snr(npy_uncertainty)
    log_likelihood_value = calculate_log_likelihood(explanations[i], npy_uncertainty)

    if(find_class_row(data_loader.dataset.classes[data_classes]) == original_class):
        list_result_true.append([i, original_prob*100, iou_score, snr_value, log_likelihood_value])

    else:
        list_result_false.append([i, original_prob*100, iou_score, snr_value, log_likelihood_value])

    tensor_imshow(img[0])
    plt.imshow(npy_uncertainty, cmap='jet', alpha=0.5)
    plt.title('acc {:.2f}, iou {:.2f}, snr {:.2f}, log {:.2f}'.format(100*original_prob, iou_score, snr_value, log_likelihood_value))
    plt.savefig("/home/sophia/nn-uncertainty/quantitative/modelagnostic_variance_{:04d}.png".format(i+14050))
    plt.close()

    tensor_imshow(img[0])
    plt.imshow(explanations[i], cmap='jet', alpha=0.5)
    plt.title('{:.2f}% , original {} -> {}'.format(100*original_prob, get_class_name(find_class_row(data_loader.dataset.classes[data_classes])), get_class_name(original_class)))
    plt.savefig("/home/sophia/nn-uncertainty/quantitative/modelagnostic_originalexplainability_{:04d}.png".format(i+14050))
    plt.close()
    i+=1
batch_results_true = pd.concat([batch_results_true, pd.Series(list_result_true)])
batch_results_false = pd.concat([batch_results_false, pd.Series(list_result_false)])
batch_results_true.to_csv(f'/home/sophia/nn-uncertainty/evaluation/metrics_results_batch_true.csv', index=False)
batch_results_false.to_csv(f'/home/sophia/nn-uncertainty/evaluation/metrics_results_batch_false.csv', index=False)
