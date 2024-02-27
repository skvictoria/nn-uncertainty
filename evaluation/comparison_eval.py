import os
import numpy as np
import sys

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from utils import *

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

def find_class_row(class_name, filename='/home/seulgi/work/nn-uncertainty/synset_words.txt'):
    with open(filename, 'r') as file:
        for i, line in enumerate(file):
            if class_name in line:
                return i
    return -1


### TODO : change these paths
os.environ["CUDA_VISIBLE_DEVICES"]="1"
image_dir = ['/home/seulgi/work/data/random-sampled-imagenet', range(0, 500)]
model_name = 'scorecam'
explanation_dir = '/home/seulgi/work/data/explainers/{}-exp_00000-00499.npy'.format(model_name)
mask_dir = '/home/seulgi/work/data/{}masks'.format(model_name)
save_csv_file_path = '/home/seulgi/work/nn-uncertainty/evaluation/csv'

dataset = datasets.ImageFolder(image_dir[0], preprocess)

# This example only works with batch size 1. For larger batches see RISEBatch in explanations.py.
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False,
    num_workers=8, pin_memory=True, sampler=RangeSampler(image_dir[1]))

## Load models
model = models.resnet50(True)
model = nn.Sequential(model, nn.Softmax(dim=1))
model = model.eval()
model = model.cuda()

for p in model.parameters():
    p.requires_grad = False
model = nn.DataParallel(model)

# Load explanation
explanations = np.load(explanation_dir, allow_pickle=True)

# Load uncertainty masks
uncertain_files = [file for file in os.listdir(mask_dir) if file.startswith('uncertain_')]
diff_files = [file for file in os.listdir(mask_dir) if file.startswith('diff_')]
same_files = [file for file in os.listdir(mask_dir) if file.startswith('same_')]
uncertain_files = sorted(uncertain_files)
diff_files = sorted(diff_files)
same_files = sorted(same_files)

list_result_false = []
list_result_true = []
batch_results_true = pd.DataFrame(columns=['ith', 'Accuracy', 'IOU', 'SNR', 'LogLikelihood'])
batch_results_false = pd.DataFrame(columns=['ith', 'Accuracy', 'IOU', 'SNR', 'LogLikelihood'])

## whole uncertainty
i = 0
for single_npy, (img, data_classes) in zip(uncertain_files, data_loader):
    logit = model(img.cuda())
    original_prob_cuda, original_class = torch.max(logit, dim=1)
    original_prob, original_class = original_prob_cuda[0].item(), original_class[0].item()
    npy_uncertainty = np.load(os.path.join(mask_dir, single_npy), allow_pickle=True)
    
    iou_score = calculate_iou(explanations[i], npy_uncertainty)
    snr_value = calculate_snr(npy_uncertainty)
    target = torch.from_numpy(np.asarray([find_class_row(data_loader.dataset.classes[data_classes])])).cuda()
    
    log_likelihood_value = -F.nll_loss(logit, target).data.cpu().numpy()
    if(find_class_row(data_loader.dataset.classes[data_classes]) == original_class):
        #list_result_true.append([i, original_prob*100, iou_score, snr_value, log_likelihood_value])
        new_row_true = pd.DataFrame([{'ith': i, 'Accuracy': original_prob*100, 'IOU': iou_score, 'SNR': snr_value, 'LogLikelihood': log_likelihood_value}])
        batch_results_true = pd.concat([batch_results_true, new_row_true], ignore_index=True)

    else:
        #list_result_false.append([i, original_prob*100, iou_score, snr_value, log_likelihood_value])
        new_row_false = pd.DataFrame([{'ith': i, 'Accuracy': original_prob*100, 'IOU': iou_score, 'SNR': snr_value, 'LogLikelihood': log_likelihood_value}])
        batch_results_false = pd.concat([batch_results_false, new_row_false], ignore_index=True)

    i+=1
batch_results_true.to_csv(save_csv_file_path+'/'+ model_name+'_true.csv', index=False)
batch_results_false.to_csv(save_csv_file_path+'/'+ model_name+'_false.csv', index=False)



list_result_false = []
list_result_true = []
batch_results_true = pd.DataFrame(columns=['ith', 'Accuracy', 'IOU', 'SNR', 'LogLikelihood'])
batch_results_false = pd.DataFrame(columns=['ith', 'Accuracy', 'IOU', 'SNR', 'LogLikelihood'])

## whole uncertainty
i = 0
for single_npy, (img, data_classes) in zip(diff_files, data_loader):
    logit = model(img.cuda())
    original_prob_cuda, original_class = torch.max(logit, dim=1)
    original_prob, original_class = original_prob_cuda[0].item(), original_class[0].item()
    npy_uncertainty = np.load(os.path.join(mask_dir, single_npy), allow_pickle=True)
    
    iou_score = calculate_iou(explanations[i], npy_uncertainty)
    snr_value = calculate_snr(npy_uncertainty)
    target = torch.from_numpy(np.asarray([find_class_row(data_loader.dataset.classes[data_classes])])).cuda()
    
    log_likelihood_value = -F.nll_loss(logit, target).data.cpu().numpy()
    if(find_class_row(data_loader.dataset.classes[data_classes]) == original_class):
        #list_result_true.append([i, original_prob*100, iou_score, snr_value, log_likelihood_value])
        new_row_true = pd.DataFrame([{'ith': i, 'Accuracy': original_prob*100, 'IOU': iou_score, 'SNR': snr_value, 'LogLikelihood': log_likelihood_value}])
        batch_results_true = pd.concat([batch_results_true, new_row_true], ignore_index=True)

    else:
        #list_result_false.append([i, original_prob*100, iou_score, snr_value, log_likelihood_value])
        new_row_false = pd.DataFrame([{'ith': i, 'Accuracy': original_prob*100, 'IOU': iou_score, 'SNR': snr_value, 'LogLikelihood': log_likelihood_value}])
        batch_results_false = pd.concat([batch_results_false, new_row_false], ignore_index=True)

    i+=1
batch_results_true.to_csv(save_csv_file_path+'/'+ model_name+'_true_diff.csv', index=False)
batch_results_false.to_csv(save_csv_file_path+'/'+ model_name+'_false_diff.csv', index=False)


list_result_false = []
list_result_true = []
batch_results_true = pd.DataFrame(columns=['ith', 'Accuracy', 'IOU', 'SNR', 'LogLikelihood'])
batch_results_false = pd.DataFrame(columns=['ith', 'Accuracy', 'IOU', 'SNR', 'LogLikelihood'])

## whole uncertainty
i = 0
for single_npy, (img, data_classes) in zip(same_files, data_loader):
    logit = model(img.cuda())
    original_prob_cuda, original_class = torch.max(logit, dim=1)
    original_prob, original_class = original_prob_cuda[0].item(), original_class[0].item()
    npy_uncertainty = np.load(os.path.join(mask_dir, single_npy), allow_pickle=True)
    
    iou_score = calculate_iou(explanations[i], npy_uncertainty)
    snr_value = calculate_snr(npy_uncertainty)
    target = torch.from_numpy(np.asarray([find_class_row(data_loader.dataset.classes[data_classes])])).cuda()
    
    log_likelihood_value = -F.nll_loss(logit, target).data.cpu().numpy()
    if(find_class_row(data_loader.dataset.classes[data_classes]) == original_class):
        #list_result_true.append([i, original_prob*100, iou_score, snr_value, log_likelihood_value])
        new_row_true = pd.DataFrame([{'ith': i, 'Accuracy': original_prob*100, 'IOU': iou_score, 'SNR': snr_value, 'LogLikelihood': log_likelihood_value}])
        batch_results_true = pd.concat([batch_results_true, new_row_true], ignore_index=True)

    else:
        #list_result_false.append([i, original_prob*100, iou_score, snr_value, log_likelihood_value])
        new_row_false = pd.DataFrame([{'ith': i, 'Accuracy': original_prob*100, 'IOU': iou_score, 'SNR': snr_value, 'LogLikelihood': log_likelihood_value}])
        batch_results_false = pd.concat([batch_results_false, new_row_false], ignore_index=True)

    i+=1
batch_results_true.to_csv(save_csv_file_path+'/'+ model_name+'_true_same.csv', index=False)
batch_results_false.to_csv(save_csv_file_path+'/'+ model_name+'_false_same.csv', index=False)