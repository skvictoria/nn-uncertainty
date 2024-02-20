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

from utils import *
from RISE import RISE
import pandas as pd

EXPLAIN_FOR_THE_FIRST_TIME = 1
os.environ["CUDA_VISIBLE_DEVICES"]="1"
## TODO: model change
## image dir
image_dir = ['/data/datasets/ImageNet/val/', range(15000, 15020)]
explanation_dir = '/home/sophia/nn-uncertainty/mask_generator/explainers'
mask_dir = '/home/sophia/nn-uncertainty/mask_generator/masks'
output_dir = '/home/sophia/nn-uncertainty/0220-result' # for visualization
csv_output_dir = '/home/sophia/nn-uncertainty/evaluation/0220-csv'
#image_dir = '/home/seulgi/imagenet-o'
#args.range = range(0, 1000)
#args.range = range(1500,2000)

def example(img, top_k=3, save_path='output.png'):
    saliency = explainer(img.cuda()).cpu().numpy()
    p, c = torch.topk(model(img.cuda()), k=top_k)
    p, c = p[0], c[0]
    
    plt.figure(figsize=(10, 5*top_k))
    for k in range(top_k):
        plt.subplot(top_k, 2, 2*k+1)
        plt.axis('off')
        plt.title('{:.2f}% {}'.format(100*p[k], get_class_name(c[k])))
        tensor_imshow(img[0])

        plt.subplot(top_k, 2, 2*k+2)
        plt.axis('off')
        plt.title(get_class_name(c[k]))
        tensor_imshow(img[0])
        
        plt.imshow(saliency[c[k]], cmap='jet', alpha=0.5)
        plt.colorbar(fraction=0.046, pad=0.04)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def explain_all(data_loader, explainer):
    # Get all predicted labels first
    target = np.empty(len(data_loader), dtype=int)
    for i, (img, _) in enumerate(tqdm(data_loader, total=len(data_loader), desc='Predicting labels')):
        p, c = torch.max(model(img.cuda()), dim=1)
        target[i] = c[0]

    # Get saliency maps for all images in val loader
    explanations = np.empty((len(data_loader), *args.input_size))
    for i, (img, _) in enumerate(tqdm(data_loader, total=len(data_loader), desc='Explaining images')):
        saliency_maps = explainer(img.cuda())
        explanations[i] = saliency_maps[target[i]].cpu().numpy()
    return explanations

def apply_mask(image, mask):
    return image * mask

def invert_masking(image, explanation):
    """
    Apply an inverted mask to the image based on explanation scores.
    
    Args:
    - image: torch.Tensor, the image tensor to mask.
    - explanation: numpy.ndarray, the explanation scores for each pixel.
    """
    # Ensure the explanation is a tensor and normalize it
    explanation = torch.from_numpy(explanation).to(image.device)
    explanation_normalized = explanation / explanation.max()
    
    # Invert the explanation scores
    inverted_explanation = 1 - explanation_normalized
    
    # Apply the inverted explanation as a mask
    masked_image = image * inverted_explanation

    return masked_image

def explainability_masking(image, explanation, mask_size=50, p1=0.5):
    """
    Apply a random fractional mask to the image based on explanation.
    Areas with higher explanation scores are more likely to be masked.
    
    Args:
    - image: torch.Tensor, the image tensor to mask.
    - explanation: numpy.ndarray, the explanation scores for each pixel.
    - mask_size: int, the size of the square mask.
    - p1: float, the base probability of masking a given pixel; 
      actual masking probability is p1 adjusted by the explanation score.
    """
    # Normalize explanation scores to range [0, 1]
    normalized_explanation = explanation / explanation.max()
    
    # Create a random mask with values between 0 and 1, adjusted by explanation scores
    random_mask = torch.rand_like(image) * torch.from_numpy(normalized_explanation).float()
    mask = random_mask > p1  # Apply base probability to decide if a pixel should be masked
    
    # Apply the mask to the image
    masked_image = image.clone().float()
    masked_image[mask] = 0  # Mask pixels where mask is True

    return masked_image

def random_masking(image, mask_size=50):
    mask = torch.ones_like(image).float()
    x = np.random.randint(0, image.shape[1] - mask_size)
    y = np.random.randint(0, image.shape[2] - mask_size)
    mask[:, x:x+mask_size, y:y+mask_size] = 0

    mask_for_save = torch.zeros_like(image)
    mask_for_save[:, x:x+mask_size, y:y+mask_size] = 1
    return image.float() * mask, mask_for_save


def random_masking_for_some_fraction(image, mask_size=30, fraction=0.95):
    mask = torch.ones_like(image).float()
    x = np.random.randint(0, image.shape[1] - mask_size)
    y = np.random.randint(0, image.shape[2] - mask_size)
    # Determine the number of points to mask within the square
    total_points = mask_size ** 2
    points_to_mask = int(total_points * fraction)

    # Create a flat array of zeros with the number of points to mask set to one
    flat_mask = torch.zeros(total_points)
    flat_mask[:points_to_mask] = 1
    # Shuffle the mask
    flat_mask = flat_mask[torch.randperm(total_points)]
    # Reshape the mask back to the mask size
    square_mask = flat_mask.view(mask_size, mask_size)

    # Apply the square mask to the selected region in the image
    mask[:, x:x+mask_size, y:y+mask_size] = 1 - square_mask

    mask_for_save = torch.zeros_like(image)
    mask_for_save[:, x:x+mask_size, y:y+mask_size] = square_mask

    ## one more time
    x = np.random.randint(0, image.shape[1] - mask_size)
    y = np.random.randint(0, image.shape[2] - mask_size)
    flat_mask = flat_mask[torch.randperm(total_points)]
    square_mask = flat_mask.view(mask_size, mask_size)

    mask[:, x:x+mask_size, y:y+mask_size] = 1 - square_mask
    mask_for_save[:, x:x+mask_size, y:y+mask_size] = square_mask

    return image.float() * mask, mask_for_save

def random_point_masking(image, num_points=10000):
    # Create a clone of the image to avoid modifying the original image
    cloned_image = image.clone()
    masked_image = cloned_image.float()
    
    # Flatten the image to work with it as a 1D array
    flat_image = masked_image.view(-1)
    
    # Generate random indices
    indices = torch.randperm(flat_image.size(0))[:num_points]
    
    # Set the selected random points to zero (mask them)
    flat_image[indices] = 0
    
    # Reshape the image back to its original shape
    masked_image = flat_image.view_as(cloned_image)
    
    # Create a mask tensor with zeros and set the selected indices to one
    mask = torch.zeros_like(flat_image)
    mask[indices] = 1
    mask = mask.view_as(cloned_image)
    
    return masked_image, mask

def random_fraction_masking(image_original, mask_size=50, p1=0.5):
    """
    Apply a random fractional mask to the image.
    Each pixel in the mask can be in the range [0, 1].
    Args:
    - image: torch.Tensor, the image tensor to mask.
    - mask_size: int, the size of the square mask.
    - p1: float, the probability of masking a given pixel in the mask; 
      this determines the sparsity of the mask.
    """
    image = image_original.clone()
    # Create a random mask with values between 0 and 1
    fractional_mask = torch.rand_like(image) > p1
    fractional_mask = fractional_mask.float()  # Convert boolean mask to float

    # Generate random positions for top-left corner of the mask
    x = np.random.randint(0, image.shape[1] - mask_size)
    y = np.random.randint(0, image.shape[2] - mask_size)

    # Apply the fractional mask to the selected square region
    image[:, x:x+mask_size, y:y+mask_size] *= fractional_mask[:, x:x+mask_size, y:y+mask_size]

    return image.float()

def find_class_row(class_name, filename='/home/sophia/nn-uncertainty/synset_words.txt'):
    with open(filename, 'r') as file:
        for i, line in enumerate(file):
            if class_name in line:
                return i
    return -1

def calculate_iou(image1, image2):
    # 교집합: 두 이미지 픽셀 값의 최소값을 사용
    intersection = np.minimum(image1, image2)
    
    # 합집합: 두 이미지 픽셀 값의 최대값을 사용
    union = np.maximum(image1, image2)
    
    # 합집합이 0인 픽셀을 제외하고 IOU 계산
    non_zero_union = union > 0
    iou_score = np.sum(intersection[non_zero_union]) / np.sum(union[non_zero_union])
    
    return iou_score

def calculate_snr(um):
    mu = np.mean(um)
    sigma = np.std(um)
    snr = mu / sigma
    return snr

def calculate_log_likelihood(image1, image2):
    # normalize images
    image1_normalized = image1 / 255.0
    image2_normalized = image2 / 255.0

    epsilon = 1e-12
    log_likelihood = np.sum(image1_normalized * np.log(image2_normalized + epsilon) + 
                            (1 - image1_normalized) * np.log(1 - image2_normalized + epsilon))
    return log_likelihood

# 배치를 처리하는 함수
def process_batch(data_loader, model, explainer, explanations, batch_number, start_index):
    batch_results_true = pd.DataFrame(columns=['Accuracy', 'IOU', 'SNR', 'LogLikelihood'])
    batch_results_false = pd.DataFrame(columns=['Accuracy', 'IOU', 'SNR', 'LogLikelihood'])

    list_result_true = []
    list_result_false = []
    for i, (img, data_classes) in enumerate(data_loader, start=start_index):
        
        data_classes = data_classes[0].item()
        original_prob, original_class = torch.max(model(img.cuda()), dim=1)
        original_prob, original_class = original_prob[0].item(), original_class[0].item()
        
        # # it means that network is not confident.
        # if(find_class_row(data_loader.dataset.classes[data_classes]) == original_class):
        #     continue

        # ### image save
        # min_val = img[0].min()
        # range_val = img[0].max() - min_val
        # image_tensor = (img[0] - min_val) / range_val
        # image_matrix = (image_tensor*255).type(torch.uint8)
        # image_matrix = image_matrix.permute(1,2,0)
        # image_matrix = Image.fromarray(image_matrix.numpy())
        # image_matrix.save('/home/sophia/voice/images/{}.png'.format(i))

        explainability_list = []
        for random_idx in range(6000):
            # Random_mask_for_save: grayscale version of mask
            randomly_masked_image, random_mask_for_save = random_masking_for_some_fraction(img[0])
            random_mask_for_save = random_mask_for_save.permute(1,2,0).numpy()
            random_mask_for_save = np.dot(random_mask_for_save[...,:3], weights)

            model_output = model(randomly_masked_image.unsqueeze(0))

            chang_prob, chang_class = torch.max(model_output, dim=1)
            chang_prob, chang_class = chang_prob[0].item(), chang_class[0].item()

            #if original_class != chang_class:
            saliency_maps = explainer(randomly_masked_image.unsqueeze(0).cuda()).cpu().numpy()
            p, c = torch.topk(model(img.cuda()), k=1)
            p, c = p[0], c[0]
            saliency_maps = saliency_maps[c[0]]

            saliency_maps = random_mask_for_save * saliency_maps
            explainability_list.append(saliency_maps)

            # probabilities = torch.nn.functional.softmax(model_output, dim=1)
            # entropy = -(probabilities * probabilities.log()).sum(1)
            # entropy = entropy.item()

        if len(explainability_list)==0:
            continue
        pixel_values = np.stack(explainability_list, axis=0)
        

        ## simply adding
        #ones_count = np.sum(pixel_values, axis=0)
        #ones_count = np.sum(pixel_values_for_RISE, axis=0)

        # ## variance
        variance = np.var(pixel_values, axis=0)

        
        ############ image save for simply adding
        # tensor_imshow(img[0])
        # plt.imshow(ones_count, cmap='jet', alpha=0.5)
        # plt.savefig("result-target-diff/modelagnostic_add_{:04d}.png".format(i+14050))
        # plt.close()
        
        iou_score = calculate_iou(explanations[i], variance)

        snr_value = calculate_snr(variance)

        log_likelihood_value = calculate_log_likelihood(explanations[i], variance)

        ############# image save for variance
        tensor_imshow(img[0])
        plt.imshow(variance, cmap='jet', alpha=0.5)
        plt.title('acc {:.2f}, iou {:.2f}, snr {:.2f}, log {:.2f}'.format(100*original_prob, iou_score, snr_value, log_likelihood_value))
        plt.savefig(output_dir+"/variance_{:04d}.png".format(i+14050))
        plt.close()

        tensor_imshow(img[0])
        plt.imshow(explanations[i], cmap='jet', alpha=0.5)
        plt.title('{:.2f}% , original {} -> {}'.format(100*original_prob, get_class_name(find_class_row(data_loader.dataset.classes[data_classes])), get_class_name(original_class)))
        plt.savefig(output_dir+"/originalexplainability_{:04d}.png".format(i+14050))
        plt.close()

        # if class is same
        if(find_class_row(data_loader.dataset.classes[data_classes]) == original_class):
            list_result_true.append([original_prob*100, iou_score, snr_value, log_likelihood_value])

        else:
            list_result_false.append([original_prob*100, iou_score, snr_value, log_likelihood_value])
            
        np.save(mask_dir+'/uncertain_{:05}.npy'.format(i), variance)
        print("{} th for loop processed..".format(i))
    
    # 배치 결과를 CSV 파일로 저장합니다.
    batch_results_true = pd.concat(list_result_true)
    batch_results_false = pd.concat(list_result_false)
    batch_results_true.to_csv(f'{csv_output_dir}/true_{batch_number}.csv', index=False)
    batch_results_false.to_csv(f'{csv_output_dir}/false_{batch_number}.csv', index=False)
    print(f"Batch {batch_number} saved.")

cudnn.benchmark = True

args = Dummy()

weights = np.array([0.2989, 0.5870, 0.1140])

# Number of workers to load data
args.workers = 0

args.datadir = image_dir[0]
args.range = image_dir[1]

batch_size = 100
total_images = len(args.range)
# Size of imput images.
args.input_size = (224, 224)
# Size of batches for GPU. 
# Use maximum number that the GPU allows.
args.gpu_batch = 32

## Prepare data
dataset = datasets.ImageFolder(args.datadir, preprocess)

# This example only works with batch size 1. For larger batches see RISEBatch in explanations.py.
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False,
    num_workers=args.workers, pin_memory=True, sampler=RangeSampler(args.range))

print('Found {: >5} images belonging to {} classes.'.format(len(dataset), len(dataset.classes)))
print('      {: >5} images will be explained.'.format(len(data_loader) * data_loader.batch_size))


## Load models
model = models.resnet50(True)
model = nn.Sequential(model, nn.Softmax(dim=1))
model = model.eval()
model = model.cuda()

for p in model.parameters():
    p.requires_grad = False
    
# To use multiple GPUs
model = nn.DataParallel(model)

## create explainer instance
explainer = RISE(model, args.input_size, args.gpu_batch)
# Generate masks for RISE or use the saved ones.
maskspath = 'masks.npy'

if 1:#not os.path.isfile(maskspath):
    explainer.generate_masks(N=6000, s=8, p1=0.1, savepath=maskspath)
else:
    explainer.load_masks(maskspath)
    print('Masks are loaded.')

if (EXPLAIN_FOR_THE_FIRST_TIME == 1):
    explanations = explain_all(data_loader, explainer)
    np.save(explanation_dir+'/exp_{:05}-{:05}.npy'.format(args.range[0], args.range[-1]), explanations)
else:
    explanations_filename = explanation_dir+'/exp_{:05}-{:05}.npy'.format(args.range[0], args.range[-1])
    explanations = np.load(explanations_filename, allow_pickle=True)

# 전체 이미지를 배치 단위로 처리합니다.
for batch_num in range(0, total_images // batch_size):
    process_batch(data_loader, model, explainer, explanations, batch_num, batch_num * batch_size)
