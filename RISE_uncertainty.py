import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.utils import save_image

from utils import *
from RISE import RISE

from PIL import Image

EXPLAIN_FOR_THE_FIRST_TIME = 0

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


def random_masking_for_some_fraction(image, mask_size=50, fraction=0.7):
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

cudnn.benchmark = True

args = Dummy()

weights = np.array([0.2989, 0.5870, 0.1140])

# Number of workers to load data
args.workers = 8
# Directory with images split into class folders.
# Since we don't use ground truth labels for saliency all images can be 
# moved to one class folder.
args.datadir = '/data/datasets/ImageNet/val/'
# Sets the range of images to be explained for dataloader.
args.range = range(95, 105)
# Size of imput images.
args.input_size = (224, 224)
# Size of batches for GPU. 
# Use maximum number that the GPU allows.
args.gpu_batch = 250

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

if not os.path.isfile(maskspath):
    explainer.generate_masks(N=6000, s=8, p1=0.1, savepath=maskspath)
else:
    explainer.load_masks(maskspath)
    print('Masks are loaded.')

if (EXPLAIN_FOR_THE_FIRST_TIME == 1):
    explanations = explain_all(data_loader, explainer)
    np.save('exp_{:05}-{:05}.npy'.format(args.range[0], args.range[-1]), explanations)
else:
    explanations_filename = 'exp_{:05}-{:05}.npy'.format(args.range[0], args.range[-1])
    explanations = np.load('/home/sophia/nn-uncertainty/exp_00095-00104.npy', allow_pickle=True)

for i, (img, _) in enumerate(data_loader):
    original_prob, original_class = torch.max(model(img.cuda()), dim=1)
    original_prob, original_class = original_prob[0].item(), original_class[0].item()
    #save_image(img[0], 'original_img_{:04d}.png'.format(i))
    explanation_tensor = torch.from_numpy(explanations[i]).unsqueeze(0)
    #save_image(explanation_tensor, 'explanations_{:04d}.png'.format(i))

    ## just applying mask
    #masked_image = apply_mask(img[0].float(), explanations[i])

    ## invert masking
    #masked_image = invert_masking(img[0].float(), explanations[i])

    ##### 1. Set Threshold #####
    thres_prob = 0.75
    while 1:
        masked_image = explainability_masking(img[0].float(), explanations[i], p1=thres_prob)
        _, cl = torch.max(model(masked_image.unsqueeze(0)), dim=1)
        if(original_class != cl[0].item()):
            thres_prob += 0.1
            break
        thres_prob -= 0.05

    masked_image = explainability_masking(img[0].float(), explanations[i], p1=thres_prob)
    masked_prob, masked_class = torch.max(model(masked_image.unsqueeze(0)), dim=1)
    masked_prob, masked_class = masked_prob[0].item(), masked_class[0].item()
    #save_image(masked_image, 'masked_img_{:04d}.png'.format(i))

    image_list = []
    image_list_for_RISE = []
    for random_idx in range(10000):
        randomly_masked_image, random_mask_for_save = random_masking_for_some_fraction(masked_image)###masked_image
        #save_image(randomly_masked_image, 'randomly_masked_img_{:04d}.png'.format(i))

        chang_prob, chang_class = torch.max(model(randomly_masked_image.unsqueeze(0)), dim=1)
        chang_prob, chang_class = chang_prob[0].item(), chang_class[0].item()

        if original_class != chang_class:
            random_mask_for_save = random_mask_for_save.permute(1,2,0).numpy()
            random_mask_for_save = np.dot(random_mask_for_save[...,:3], weights)
            random_mask_for_RISE = np.dot(random_mask_for_save, chang_prob)
            image_list.append(random_mask_for_save)
            image_list_for_RISE.append(random_mask_for_RISE)

    pixel_values = np.stack(image_list, axis=0)
    pixel_values_for_RISE = np.stack(image_list_for_RISE, axis=0)

    ## simply adding
    ones_count = np.sum(pixel_values, axis=0)

    ## variance
    variance = np.var(pixel_values, axis=0)

    ## RISE
    rise = np.var(pixel_values_for_RISE, axis=0)

    ## entropy
    probability = ones_count / len(image_list)
    with np.errstate(divide='ignore', invalid='ignore'):
        entropy = - (probability * np.log2(probability + 1e-10))
        entropy[~np.isfinite(entropy)] = 0

    ## entropy-RISE
    ones_count_for_RISE = np.sum(pixel_values_for_RISE, axis=0)
    probability_for_RISE = ones_count_for_RISE / len(image_list_for_RISE)
    with np.errstate(divide='ignore', invalid='ignore'):
        entropy_for_RISE = - (probability_for_RISE * np.log2(probability_for_RISE + 1e-10))
        entropy_for_RISE[~np.isfinite(entropy_for_RISE)] = 0

    ############ image save for simply adding
    tensor_imshow(img[0])
    plt.imshow(ones_count, cmap='jet', alpha=0.5)
    plt.savefig("result_masks_point_2401201723/uncertainty_res_img_add_{:04d}.png".format(i))
    plt.close()

    ############# image save for variance
    tensor_imshow(img[0])
    plt.imshow(variance, cmap='jet', alpha=0.5)
    plt.savefig("result_masks_point_2401201723/uncertainty_res_img_variance_{:04d}.png".format(i))
    plt.close()

    ############ image save for RISE
    tensor_imshow(img[0])
    plt.imshow(rise, cmap='jet', alpha=0.5)
    plt.savefig("result_masks_point_2401201723/uncertainty_res_img_RISE_{:04d}.png".format(i))
    plt.close()

    ############ image save for entropy
    tensor_imshow(img[0])
    plt.imshow(entropy, cmap='jet', alpha=0.5)
    plt.savefig("result_masks_point_2401201723/uncertainty_res_img_entropy_{:04d}.png".format(i))
    plt.close()

    ############ image save for entropy+RISE
    tensor_imshow(img[0])
    plt.imshow(entropy_for_RISE, cmap='jet', alpha=0.5)
    plt.savefig("result_masks_point_2401201723/uncertainty_res_img_entropyRISE_{:04d}.png".format(i))
    plt.close()

    #im = Image.fromarray(save_mask)
    #im.save("result_masks/uncertainty_res_img_{:04d}.png".format(i))