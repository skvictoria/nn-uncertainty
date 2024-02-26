import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt

# Paths to your .npy files, each containing 500 maps
paths = ['/home/seulgi/work/nn-uncertainty/mask_generator/explainers/gradcam-exp_00000-00499.npy',
         '/home/seulgi/work/nn-uncertainty/mask_generator/explainers/gradcampp200-exp_00000-00499.npy',
         '/home/seulgi/work/nn-uncertainty/mask_generator/explainers/scorecam-exp_00000-00499.npy',
         ]

# Number of maps in each file
num_maps = 500

# Loop through each map index
for map_index in range(num_maps):
    # Load the specific map from each file and store in a list
    maps = np.array([np.load(path, allow_pickle=True)[map_index, :, :] for path in paths])
    
    # Now, `maps` is an array of shape (8, 224, 224) for the current map_index
    # Proceed with variance and entropy calculations for this set of maps
    
    # Variance-based disagreement map for the current map_index
    variance_map = np.var(maps, axis=0)
    # Save the variance map with a unique name for each map_index
    np.save(f'/home/seulgi/work/nn-uncertainty/evaluation/disagreement/npy/variance_map_{map_index}.npy', variance_map)
    plt.imshow(variance_map, cmap='viridis')
    plt.colorbar()
    plt.title(f'Variance Map {map_index}')
    plt.savefig(f'/home/seulgi/work/nn-uncertainty/evaluation/disagreement/png/variance_map_{map_index}.png')
    plt.close()  # Close the figure to free memorys
    # Entropy-based disagreement map (with normalization if needed)
    # Initialize an empty array for the entropy map
    entropy_map = np.zeros((224, 224))
    
    for i in range(224):
        for j in range(224):
            # Extract the pixel values across the 8 maps for the current pixel position
            pixel_values = maps[:, i, j]
            sum_pixel_values = np.sum(pixel_values)
            if sum_pixel_values > 0:
                normalized_values = pixel_values / sum_pixel_values
            else:
                # Handle the zero sum case; here we use a uniform distribution as an example
                normalized_values = np.ones_like(pixel_values) / len(pixel_values)
            entropy_map[i, j] = entropy(normalized_values, base=2)
    
    # Save the entropy map with a unique name for each map_index
    np.save(f'/home/seulgi/work/nn-uncertainty/evaluation/disagreement/npy/entropy_map_{map_index}.npy', entropy_map)
    # Save the entropy map as an image
    plt.imshow(entropy_map, cmap='viridis')
    plt.colorbar()
    plt.title(f'Entropy Map {map_index}')
    plt.savefig(f'/home/seulgi/work/nn-uncertainty/evaluation/disagreement/png/entropy_map_{map_index}.png')
    plt.close()