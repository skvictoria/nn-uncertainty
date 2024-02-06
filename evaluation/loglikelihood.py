import numpy as np

def calculate_log_likelihood(image1, image2):
    # normalize images
    image1_normalized = image1 / 255.0
    image2_normalized = image2 / 255.0

    epsilon = 1e-12
    log_likelihood = np.sum(image1_normalized * np.log(image2_normalized + epsilon) + 
                            (1 - image1_normalized) * np.log(1 - image2_normalized + epsilon))
    return log_likelihood

if __name__ == "__main__":
    image1 = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
    image2 = np.random.randint(0, 256, (224, 224), dtype=np.uint8)

    log_likelihood_value = calculate_log_likelihood(image1, image2)
    print(f"Log-Likelihood: {log_likelihood_value}")
