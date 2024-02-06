import numpy as np

def calculate_snr(um):
    mu = np.mean(um)
    sigma = np.std(um)
    snr = mu / sigma
    return snr

if __name__ == "__main__":
    um = np.random.rand(224, 224)

    snr_value = calculate_snr(um)
    print(f"SNR: {snr_value}")
