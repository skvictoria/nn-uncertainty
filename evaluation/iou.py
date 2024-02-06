import numpy as np

def calculate_iou(image1, image2):
    intersection = np.logical_and(image1, image2)
    union = np.logical_or(image1, image2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

if __name__ == "__main__":
    # image load
    image1 = np.random.randint(0, 2, (224, 224), dtype=np.uint8)
    image2 = np.random.randint(0, 2, (224, 224), dtype=np.uint8)

    iou_score = calculate_iou(image1, image2)
    print(f"IOU Score: {iou_score}")
