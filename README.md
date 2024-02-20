# nn-uncertainty

## Folder structure
1. Mask generator
- Input: model(default: ResNet50), explainer(default: RISE), image(default: ImageNet)
- Output: masks(.npy format), visualizer

2. Evaluation
- Input: masks, explainer, model
- Output: iou, snr, 