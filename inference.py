# Copyright 2025 Tain39
# Licensed under the Apache License, Version 2.0
# This file is an original inference script added to the Keras RetinaNet reproduction.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption

#  SOFT-NMS 
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    inter = w * h
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-8)

def soft_nms(boxes, scores, sigma=0.5, score_thr=0.3):
    scores_soft = scores.copy()
    N = boxes.shape[0]
    keep = []
    indices = np.arange(N, dtype=np.int32)

    while len(indices) > 0:
        max_idx = np.argmax(scores_soft[indices])
        curr = indices[max_idx]
        keep.append(curr)
        curr_box = boxes[curr]
        rest_indices = indices[indices != curr]

        for j in rest_indices:
            iou = compute_iou(curr_box, boxes[j])
            scores_soft[j] *= np.exp(-(iou ** 2) / sigma)

        indices = rest_indices[scores_soft[rest_indices] >= score_thr]

    return np.array(keep, dtype=np.int32)

VOC2007_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--backbone', default='resnet50')
    parser.add_argument('--save-path', dest='save_path', default='results')
    parser.add_argument('image_dir')
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    model = models.load_model(args.model, backbone_name=args.backbone)
    
    # 关闭模型自带的 NMS  

    for img_name in os.listdir(args.image_dir):
        if not img_name.lower().endswith(('jpg','png','jpeg')):
            continue

        img_path = os.path.join(args.image_dir, img_name)
        print("Processing", img_path)

        image = read_image_bgr(img_path)
        draw = image.copy()
        image = preprocess_image(image)
        image, scale = resize_image(image)

        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes /= scale
        boxes = boxes[0]
        scores = scores[0]
        labels = labels[0]

        keep = soft_nms(boxes, scores, sigma=0.5, score_thr=0.4)
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        for box, score, label in zip(boxes, scores, labels):
            if score < 0.4:
                continue
            cls = VOC2007_CLASSES[label]
            draw_box(draw, box, (0,255,0))
            draw_caption(draw, box, f"{cls}: {score:.2f}")

        save_path = os.path.join(args.save_path, img_name)
        cv2.imwrite(save_path, draw)
        print("Saved to", save_path)

if __name__ == '__main__':
    main()
