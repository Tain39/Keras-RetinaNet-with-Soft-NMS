# Copyright 2025 Tain39
# Licensed under the Apache License, Version 2.0
# This file is an original inference script added to the Keras RetinaNet reproduction.
import os
import numpy as np
import cv2
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from softnms import retinanet_soft_nms

VOC2007_CLASSES = {
    0: 'aeroplane', 1: 'bicycle', 3: 'boat',
    4: 'bottle', 5: 'bus', 6: 'car', 7: 'cat',
    8: 'chair', 9: 'cow', 10: 'diningtable', 11: 'dog',
    12: 'horse', 13: 'motorbike', 14: 'person', 15: 'pottedplant',
    16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'
}

MODEL_PATH = "inference_model_step1000.h5"  # 推理模型
INPUT_FOLDER = "test_images"        # 待检测图片文件夹
OUTPUT_FOLDER = "results"           # 输出文件夹
CONFIDENCE = 0.5                    # 置信度门槛

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 加载模型
model = models.load_model(MODEL_PATH, backbone_name='resnet50')

# 批量检测
for img_name in os.listdir(INPUT_FOLDER):
    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(INPUT_FOLDER, img_name)

        # 读取 + 预处理
        image = read_image_bgr(img_path)
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        image = preprocess_image(image)
        image, scale = resize_image(image)

        # 预测
        boxes, scores, labels = model.predict_on_batch(
            np.expand_dims(image, axis=0))
        boxes /= scale

        # softnms处理
        boxes, scores, labels = retinanet_soft_nms(
            boxes=boxes[0],    # 因为是batch维度 [1, N, 4]，取第0个样本
            scores=scores[0],  # 同理，取第0个样本的置信度
            labels=labels[0],  # 同理，取第0个样本的类别
            iou_thr=0.5,
            sigma=0.5,
            score_thr=0.5
        )
	import numpy as np
	boxes = np.array(boxes)
	scores = np.array(scores)
	labels = np.array(labels)
	keep = labels != 2
	boxes = boxes[keep]
	scores = scores[keep]
	labels = labels[keep]
            # 画框
	for box, score, label in zip(boxes, scores, labels):
    		if score < CONFIDENCE:
        		continue
    		if label == 2:
        		continue
        	if label not in VOC2007_CLASSES:
        		continue
    		class_name = VOC2007_CLASSES[label]
    		draw_box(draw, box, color=(0, 255, 0))
    		draw_caption(draw, box, f"{class_name}: {score:.2f}")

        # 保存结果
        output_path = os.path.join(OUTPUT_FOLDER, img_name)
        cv2.imwrite(output_path, cv2.cvtColor(draw, cv2.COLOR_RGB2BGR))
