import tensorflow as tf
import numpy as np
from PIL import Image
import io

def load_model():
    model = tf.saved_model.load("ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model")
    return model

def load_class_names(file_path):
    with open(file_path, 'r') as f:
        class_names = f.read().strip().split('\n')
    return class_names

def detect_object(image, model, class_names):
    image = Image.open(image)
    image_np = np.array(image)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = model(input_tensor)

    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(np.int32)
    detection_scores = detections['detection_scores'][0].numpy()

    threshold = 0.5
    results = []
    for i in range(len(detection_scores)):
        if detection_scores[i] >= threshold:
            box = detection_boxes[i].tolist()
            class_name = int(detection_classes[i])
            score = float(detection_scores[i])
            results.append({
                'box': box,
                'class': class_name,
                'class_name': class_names[class_name],
                'score': score
            })
    return results
