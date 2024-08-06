from flask import Flask, request, jsonify
from object_detection import load_model, detect_object, load_class_names

app = Flask(__name__)
model = load_model()
class_names = load_class_names('coco_classes.txt')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    image = request.files['image']
    detections = detect_object(image, model, class_names)

    return jsonify(detections)

if __name__ == '__main__':
    app.run(debug=True)

#test