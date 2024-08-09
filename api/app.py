from flask import Flask, request, jsonify
from object_detection import load_model, detect_object, load_class_names

app = Flask(__name__)
model = load_model()
class_names = load_class_names('coco_classes.txt')

@app.route('/detect', methods=['POST'])
@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    image_url = data['image_url']

    # Call the detect_object function with the image URL
    detections = detect_object(image_url, model, class_names)

    return jsonify(detections)

if __name__ == '__main__':
    app.run()

#test