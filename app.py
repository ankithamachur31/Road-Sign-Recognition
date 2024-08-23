from flask import Flask, request, render_template
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import torch
import os

app = Flask(__name__)

model_name = "bazyl/gtsrb-model"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

classes = { 0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)', 
            9:'No passing', 10:'No passing for vehicles over 3.5 metric tons', 
            11:'Right-of-way at the next intersection', 12:'Priority road', 13:'Yield', 
            14:'Stop', 15:'No vehicles', 16:'Vehicles over 3.5 metric tons prohibited', 
            17:'No entry', 18:'General caution', 19:'Dangerous curve to the left', 
            20:'Dangerous curve to the right', 21:'Double curve', 22:'Bumpy road', 
            23:'Slippery road', 24:'Road narrows on the right', 25:'Road work', 
            26:'Traffic signals', 27:'Pedestrians', 28:'Children crossing', 
            29:'Bicycles crossing', 30:'Beware of ice/snow', 31:'Wild animals crossing', 
            32:'End of all speed and passing limits', 33:'Turn right ahead', 
            34:'Turn left ahead', 35:'Ahead only', 36:'Go straight or right', 
            37:'Go straight or left', 38:'Keep right', 39:'Keep left', 
            40:'Roundabout mandatory', 41:'End of no passing', 
            42:'End of no passing by vehicles over 3.5 metric tons', 43:'Not a Sign'}  

def load_image(image_path):
    image = Image.open(image_path)
    return image

def preprocess_image(image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs

def predict_image(image_path):
    image = load_image(image_path)
    inputs = preprocess_image(image)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
         
        confidence_threshold = 0.7
        softmax_logits = torch.softmax(logits, dim=-1)
        max_confidence = softmax_logits[0][predicted_class_idx].item()
        if max_confidence < confidence_threshold:
            sign_name = "Not a Sign"
        else:
            sign_name = classes[predicted_class_idx]
        return sign_name

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            
            if not os.path.exists('Test'):
                os.makedirs('Test')
            
            image_path = os.path.join('Test', image_file.filename)
            image_file.save(image_path)
            sign_name = predict_image(image_path)
            return render_template("index.html", prediction=sign_name, image_path=image_path)
    return render_template("index.html", prediction=None, image_path=None)

if __name__ == "__main__":
    app.run(debug=True)
