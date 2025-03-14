from flask import Flask, request, jsonify, render_template_string
import torch
from torchvision import models, transforms
from PIL import Image
import io
import os
from twilio.rest import Client

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # Two output classes: Caterpillar or No Caterpillar
try:
    model.load_state_dict(torch.load('/home/manoj/Desktop/Caterpillar/caterpillar_detector.pth', map_location=torch.device('cpu')))
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

def process_image(image_data):
    try:
        image = Image.open(io.BytesIO(image_data)).convert('RGB')  # Convert image to RGB
        tensor_image = transform(image).unsqueeze(0)  # Add batch dimension
        print(f"Processed image shape: {tensor_image.shape}")  # Debugging
        return tensor_image
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def predict_with_model(image_tensor):
    try:
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            print(f"Model output: {output}")  # Debugging
            print(f"Probabilities: {probabilities}")  # Debugging
            return probabilities[0][1].item()  # Probability of caterpillar presence
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

# Twilio setup
account_sid = os.getenv('ACd8a95b4c1ddcbb28e6262511b464d621')  # Set in environment variables
auth_token = os.getenv('d3e51e59f36ec4fc6d757577a0f523e2')    # Set in environment variables
twilio_phone_number = '+14707779672'# Replace with Twilio phone number
destination_phone_number = '+918438923377'# Replace with your phone number

client = Client(account_sid, auth_token)

def send_text_message(message):
    try:
        client.messages.create(
            body=message,
            from_=twilio_phone_number,
            to=destination_phone_number
        )
        print("SMS sent successfully!")
    except Exception as e:
        print(f"Error sending SMS: {e}")

@app.route('/')
def upload_form():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Caterpillar Detection</title>
    </head>
    <body>
        <h1>Upload Image for Caterpillar Detection</h1>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required>
            <button type="submit">Upload and Detect</button>
        </form>
    </body>
    </html>
    """
    return render_template_string(html_content)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image'].read()
    if not image:
        return jsonify({'error': 'Empty image file'}), 400

    try:
        tensor_image = process_image(image)
        if tensor_image is None:
            return jsonify({'error': 'Error processing image'}), 500

        prediction = predict_with_model(tensor_image)
        if prediction is None:
            return jsonify({'error': 'Error in prediction process'}), 500

        has_caterpillar = prediction > 0.5  # Adjust threshold as necessary
        if has_caterpillar:
            send_text_message("Caterpillar detected in your tree!")
        return jsonify({'has_caterpillar': has_caterpillar})
    except Exception as e:
        print(f"Error in prediction route: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # Accessible on the local network