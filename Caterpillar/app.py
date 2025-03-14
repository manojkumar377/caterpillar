from flask import Flask, request, jsonify, render_template_string
import torch
from torchvision import models, transforms
from PIL import Image
import io
from twilio.rest import Client

app = Flask(__name__)

# Define the model architecture to match the trained model
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # Use the same output size as during training
model.load_state_dict(torch.load('/home/manoj/Desktop/Caterpillar/caterpillar_detector.pth'))  # Load the state dict
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_with_model(image):
    try:
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(image)
        print(f'Model output: {output}')  # Debugging print
        probabilities = torch.nn.functional.softmax(output, dim=1)
        print(f'Probabilities: {probabilities}')  # Debugging print
        return probabilities[0][1].item()  # Probability of the positive class
    except Exception as e:
        print(f'Error in prediction: {e}')  # Debugging print
        return None

@app.route('/')
def upload_form():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Upload</title>
        <style>
            body {
                background: linear-gradient(135deg, #74ebd5 0%, #acb6e5 100%);
                font-family: Arial, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                color: #333;
            }

            #container {
                background: rgba(255, 255, 255, 0.8);
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                padding: 20px;
                text-align: center;
                animation: fadeIn 1s ease-out;
            }

            h1 {
                font-size: 24px;
                margin-bottom: 20px;
            }

            #upload-form {
                display: flex;
                flex-direction: column;
                align-items: center;
            }

            input[type="file"] {
                margin-bottom: 20px;
            }

            button {
                background: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                cursor: pointer;
                transition: background 0.3s;
            }

            button:hover {
                background: #45a049;
            }

            #result {
                margin-top: 20px;
                font-size: 18px;
                font-weight: bold;
            }

            @keyframes fadeIn {
                from {
                    opacity: 0;
                }
                to {
                    opacity: 1;
                }
            }
        </style>
    </head>
    <body>
        <div id="container">
            <h1>Upload an Image</h1>
            <form id="upload-form">
                <input type="file" id="image" name="image" accept="image/*" required>
                <button type="submit">Upload</button>
            </form>
            <p id="result"></p>
        </div>

        <script>
            document.getElementById('upload-form').onsubmit = async function(event) {
                event.preventDefault();
                let formData = new FormData();
                formData.append('image', document.getElementById('image').files[0]);

                let response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    let result = await response.json();
                    document.getElementById('result').textContent = result.has_caterpillar ? "Caterpillar detected!" : "No caterpillar detected.";
                } else {
                    document.getElementById('result').textContent = "Error in prediction.";
                }
            };
        </script>
    </body>
    </html>
    """
    return render_template_string(html_content)

# Twilio configuration
account_sid = 'ACd8a95b4c1ddcbb28e6262511b464d621'  # Replace with your Twilio account SID
auth_token = 'd3e51e59f36ec4fc6d757577a0f523e2'  # Replace with your Twilio auth token
twilio_phone_number = '+14707779672'  # Replace with your Twilio phone number
destination_phone_number = '+918438923377'  # Replace with your phone number

client = Client(account_sid, auth_token)

def send_text_message(message):
    client.messages.create(
        body=message,
        from_=twilio_phone_number,
        to=destination_phone_number
    )

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    image = request.files['image'].read()
    if not image:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        image = Image.open(io.BytesIO(image)).convert('RGB')  # Convert image to RGB
        prediction = predict_with_model(image)
        if prediction is None:
            return jsonify({'error': 'Error in prediction process'}), 500
        
        has_caterpillar = prediction > 0.5  # Adjust threshold based on your model
        if has_caterpillar:
            send_text_message("Caterpillar detected in your tree!")  # Send text message if caterpillar is detected
        return jsonify({'has_caterpillar': has_caterpillar})
    except Exception as e:
        print(f'Error processing image: {e}')  # Debugging print
        return jsonify({'error': 'Error processing image'}), 500

if __name__ == '__main__':
    app.run(debug=True)
