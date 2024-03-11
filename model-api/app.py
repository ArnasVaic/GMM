from flask import Flask
from flask_restx import Resource, Api
import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms

from model import Net

app = Flask(__name__)
api = Api(app)

model = Net()
model.load_state_dict(torch.load('model.txt'))
model.eval()

from werkzeug.datastructures import FileStorage

upload_parser = api.parser()
upload_parser.add_argument(
    'file', 
    location='files',
    type=FileStorage, required=True)

@api.route('/upload/')
@api.expect(upload_parser)
class Upload(Resource):
    def post(self):
        args = upload_parser.parse_args()
        uploaded_file = args['file']  # This is FileStorage instance
        
        image = Image.open(uploaded_file)
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        image = transform(image)
        # Add batch dimension since model expects a batch
        image = image.unsqueeze(0)

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        with torch.no_grad():
            outputs = model(image)

        predicted_class = torch.argmax(outputs, dim=1).item()

        return { 'object': classes[predicted_class] }, 201

if __name__ == '__main__':
    app.run(debug=True)