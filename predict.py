import argparse
import torch
from torchvision import transforms, models
from PIL import Image
import json
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', type=str, help='checkpoint file')
    parser.add_argument('--top_k', type=int, default=1, help='Number of top classes to show')
    parser.add_argument('--image_path', type=str, help='the image file')
    parser.add_argument('--category_names', type=str, help='JSON file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    return parser.parse_args()

def load_checkpoint(filepath):
    # Load the checkpoint and rebuild the model
    checkpoint = torch.load(filepath)
    model = models.__dict__[checkpoint['arch']](pretrained=True)  # Load the pre-trained model architecture
    model.classifier = checkpoint['classifier']  # Restore the classifier
    model.load_state_dict(checkpoint['state_dict'])  # Load the trained model weights
    model.class_to_idx = checkpoint['class_to_idx']  # Load the mapping of class indices to labels
    return model

def process_image(image):
    # Preprocess the image
    image_transforms = transforms.Compose([
        transforms.Resize(256),  # Resize the image to 256 pixels
        transforms.CenterCrop(224),  # Crop the center of the image to 224 pixels
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
    ])
    image = image_transforms(image)
    return image

def predict(image_path, model, top_k=1, gpu=False):
    # Predict the class probabilities of an image
    if gpu and torch.cuda.is_available():
        device = torch.device('cuda')  # Use GPU if available
    else:
        device = torch.device('cpu')  # Use CPU if GPU is not available
    model.to(device)
    model.eval()
    image = Image.open(image_path)
    image = process_image(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    
    with torch.no_grad():
        output = model(image)
        probabilities = torch.exp(output)
        top_probabilities, top_indices = torch.topk(probabilities, top_k)
    
    top_probabilities = top_probabilities.squeeze().tolist()
    top_indices = top_indices.squeeze().tolist()
    
    class_mapping = model.class_to_idx
    idx_to_class = {value: key for key, value in class_mapping.items()}
    
    top_classes = [idx_to_class[index] for index in top_indices]
    
    return top_probabilities, top_classes

def load_category_names(json_file):
    # Load the JSON file containing class labels
    with open(json_file, 'r') as f:
        class_mapping = json.load(f)
    return class_mapping

args = parse_args()

model = load_checkpoint(args.checkpoint_path)

if args.category_names:
    class_mapping = load_category_names(args.category_names)
else:
    class_mapping = None

top_probabilities, top_classes = predict(args.image_path, model, args.top_k, args.gpu)

for prob, class_index in zip(top_probabilities, top_classes):
    if class_mapping:
        class_label = class_mapping[class_index]
    else:
        class_label = class_index
    print(f'Class: {class_label}, Probability: {prob}')