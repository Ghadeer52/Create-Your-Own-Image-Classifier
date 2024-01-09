import argparse
import torch
from torch import nn, optim
from torchvision import datasets, models, transforms

# Parse command line arguments using argparse
parser = argparse.ArgumentParser()
# Specify the data directory where the dataset is located
parser.add_argument('data_directory', type=str, default='ImageClassifier/flowers', help='dataset')
# Specify the directory and filename to save the checkpoint
parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='checkpoint')
# Specify the architecture for the model (choices: vgg16 or resnet18)
parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'resnet18'], help='Architecture for the model')
# Specify the learning rate for the optimizer
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
# Specify the number of hidden units in the classifier
parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
# Specify the number of epochs for training
parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
# Use GPU for training if specified
parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
args = parser.parse_args()

# Load and preprocess the dataset
data_dir = args.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

# Define the transformations for training and validation datasets
train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the training and validation datasets
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

# Create data loaders for training and validation datasets
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

# Load the pre-trained model architecture
if args.arch == 'vgg16':
    model = models.vgg16(pretrained=True)
elif args.arch == 'resnet18':
    model = models.resnet18(pretrained=True)
else:
    print('Please choose "vgg16" or "resnet18".')
    exit()

# Freeze the pre-trained model parameters
for param in model.parameters():
    param.requires_grad = False

#Modify the classifier of the pre-trained model
classifier = nn.Sequential(
    nn.Linear(model.classifier[0].in_features, args.hidden_units),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(args.hidden_units, len(train_data.classes)),
    nn.LogSoftmax(dim=1)
)

model.classifier = classifier

#Train the model
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

epochs = args.epochs
steps = 0
running_loss = 0
print_every = 10
#the training loop!
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move inputs and labels to the device (GPU)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        # Forward pass
        logps = model(inputs)
        # Compute the loss
        loss = criterion(logps, labels)
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            # Validation step
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    # Forward pass in evaluation mode
                    logps = model(inputs)
                    batch_loss = criterion(logps, labels)
                    #Compute the validation loss
                    valid_loss += batch_loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Steps {steps} - "
                  f"Train loss: {running_loss/print_every:.4f} - "
                  f"Validation loss: {valid_loss/len(validloader):.4f} - "
                  f"Validation accuracy: {accuracy/len(validloader):.4f}")

            running_loss = 0

# Step 7: Save the trained model checkpoint
# Saving the trained model checkpoint including the model architecture, classifier, optimizer state, etc.
checkpoint = {
    'arch': args.arch,
    'classifier': model.classifier,
    'class_to_idx': train_data.class_to_idx,
    'state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epochs': epochs
}

torch.save(checkpoint, args.save_dir)