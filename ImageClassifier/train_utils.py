import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from workspace_utils import active_session

# Define transforms for training, validation, and testing datasets
def setup_transforms():
    """  Define transforms for training, validation and testing datasets
    
    return: transforms dictionary """
    global transforms
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225]) ])
    
    test_transforms = transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                      [0.229, 0.224, 0.225]) ])
    
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                      [0.229, 0.224, 0.225])])
    
    transforms = {'train': train_transforms, 'valid': valid_transforms, 'test': test_transforms}
    
    return transforms

# Create DataLoaders for training, validation and testing datasets
def setup_loaders():
    """ Create dataloaders for training, validation and testing datasets 
    
    return: trainloader, validationloader and testloader """
    global trainloader
    global testloader
    global validloader
    global image_datasets
    
    transforms = setup_transforms()
    # Define train_dir, valid_dir, and test_dir
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=transforms['train'])
    valid_data = datasets.ImageFolder(valid_dir, transform=transforms['valid'])
    test_data = datasets.ImageFolder(test_dir, transform=transforms['test'])
    
    image_datasets = {'train': train_data, 'valid': valid_data, 'test': test_data}
    
    # TODO: Create training, validation and testing loaders
    trainloader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32)
    testloader = torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)
    
    return trainloader, validloader, testloader

# Load a pre-trained network architecture and Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
def setup_model(arch, lr_temp=0.001, hidden_units=5000):
    """ Load a pre-trained network architure and Define a new, untrained feed-forward network as a classifier 
    
    return: model, criterion, optimizer"""
    
    global device
    global ip_unit
    global model
    global classifier
    global optimizer
    global criterion
    
    # Load model architecture
    if(arch=='vgg16'):
        model = models.vgg16(pretrained=True)
        ip_unit = 25088
        
    elif(arch =='vgg19'):
        #model = models.vgg19(pretrained=True)
        model.features[30] = nn.AdaptiveAvgPool2d(output_size(7,7))
        ip_unit = 25088
    elif(arch == 'densenet121'):
        model=models.densenet121(pretrained=True)
        ip_unit = 1024
    else:
        print('Invalid model architecture')
    # Freeze parameters of model so we don't backprog through them
    for param in model.parameters():
        param.requires_grad = False     
    
    # Create classifier as fcl
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(ip_unit, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),  
                          ('fc2', nn.Linear(hidden_units, 1000)),
                          ('relu2', nn.ReLU()),
                          ('dropout2', nn.Dropout(p=0.5)),
                          ('fc3', nn.Linear(1000, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    
    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    #Add momentum to optimizer for SGD
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr_temp)
    
    # Move model to device available
    device = torch.device("cuda:0" if "gpu" else "cpu")
    model.to(device)
    
    return model, criterion, optimizer

# Implement function for pass through validation or testing loader
def validation(model, _loader, criterion, device):
    """ Perform pass through validation or testing loader  
    
    return: test_loss, accuracy"""
    test_loss = 0
    accuracy = 0
    
    model.to(device)
    
    for images, labels in _loader:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        
        #class probability
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        
    return test_loss, accuracy

def train_model(epochs=10):
    ''' Train the model '''
    model.to(device)
    steps = 0
    running_loss = 0
    print_every = 40
    
    for e in range(epochs):
        model.train()
        
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            steps += 1
            
            with active_session():
                outputs = model.forward(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            if steps % print_every == 0:
                # Put network in eval mode for inference
                model.eval()
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validloader, criterion, device)
                
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Validation Loss: {:.3f}.. ".format(test_loss/len(validloader)),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
            
                running_loss = 0
                
                # Make sure training is back on
                model.train()
           
# Do validation on the test dataset
def test_model():
    ''' Perform validation on the test set '''
    with torch.no_grad():
        test_loss, accuracy = validation(model, testloader, criterion, device)
        
    print("Test Loss.. {:.3f}".format(test_loss/len(testloader)),
          "Test Accuracy.. {:.3f}".format(accuracy/len(testloader)))   
            
# Save model to checkpoint
def save_checkpoint(arch):
    '''Save model parameters to checkpoint to later rebuild model for inference'''
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {'arch': arch,
                  'model_state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'classifier': classifier}
    
    #checkpoint={#'input_size': ip_unit,
                #'epochs': epochs,
                #'optimizer_state_dict': optimizer.state_dict(),
                #'lr': lr,
                #'classifier': classifier }
    
    torch.save(checkpoint, 'checkpoint.pth')

