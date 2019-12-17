import torch
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models

def load_checkpoint(checkpoint):
    ''' Load checkpoint to be used for inference 
    returns checkpoint'''
    global model
    global device
    
    device = torch.device("cuda:0" if "gpu" else "cpu")
    
    checkpoint = torch.load(checkpoint)
    if(checkpoint['arch'] == 'vgg16'):
        model = models.vgg16(pretrained=True)
        # Make sure the final feature maps have a dimension of '(512, 7, 7)' irrespective of the input size.
        model.features[30] = nn.AdaptiveAvgPool2d(output_size=(7,7))
        
    elif(checkpoint['arch'] == 'vgg19'):
        model = models.vgg19(pretrained=True)
        # Make sure the final feature maps have a dimension of '(512, 7, 7)' irrespective of the input size.
        model.features[30] = nn.AdaptiveAvgPool2d(output_size=(7,7))
        
    elif(checkpoint['arch'] == 'vgg19'):
        model=models.densenet121(pretrained=True)
        
   
    for param in model.parameters():
        param.requires_grad = False
        
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
        
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # TODO: Process a PIL image for use in a PyTorch model
    with Image.open(image) as pil_image:
        width, height = pil_image.size
        
        if width >= height:
            ratio = width / height
            pil_image.thumbnail((ratio*256, 256))
        elif height >= width:
            ratio = height /width
            pil_image.thumbnail((256, ratio*256))
            
        new_width, new_height = pil_image.size
        left = (new_width - 224)/2
        top = (new_height - 224)/2
        right = (new_width + 224)/2
        bottom = (new_height + 224)/2
        
        pil_image.crop((left, top, right, bottom))
        np_image = np.array(pil_image)/255
        #Normalize image
        np_image = (np_image-mean)/std 
        np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def predict(image_path, checkpoint, top_k=5):
    ''' Predict the class (or classes)  of an image using a trained deep learning model.
    returns highest k probabilities and the indices of those probabilities corresponding to the classes.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    model = load_checkpoint(checkpoint)

    input_image = process_image(image_path)
    image = torch.from_numpy(input_image).type(torch.FloatTensor)
    
    model = model.to(device)
    image = image.to(device)
    
    # Add batch dimension
    image = image.unsqueeze_(dim=0)
    
    model.eval()
    with torch.no_grad():
        output = model.forward(image)
        
    #class probability
    ps = torch.exp(output)
    top_p, top_class = ps.topk(top_k)
    
    #convert top_p and top_class tensors to numpy arrays
    top_p, top_class = top_p.cpu().squeeze().numpy(), top_class.cpu().squeeze().numpy()
    
    #Invert class_to_idx dict to idx_to_class dict
    class_to_idx = model.class_to_idx
    idx_to_class = {str(v):int(k) for k, v in class_to_idx.items()}
    
    # Get class labels using idx_to_class dictionary
    top_class = np.array([idx_to_class.get(str(idx)) for idx in top_class])
    
    # Map categories to real names
    return top_p, top_class


# Map categories to real names 
def cat_to_names(cat_to_name_file, image_path, checkpoint, top_k):
    '''Map classes of image from prediction to their actual names
    
    returns: Real names for classs of images'''
    with open(cat_to_name_file) as f:
        cat_to_names = json.load(f)
    
    top_p, top_class = predict(image_path, checkpoint, top_k)
    names = [cat_to_names.get(str(idx)) for idx in top_class]
    
    return names