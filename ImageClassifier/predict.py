import argparse
from predict_utils import *

# Initialize command line inputs to this script
def init_args():
    '''Initialize command line arguments (or inputs) for the script
    return: parser object'''
    parser = argparse.ArgumentParser(description='Specify Predict File Arguments')
    
    #CLI for /path/to/image
    parser.add_argument("--input",  default = 'flowers/test/1/image_06752.jpg', required=True, help='specify path/to/image for class                                   inference')
    
    #CLI for directory to load saved checkpoints
    parser.add_argument("--checkpoint", action='store', default='checkpoint.pth', help='specify directory to load saved                                               checkpoints')
    #CLI for top_k parameter
    parser.add_argument("--top_k", type=int, action='store', dest='top_k', default=5, help='specify number of k probabilities and classes to                         be returned')
    #CLI for label mapping file parameter
    parser.add_argument("--category_names", help='specify file to map labels to name')
    
    #CLI for gpu parameter
    parser.add_argument("--gpu", action='store', dest='gpu', default='gpu', help='specify whether to use gpu or not')
    
    return parser


def parse_args():
    ''' Parse the command line arguments to the script '''
    
    global image_path
    global checkpoint_path
    global top_k
    global cat_to_name_file
    global gpu
    
    parser = init_args()
    
    args = parser.parse_args()
    
    #parse input as path_to_image
    image_path = args.input
    #parse checkpoint
    checkpoint_path = args.checkpoint
    #parse top_k
    top_k = args.top_k
    #parse category_names
    cat_to_name_file = args.category_names
    #parse gpu
    gpu = args.gpu
    
def main():
    
    # Initialize and parse CL Arguments
    parse_args()
    
    print('Predicting..')
    
    if cat_to_name_file is None:
        #Predict image probs and classes
        top_p, top_class = predict(image_path, checkpoint_path, top_k)
        print(top_p)
        print(top_class)
        
    else:
        #Map category labels to real names
        names = cat_to_names(cat_to_name_file, image_path, checkpoint_path, top_k)
        print(names)
        
    print('Prediction Completed!')
    
if __name__ == '__main__': 
    main()