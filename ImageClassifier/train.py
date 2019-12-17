import argparse
from train_utils import *

# Initialize command line inputs to this script
def init_args():
    '''Initialize command line arguments (or inputs) for the script
    return: parser object'''
    parser = argparse.ArgumentParser(description='Specify Training File Arguments')
    
    #CLI for image datasets
    parser.add_argument("data_dir", nargs='*', action='store', default='flowers', help='specify directory for image datasets')
    
    #CLI for model architecture
    parser.add_argument("--arch", action='store', dest='arch', default='vgg16', help='specify model architecture')
    
    #CLI for directory to save checkpoints
    parser.add_argument("--save_dir", action='store', dest='save_dir', default='checkpoint.pth', help='specify directory to save                                     checkpoints')
    
    #CLI for learning_rate parameter
    parser.add_argument("--learning_rate", type=float, action='store', dest='learning_rate', default=0.001, help='specify learning_rate for                         model training')
    
    #CLI for hidden_units parameter
    parser.add_argument("--hidden_units", type=int, action='store', dest='hidden_units', default=5000, help='specify model hidden units')
    
    #CLI for epochs parameter
    parser.add_argument("--epochs", type=int, action='store', dest='epochs', default=10, help='specify number of epochs')
    
    #CLI for gpu parameter
    parser.add_argument("--gpu", action='store', dest='gpu', default='gpu', help='specify whether to use gpu or not')
    
    return parser

def parse_args():
    ''' Parse the command line arguments to the script '''
    global data_dir
    global arch
    global gpu
    global lr
    global checkpoint_path
    global epochs
    global hidden_units
    
    parser = init_args()
    
    args = parser.parse_args()
    
    #parse data_dir
    data_dir = args.data_dir
    #parse arch
    arch =  args.arch
    #parse save_dir (checkpoint_path)
    checkpoint_path = args.save_dir
    #parse learning_rate(lr)
    lr = args.learning_rate
    #parse hidden_units
    hidden_units = args.hidden_units
    #parse epochs
    epochs = args.epochs
    #parse gpu
    gpu = args.gpu
    
    #Dataset dir
    data_dir = data_dir[0]

def main():
    # Initialize and parse CL Arguments
    parse_args()
    # Set train, valid and test loaders
    setup_loaders()
    # Set up model
    setup_model(arch, lr, hidden_units)
    # Train model
    print('Training...')
    train_model(epochs)
    print('Training stopped!')
    # Test model
    print('Testing...')
    test_model()
    print('Testing stopped!')
    # Save model to checkpoint
    print('Saving model to checkpoint..')
    save_checkpoint(arch)
    print('Model saved successfully!')
    print('train.py Done!')
     

if __name__ == '__main__':
    main()      