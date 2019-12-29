# Image Classifier

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smartphone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.

In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice, you'd train this classifier, then export it for use in your application. We'll be using this dataset of 102 flower categories.

When you've completed this project, you'll have an application that can be trained on any set of labelled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset.

# Requirements
 ## Programming language
  - [Python (v3.7)](https://www.python.org/)
  
 ## Packages
  - [Numpy](https://numpy.org)
  - [Pandas](https://pandas.pydata.org/)
  - [MatplotLib](https://matplotlib.org/)
  - [PIL(Python Imaging Library)](https://pillow.readthedocs.io/en/stable/reference/Image.html)
  - [JSON(Javascript Object Notation)](https://www.json.org/json-en.html)
  - [Pytorch(v0.4.0)](https://pytorch.org/docs/0.4.0/)
  
  The best and recommended option to get all packages and python programming language is to download and install Anaconda [here](https://www.anaconda.com/distribution/).   Anaconda comes preinstalled with [Conda](https://conda.io/en/latest/); an open-source package manager that helps you find and install packages.
  
  To manage(install, update, remove, etc..) all packages directly except Pytorch , please reference these Conda commands [here](https://conda.io/projects/conda/en/latest/commands.html).
  
  All except Pytorch does not come preinstall with Anaconda or can't be managed directly with Conda. 
  
  **After successfully downloading and installing Anaconda:**
  - Go to [PyTorch's site](https://pytorch.org/) and find the *QUICK START LOCALLY* section.
  - Specify the appropriate configuration options for your particular environment.
  - Run the presented command in the terminal to install Pytorch.
  
# Viewing the Jupyter Notebook
As specified under *Accessing Individual Folders of the Repository* [here](https://github.com/Besong/Data-Science-Portfolio/blob/master/README.md/), it is very much better to work and view the jupyter notebook using [jupyter nbviewer](https://nbviewer.jupyter.org/). This simply requires you to cut and paste the path of the notebook on the website.

Alternatively, read through *Accessing Individual Folders of the Repository* [here](https://github.com/Besong/Data-Science-Portfolio/blob/master/README.md/) to clone the project locally. 

Using the command `jupyter notebook` on the command line, open jupyter notebook and locate project notebook and run it.

# Command Line Application
- Train a new network on a dataset with `train.py`
  - Basic Usage : `python train.py data_dir`
  - Prints out current epoch, training loss, validation loss, and validation accuracy as the network trains
  - Options:
    - Set directory to save checkpoints: `python train.py data_dir --save_dir save_dir`
    - Chose architecture(densenet121, vgg16 and vgg19 available): `python train.py data_dir --arch "vgg16"`
    - Set hyperparameters: `python train.py dat_dir --learning_rate 0.001 --hidden_units 512 --epochs 20`
    - Use GPU for training: `python train.py data_dir --gpu`
    
- Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image `/path/to/image` and return the flower name and class probability.
  - Basic Usage : `python predict.py /path/to/image checkpoint`
  - Options:
    - Return top **K** most likely classes: `python predict.py input checkpoint --top_k 3`
    - Use a mapping of categories to real names: `python predict.py input checkpoint --category_names cat_to_name.json`
    - Use GPU for inference: `python predict.py input checkpoint --gpu`
    
# Json File
For the network to print out the name of he flower, a *.json* file is required.  By using a *.json* file, the data can be sorted into folders with numbers, with those numbers corresponding to specific names specified in the *.json file*

# Data
The data used for this project are a flower database, not provided in this repository as it is larger than what gihub allows. But, you can create your database and train the model on them to use with with your own projects. 

***Structure the data in folders and proportions respectively as follows:***
- train(70%)
- test(20%)
- validate(10%).

Inside each folder, there should be folders bearing a specific number which corresponds to a specific category; clarified in the json file. E.g, if we have an image *r.jpg* and it is a rose, it could be in a path like this */train/4/r.jpg* and the json file would be like this *{..., 4:"rose", ...}*.

Make sure to include alot of photos for each category(more than 10) with different angles, lightening conditions inorder for the network to generalizee better.

# Working with train.py
As you can see in the *Command Line Application* Heading of this *Readme*, training the model is performed when the *train.py* python file is executed, with its optional multiple hyperparameters. 

***Here are some hints on choosing the right hyperparameters to take note when training the model:***

- **Number of epochs:** Increasing the number of epochs, causes the accuracy of the network on the training set to get better and better. However, picking large numbers of epochs won't generalize  the network well, i.e, it won't perform well on the test images rather it will on the training images.
- **Learning rate:** Choose the appropriate learning rate for the network to converge steadily to a small error without overshoting, thus reaching greater accuracy.
- **Architecture for model used:** Vgg16 or Vgg19 takes a shorter time to train compared to Densenet121. However, Densenet  works best with images.
- **GPU:** Becuase of the network use of complex deep convolutional neural network, the training process is impossible to be done by your desktop or laptop. In order to train your network on your local machine efficiently, you have three options:
    1. **Cuda.** If you have an NVIDIA GPU then you can install CUDA from [here](https://developer.nvidia.com/cuda-downloads). With Cuda you will be able to train your model, however the process will still be time consuming.
    
    2. **Cloud Services.** [AWS](https://aws.amazon.com/) or [Google Cloud](https://cloud.google.com/) are a few of many paid cloud services that lets you train you models.
    
    3. **Google Colab.** [Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb#recent=true) gives you free access to a tesla K80 GPU for 12 hours at a time. Once 12 hours have ellapsed you can just reload and continue!. The only limitation is that you have to upload the data to Google Drive and if the dataset is massive, you may run out of space.
    
# Working with predict.py
Once the model is trained, then normal CPU can be used for the *predict.py python file* and you will have an answer within seconds.

The *checkpoint.pth file* contains the information of the network trained to recognise 102 different species of flowers. It has been trained with specific hyperparameters, thus if you don't set them right, the network will fail. In order to have a prediction for an image located in the path */path/to/image* using my pretrained model, you can simply type `python predict.py /path/to/image checkpoint.pth`

# Authors
- **Besong** - Initial work
- **Udacity** - Final Project of the AI with Python Nanodegree

Full Curriculum for Nanodegree is available [here](https://www.udacity.com/course/ai-programming-python-nanodegree--nd089)
