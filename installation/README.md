# Installation Instructions

## Colab

The easy, on-the-fly way to use the Jupyter notebooks provided in this repository is to load them in [Colab](https://colab.research.google.com/notebooks/welcome.ipynb). The shortcoming of using Colab is that we can't control or freeze the versions of software libraries it uses by default and so some code may break in the future. If you'd like to be 100% sure that the Jupyter notebooks in this repo run as we intended, then follow the installation instructions for the operating system of your choosing below.


## Mac OS X

Detailed step-by-step instructions for running the code notebooks for on a Mac can be found [here](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/installation/step_by_step_MacOSX_install.md).


## Unix

#### Where You Already Have the Dependencies

The dependencies are provided in this repository's [Dockerfile](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/Dockerfile). If you have these packages configured as you like them, you can simply:
`https://github.com/the-deep-learners/deep-learning-illustrated.git`

#### Where You Are Missing Dependencies

1. Get Docker CE for, e.g., [Ubuntu](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/)
2. Follow all the steps in my [Step-by-Step Instructions for Mac](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/installation/step_by_step_MacOSX_install.md) that involve executing code at the command line. That is, execute all steps but one, four and five. 

## Windows

Community members have kindly contributed several different sets of Windows installation instructions, each suited to a different use-case: 

1. If you have a 64-bit installation of Windows 10 Professional or Enterprise, you can follow the [full Docker container installation](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/installation/step_by_step_Windows_Docker_install.md), which will ensure that you have all the dependencies. 
2. If you've never heard of *Anaconda* as being anything other than a snake, you can follow the simple step-by-step instructions [here](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/installation/simple_Windows_Anaconda_install.md). 
3. If you already have Anaconda or a similar Python 3 distribution set up on your machine (e.g., WinPython, Canopy), then you can install TensorFlow in a virtual environment as outlined [here](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/installation/conda_TensorFlow_install.md).

## GPU Considerations

Most of the examples in this repo involve relatively small (in Deep Learning terms) data sets so you will be in great shape using your CPU alone for training the models. That said, some of the notebook covered later in our book will train much more quickly if you employ a GPU. Alternatively, you may enjoy leveraging the efficient, highly-parallelised computations that a GPU affords for your own projects. Whatever the reason, here are TensorFlow GPU instructions for [Mac/Unix](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/installation/step_by_step_MacOSX_install.md#bonus-training-models-with-an-nvidia-gpu) or [Windows](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/installation/windows_TF_GPU.md).  

