# Installation Instructions

The easy, on-the-fly way to use the Jupyter notebooks provided in this repository is to execute them in [Colab](https://colab.research.google.com/notebooks/welcome.ipynb). 
To make this as simple as possible, we've included `Open in Colab` buttons at the top of individual notebooks. 
Click on these buttons and you'll be transported to an environment where you can execute the notebooks instantaneously (and for no charge) in the Google Cloud, including on high-performance hardware like GPUs and TPUs (select `Runtime` from the Colab menu bar, `Change runtime type`, and then change `Hardware accelerator` from `None` to `GPU` or `TPU`).

The major shortcoming of using Colab is that we can't control or freeze the versions of software libraries it uses by default and so some code may break in the future (indeed, this is likely).
**If you'd like to be 100% sure that the Jupyter notebooks in this repo run as we intended, then follow the installation instructions for the operating system of your choosing below.**


## macOS

Detailed step-by-step instructions for running the code notebooks for on a Mac can be found [here](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/installation/step_by_step_MacOSX_install.md).


## Unix

#### Where You Already Have the Dependencies

The dependencies are provided in this repository's [Dockerfile](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/Dockerfile). If you have these packages configured as you like them, you can simply `git clone https://github.com/the-deep-learners/deep-learning-illustrated`.

#### Where You Are Missing Dependencies

1. Get Docker CE for, e.g., [Ubuntu](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/)
2. Follow all the steps in our [Step-by-Step Instructions for Mac](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/installation/step_by_step_MacOSX_install.md) that involve executing code at the command line. That is, execute all steps but one, four and five. 

## Windows

Community members have kindly contributed different Windows installation instructions, different use-cases: 

1. If you have a 64-bit installation of Windows 10 Professional or Enterprise, you can follow the [full Docker container installation](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/installation/step_by_step_Windows_Docker_install.md), which will ensure that you have all the dependencies. 
2. Otherwise, you try the simple step-by-step instructions [here](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/installation/simple_Windows_Anaconda_install.md). 


