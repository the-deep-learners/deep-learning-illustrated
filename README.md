# Deep Learning Illustrated (2019)

This repository is home to the code that accompanies [Jon Krohn](https://www.jonkrohn.com/), [Grant Beyleveld](http://grantbeyleveld.com/about/) and [Aglaé Bassens](https://www.aglaebassens.com/)' book [Deep Learning Illustrated](https://www.deeplearningillustrated.com/). This visual, interactive guide to artificial neural networks was published on Pearson's Addison-Wesley imprint in 2019. 

## Installation

Step-by-step guides for running the code in this repository can be found in the [installation directory](https://github.com/the-deep-learners/deep-learning-illustrated/tree/master/installation). For installation difficulties, please consider visiting our book's [Q&A forum](https://groups.google.com/forum/#!forum/deep-learning-illustrated) instead of creating an _Issue_.

## Notebooks

All of the code covered in the book can be found in [the notebooks directory](https://github.com/the-deep-learners/deep-learning-illustrated/tree/master/notebooks) as [Jupyter notebooks](http://jupyter.org/). Note that while TensorFlow 2.0 was released after the book had gone to press, as detailed in Chapter 14 (specifically, Example 14.1), all of our notebooks can be trivially converted into TensorFlow 2.x code if desired. 

Below is the book's table of contents with links to all of the individual notebooks: 

### Part 1: Introducing Deep Learning

#### Chapter 1: Biological and Machine Vision

* Biological Vision
* Machine Vision
	* The Neocognitron
	* LeNet-5
	* The Traditional Machine Learning Approach
	* ImageNet and the ILSVRC
	* AlexNet
* TensorFlow PlayGround
* The _Quick, Draw!_ Game

#### Chapter 2: Human and Machine Language

* Deep Learning for Natural Language Processing
	* Deep Learning Networks Learn Representations Automatically
	* A Brief History of Deep Learning for NLP
* Computational Representations of Language
	* One-Hot Representations of Words
	* Word Vectors
	* Word Vector Arithmetic
	* word2viz
	* Localist Versus Distributed Representations
* Elements of Natural Human Language
* Google Duplex

#### Chapter 3: Machine Art

* A Boozy All-Nighter
* Arithmetic on Fake Human Faces
* Style Transfer: Converting Photos into Monet (and Vice Versa)
* Make Your Own Sketches Photorealistic
* Creating Photorealistic Images from Text
* Image Processing Using Deep Learning

#### Chapter 4: Game-Playing Machines

* Deep Learning, AI, and Other Beasts
	* Artificial Intelligence
	* Machine Learning
	* Representation Learning
	* Artificial Neural Networks
* Three Categories of Machine Learning Problems
	* Supervised Learning
	* Unsupervised Learning
	* Reinforcement Learning
* Deep Reinforcement Learning
* Video Games
* Board Games
	* AlphaGo
	* AlphaGo Zero
	* AlphaZero
* Manipulation of Objects
* Popular Reinforcement Learning Environments
	* OpenAI Gym
	* DeepMind Lab
	* Unity ML-Agents
* Three Categories of AI
	* Artificial Narrow Intelligence
	* Artificial General Intelligence
	* Artificial Super Intelligence

### Part II: Essential Theory Illustrated

#### Chapter 5: The (Code) Cart Ahead of the (Theory) Horse

* Prerequisites
* Installation
* A Shallow Neural Network in Keras ([shallow_net_in_keras.ipynb](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/shallow_net_in_keras.ipynb))
	* The MNIST Handwritten Digits ([mnist_digit_pixel_by_pixel.ipynb](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/mnist_digit_pixel_by_pixel.ipynb))
	* A Schematic Diagram of the Network
	* Loading the Data
	* Reformatting the Data
	* Designing a Neural Network Architecture
	* Training a Deep Learning Model

#### Chapter 6: Artificial Neurons Detecting Hot Dogs

* Biological Neuroanatomy 101
* The Perceptron 
	* The Hot Dog / Not Hot Dog Detector
	* The Most Important Equation in the Book
* Modern Neurons and Activation Functions 
	* Sigmoid Neurons ([sigmoid_function.ipynb](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/sigmoid_function.ipynb))
	* Tanh Neurons 
	* ReLU: Rectified Linear Units
* Choosing a Neuron

#### Chapter 7: Artificial Neural Networks

* The Input Layer
* Dense Layers
* A Hot Dog-Detecting Dense Network 
	* Forward Propagation through the First Hidden Layer
	* Forward Propagation through Subsequent Layers
* The Softmax Layer of a Fast Food-Classifying Network ([softmax_demo.ipynb](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/softmax_demo.ipynb))
* Revisiting our Shallow Neural Network

#### Chapter 8: Training Deep Networks

* Cost Functions 
	* Quadratic Cost ([quadratic_cost.ipynb](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/quadratic_cost.ipynb))
	* Saturated Neurons
	* Cross-Entropy Cost ([cross_entropy_cost.ipynb](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/cross_entropy_cost.ipynb))
* Optimization: Learning to Minimize Cost 
	* Gradient Descent
	* Learning Rate ([measuring_speed_of_learning.ipynb](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/measuring_speed_of_learning.ipynb))
	* Batch Size and Stochastic Gradient Descent
	* Escaping the Local Minimum
* Backpropagation
* Tuning Hidden-Layer Count and Neuron Count
* An Intermediate Net in Keras ([intermediate_net_in_keras.ipynb](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/intermediate_net_in_keras.ipynb))

#### Chapter 9: Improving Deep Networks

* Weight Initialization ([weight_initialization.ipynb](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/weight_initialization.ipynb))
	* Xavier Glorot Distributions
* Unstable Gradients 
	* Vanishing Gradients
	* Exploding Gradients
	* Batch Normalization
* Model Generalization — Avoiding Overfitting 
	* L1 and L2 Regularization
	* Dropout
	* Data Augmentation
* Fancy Optimizers
	* Momentum
	* Nesterov Momentum
	* AdaGrad
	* AdaDelta and RMSProp
	* Adam
* A Deep Neural Network in Keras ([deep_net_in_keras.ipynb](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/deep_net_in_keras.ipynb))
* Regression ([regression_in_keras.ipynb](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/regression_in_keras.ipynb))
* TensorBoard ([deep_net_in_keras_with_tensorboard.ipynb](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/deep_net_in_keras_with_tensorboard.ipynb))

### Part III: Interactive Applications of Deep Learning

#### Chapter 10: Machine Vision

* Convolutional Neural Networks 
	* The Two-Dimensional Structure of Visual Imagery
	* Computational Complexity
	* Convolutional Layers
	* Multiple Filters
	* A Convolutional Example
	* Convolutional Filter Hyperparameters
	* Stride Length
	* Padding
* Pooling Layers
* LeNet-5 in Keras ([lenet_in_keras.ipynb](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/lenet_in_keras.ipynb))
* AlexNet ([alexnet_in_keras.ipynb](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/alexnet_in_keras.ipynb)) and VGGNet ([vggnet_in_keras.ipynb](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/vggnet_in_keras.ipynb))
* Residual Networks 
	* Vanishing Gradients: The Bête Noire of Deep CNNs
	* Residual Connection
* Applications of Machine Vision
	* Object Detection
	* Image Segmentation
	* Transfer Learning ([transfer_learning_in_keras.ipynb](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/transfer_learning_in_keras.ipynb))
	* Capsule Networks

#### Chapter 11: Natural Language Processing

* Preprocessing Natural Language Data ([natural_language_preprocessing.ipynb](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/natural_language_preprocessing.ipynb))
	* Tokenization
	* Converting all Characters to Lower Case
	* Removing Stop Words and Punctuation
	* Stemming; Handling *n*-grams
	* Preprocessing the Full Corpus
* Creating Word Embeddings with word2vec
	* The Essential Theory Behind word2vec
	* Evaluating Word Vectors
	* Running word2vec
	* Plotting Word Vectors
* The Area Under the ROC Curve
	* The Confusion Matrix
	* Calculating the ROC AUC Metric
* Natural Language Classification with Familiar Networks
	* Loading the IMDB Film Reviews
	* Examining the IMDB Data
	* Standardizing the Length of the Reviews
	* Dense Network ([dense_sentiment_classifier.ipynb](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/dense_sentiment_classifier.ipynb))
	* Convolutional Networks ([convolutional_sentiment_classifier.ipynb](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/convolutional_sentiment_classifier.ipynb))
* Networks Designed for Sequential Data 
	* Recurrent Neural Networks ([rnn_sentiment_classifier.ipynb](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/rnn_sentiment_classifier.ipynb))
	* Long Short-Term Memory Units ([lstm_sentiment_classifier.ipynb](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/lstm_sentiment_classifier.ipynb))
	* Bidirectional LSTMs ([bi_lstm_sentiment_classifier.ipynb](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/bi_lstm_sentiment_classifier.ipynb))
	* Stacked Recurrent Models ([stacked_bi_lstm_sentiment_classifier.ipynb](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/stacked_bi_lstm_sentiment_classifier.ipynb))
	* Seq2seq and Attention
	* Transfer Learning in NLP
* Non-Sequential Architectures: The Keras Functional API ([multi_convnet_sentiment_classifier.ipynb](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/multi_convnet_sentiment_classifier.ipynb))

#### Chapter 12: Generative Adversarial Networks

* Essential GAN Theory
* The _Quick, Draw!_ Dataset
* The Discriminator Network
* The Generator Network
* The Adversarial Network
* GAN Training ([generative_adversarial_network.ipynb](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/generative_adversarial_network.ipynb))

#### Chapter 13: Deep Reinforcement Learning

* Essential Theory of Reinforcement Learning 
	* The Cart-Pole Game
	* Markov Decision Processes
	* The Optimal Policy
* Essential Theory of Deep Q-Learning Networks
	* Value Functions
	* Q-Value Functions
	* Estimating an Optimal Q-Value
* Defining a DQN Agent ([cartpole_dqn.ipynb](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/cartpole_dqn.ipynb))
	* Initialization Parameters
	* Building the Agent’s Neural Network Model
	* Remembering Gameplay
	* Training via Memory Replay
	* Selecting an Action to Take
	* Saving and Loading Model Parameters
* Interacting with an OpenAI Gym Environment
* Hyperparameter Optimization with SLM Lab
* Agents Beyond DQN 
	* Policy Gradients and the REINFORCE Algorithm
	* The Actor-Critic Algorithm

### Part IV: You and AI

#### Chapter 14: Moving Forward with Your Own Deep Learning Projects

* Ideas for Deep Learning Projects
	* Machine Vision and GANs ([fashion_mnist_pixel_by_pixel.ipynb](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/fashion_mnist_pixel_by_pixel.ipynb))
	* Natural Language Processing
	* Deep Reinforcement Learning
	* Converting an Existing Machine-Learning Project
* Resources for Further Projects 
	* Socially-Beneficial Projects
* The Modeling Process, including Hyperparameter Tuning 
	* Automation of Hyperparameter Search
* Deep Learning Libraries
	* Keras and TensorFlow ([deep_net_in_tensorflow.ipynb](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/deep_net_in_tensorflow.ipynb))
	* PyTorch ([pytorch.ipynb](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/pytorch.ipynb))
	* MXNet, CNTK, Caffe, and Beyond
* Software 2.0
* Approaching Artificial General Intelligence

## Book Cover

![](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/img/cover.jpeg)

