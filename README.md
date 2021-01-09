# ICV-project


## Introduction
Computer vision and Machine learning have been a hot topic in research in the past decade. The substantial evolvement of technology and rapidly improving computation resources has allowed us to process a large amount of data and derive practical insights from them. Advanced AI models are taking over many industries and performing human tasks with high accuracy and precision. Food classification is one of the applications of image processing. There are quite a number of researches and Machine Learning solutions already available when it comes to classifying food items from images. However, identifying food items and classifying them directly in a video is a slightly more complicated problem which can be approached in many different ways.

## Related work
There have been various food classifications already done. Some of those works are briefly reported below. We actually took insights from some of them for our project:

- [Food Image Classification with Convolutional Neural Network](https://ieeexplore.ieee.org/document/8550005) 
- [Analysis of food images: Features and Classification](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5448982/)
- [Multiclass Food Classification](https://www.kaggle.com/theimgclist/multiclass-food-classification-using-tensorflow)
- [Build Video Classification Model](https://www.analyticsvidhya.com/blog/2019/09/step-by-step-deep-learning-tutorial-video-classification-python/)




## Implemented Model (with link to github)
The entire structure consists of two parts: an image classifier and a video stream processor.

The image classifier is trained on top of the Inceptionv3 model with ImageNet weights so as to leverage the benefits of transfer learning. To train this classifier, we used the Food 101 dataset. Here’s a quick overview of the dataset:
	Total classes: 101,	Total images: 101000
	Training images: 75750,	Test images: 25250
We finetune the model with the following settings:
batch_size = 32

Two dense layers, one with “Softmax” activation function and L2 regularizer with a value of 0.005, another with “relu” activation function and a dropout of 0.2.
The model is compiled with an SGD (Stochastic Gradient Descent) optimizer with hyperparameters: (learning rate) lr=0.0001, and momentum=0.9.

Given the huge size of dataset (approximately 5 GB), the model was trained for only 10 epochs which took almost 6 hours even in Google Colaboratory with GPU runtime enabled.


## Reference

Food-101 Dataset
- [Food-101 – Mining Discriminative Components with Random Forests](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)

Weight Intializers: 
- [Hyper-parameters in Action! Part II — Weight Initializers](https://towardsdatascience.com/hyper-parameters-in-action-part-ii-weight-initializers-35aee1a28404)
- [Priming neural networks with an appropriate initializer.](https://becominghuman.ai/priming-neural-networks-with-an-appropriate-initializer-7b163990ead)

Optimizers:
- [An overview of gradient descent optimization algorithms](http://ruder.io/optimizing-gradient-descent/index.html#visualizationofalgorithms)

Batch Normalization:
- [Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift](https://arxiv.org/abs/1801.05134)
- [Glossary of Deep Learning: Batch Normalisation](https://medium.com/deeper-learning/glossary-of-deep-learning-batch-normalisation-8266dcd2fa82)

Activation Functions:
- [Understanding Activation Functions in Neural Networks](https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0)

Grad CAM:
- [Grad-CAM: Why did you say that? Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/pdf/1610.02391v1.pdf)
- [Deep Learning with Python Book by Francois Chollet](http://www.deeplearningitalia.com/wp-content/uploads/2017/12/Dropbox_Chollet.pdf)

Regularization: 
- [L1 and L2 Regularization Methods](https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c)
