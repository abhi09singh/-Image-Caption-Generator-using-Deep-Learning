# IMAGE CAPTION GENERATOR USING DEEP LEARNING

## Objectives
* To recognize the context of an image and describe them in English.

## Block Diagram
![](/images/systemdiagram.PNG)
The objective of this project is to develop an image caption generator which can generate natural language captions for images implementing a CNN-RNN model.  
The task of image captioning can be divided into two modules logically – one is an image based model – which extracts the features and nuances out of our image, and the other is a language based model – which translates the features and objects given by our image based model to a natural sentence.


For our image based model (viz encoder) – we usually rely on a Convolutional Neural Network model. And for our language based model (viz decoder) – we rely on a Recurrent Neural Network. The image below summarizes the approach given above.

![alt text](https://raw.githubusercontent.com/yunjey/pytorch-tutorial/master/tutorials/03-advanced/image_captioning/png/model.png)

Here I'm using a pre-trained [VGG16](https://drive.google.com/file/d/1UyRuoLBD_leyeyRgzyz0vBSXk9A7bw9x/view?usp=drive_link) model with weights as my CNN which will act as an input to my RNN(LSTM)

The Dataset which I used for this project is [Flickr30k](http://shannon.cs.illinois.edu/DenotationGraph/) , which is widey used dataset for image captioning and can also be applied in this case.
The dataset consist of 31783 images of which 27015 images are used to train the model and rest to test and validate.

## RESULTS
* Generated captions on test images:
- ![](/images/img1.png)
- ![](/images/img2.png)

#### UI was created using Flask.
Screenshot:
- ![](/images/f1.png)
- ![](/images/f2.png)

## Requirements
1. Tensorflow
2. Anaconda
3. Python 
4. Flask
5. Numpy
6. Pillow
