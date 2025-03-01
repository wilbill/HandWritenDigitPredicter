'''
Image Recognition with Machine Learning
/
Here, we will be writing a basic convolutional neural network (CNN) for handwritten digit recognition of the MNIST dataset. I will take an in-depth look at a model based off of Yann LeCun's LeNet-5.
One of the simplest tasks we can perform is handwritten digit recognition. Given an image of a handwritten digit (i.e., 0, 1, …, 9), we want our model to be able to correctly classify its numeric value. Though this task seems relatively simple, it is actually used fairly often in real life, such as automatically extracting credit card numbers from a picture. The dataset we will use for digit recognition is the MNIST dataset, which is the de facto dataset used for machine learning-based digit recognition.
Initialization
Talk about the MNIST dataset along with the input and output sizes for the data.

I will talk about the MNIST database
The MNIST (Modified National Institute of Standards and Technology) database contains 60,000 training examples and 10,000 testing examples. The database contains grayscale handwritten digits that were resized to fit in a 20x20 pixel box, which was then centered in a 28x28 image (padded with whitespace). The images were resized using an anti-aliasing technique to minimize image distortion.

Examples from the MNIST database.

Examples from the MNIST database.
The database is normalized to have floating point values between 0.0 and 1.0. In this case, 0.0 corresponds to a grayscale pixel value of 255 (pure white), while 1.0 corresponds to a grayscale pixel value of 0 (pure black).

B. Inputs and labels
Since each grayscale image has dimensions 28x28, there are 784 pixels per image. Therefore, each input image corresponds to a tensor of 784 normalized floating point values between 0.0 and 1.0. The label for an image is a one-hot tensor with 10 classes (each class represents a digit). In terms of our code, we have input_dim = 28 and output_size = 10 for the MNIST dataset.

When we use a batch of input data, the shape of inputs is (batch_size, self.input_dim**2) and the shape of labels is (batch_size, self.output_size), where batch_size represents the size of the batch.

The shape of inputs is (batch_size, self.input_dim**2) because there are batch_size images and each image is a square with a width of input_dim pixels.
'''


import tensorflow as tf
class MNISTModel(object):
    #Model initialisation
    def __init__(self, input_dim, output_size):
        self.input_dim = input_dim
        self.output_size = output_size

    #Reshaping the image data from basic 2D Matrix format to NHWC Format
    #NHWC has a shape with 4 dimensions (1. No. of image data samples(batch_size), 2. height of each image, 3. width of each image, 4. channels per image)
    #height n width of each image ids self.input_dim, no. of channels=1 since grayscale, batch sample is unspecified coz we allow variable no. of i/p images
    #The new shape must be able to contain all the elements from the input tensor. For example, we could reshape a tensor from original shape (5, 4, 2) to (4, 10), since both shapes contain 40 elements. However, we could not reshape that tensor to (3, 10, 2) or (1, 10), since those shapes contain 60 and 10 elements, respectively.
    #We are allowed to use the special value of -1 in at most one dimension of the new shape. The dimension with -1 will take on the value necessary to allow the new shape to contain all the elements of the tensor. Using the previous example, if we set a new shape of (1, 10, -1), then the third dimension will have size 4 in order to contain the 40 elements. Since the batch size of our input image data is unspecified, we use -1 for the first dimension when reshaping inputs.
    def model_layers(self, inputs, is_training):
        reshaped_inputs = tf.reshape(inputs, [-1, self.input_dim, self.input_dim, 1])
         #Applying Convolution Layer to the reshaped_inputs
        conv1 = tf.keras.layers.Conv2D(
            filters=32,                        #filters: The number of filters to use.
            kernel_size=[5, 5],                #kernel_size: The height and width dimensions of the kernel matrix.
            padding='same',                    #padding: Either 'valid' (no padding) or 'same' (padding).
            activation='relu',                 #activativation function to be used, defaults to None, we can also use tf.nn.relu
            name='conv1')(reshaped_inputs)     #The name for the convolution layer (useful for debugging and visualization).

        pool1 = tf.keras.layers.MaxPooling2D( #We can use pooling to reduce the size of the data in the height and width dimensions. This allows the model to perform fewer computations and ultimately train faster. It also prevents overfitting, by extracting only the most salient features
            pool_size=[2, 2],
            strides=2, #stride size for the kernel matrix. Can be a single integer (same stride size for vertical/horizontal) or a tuple of 2 integers (manually specified vertical/horizontal stride sizes).
            name='pool1')(conv1)

        #adding another convolution layer, with pool1 as the i/p => more layers more distiguishing e.g we can be able to get exact curve of 7 if we add more layers
        conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu,
            name='conv1' )(pool1)
        #2nd pooling layer
        pool2 = tf.keras.layers.MaxPooling2D(
            pool_size=[2, 2],
            strides=2,
            name='pool2')(conv2) #conv2 as the input here

        #we'll create a helper function for the model_layers function. The helper, create_fc, applies a fully-connected layer to the final max-pooling layer of the model.
        #We first need to get the height, width, and number of channels in pool2 before we flatten it.
        hwc = pool2.shape.as_list()[1:] #sliced from index 1 to the end., ie # Get [height, width, channels] original without slicing was (batch_size, height, width, channels),
        flattened_size = hwc[0] * hwc[1] * hwc[2] #flattened_size equal to the product of the integers in hwc. Since fully connected layers expect a 1D input, the feature map needs to be flattened into a single vector.
        pool2_flat = tf.reshape(pool2, [-1, flattened_size]) #-1: This automatically infers the batch size, allowing the model to process variable batch sizes.
        #fully connected layer (Dense layer) => It connects every neuron from the previous layer (flattened pool2 output) to 1024 new neuron
        dense = tf.keras.layers.Dense(
                                      units=1024, #Defines the number of neurons in the dense layer.
                                      activation=tf.nn.relu, #Uses ReLU (Rectified Linear Unit) as the activation function.
                                      name='dense')(pool2_flat) #Assigns a name to the layer for easier debugging and visualization.
        #To prevent overfitting in large neural networks
        dropout = tf.keras.layers.Dropout(
            dense,
            training=is_training)(rate=0.4) #default rate=0.5, default training= false

        logits = tf.keras.layers.Dense( #The logits layer is the final output layer of a neural network before applying an activation function like softmax. It produces raw score values that represent the model's confidence in each class.
            units=self.output_size,
            name='logits')(dropout)
        return logits




