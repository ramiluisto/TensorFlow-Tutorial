# A workthrough of the TensorFlow tutorial at
# https://www.tensorflow.org/tutorials/keras/classification

# Various technical packages, some to bring python3 behaviour to python2.6. We'll keep it commented out for a bit
## absolute_import is related to the paths of imports(?)
## Division forces standard division to yield reals, i.e. now 5/2 = 2.5 and 5//2 = 2.
## Print function brings python3 behaviour to 2.6
## unicode_literals is again some kind of backporting stuff...
#### from __future__ import absolute_import, division, print_function, unicode_literals



# Importing the big stuff
import tensorflow as tf
from tensorflow import keras

# Importing the helper libraries
import numpy as np
import matplotlib.pyplot as plt

print("\n")
print("Running TF version {}.".format(tf.__version__))



# Import our testing- and training datasets
fashion_mnist = keras.datasets.fashion_mnist

# The data set is politely pre-divided, and we can just get the training- and testing
# datasets with their labels.
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#  All four objects {train,test}_{images,labels} are NumPy arrays with values going from 0 to 255. (0 = black, 255 = white)
#  More specifically,
# - train_images has dimensions (60000, 28, 28)
# - train_labels has dimensions 60000 x 1
# - test_images has dimensions (10000, 28, 28)
# - test_labels has dimensions 10000 x 1

# We next scale the images to have values in [0,1].
train_images = train_images / 255.0
test_images = test_images / 255.0


# The labels are numbers 0-9, we have to specify the human-readable labels.
class_names = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]

##################################
## The structure of the network ##
##################################

# First we set up the topology of the network. We'll construct a feed-forward network with
# an input layer of 784 neurons, a hidden layer of 128 neurons and an output layer of 10 neurons.
# Each neuron is connected to all neurons in the previous and next layers.
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),  # Change the matrix into an array of 28*28=784 neurons.
    keras.layers.Dense(128, activation='relu'), # One hidden layer of 128 neurons ReLu activation function.
    keras.layers.Dense(10,activation='softmax') # Output layer of 10 neurons corresponding to the given classes.
])

# Next we set up the parameters that define our update method (Optimizer), our error function (loss), and the
# tracking parameters (metrics).
model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])



##########################
## Training the network ##
##########################

# This constant specifies the number of epochs used in the training, i.e.\
# how many times we go through the training data.
EPOCH_NO = 10


print('\nStarting to train the network for %3.0f epochs.' % EPOCH_NO)

# Now we start the training of the network.
# We naturally use the training images and labels.
model.fit(train_images, train_labels, epochs=EPOCH_NO)


print('\nTraining complete!\n')

#########################
## Testing the network ##
#########################

# Next we test and record the network accuracy based on the testing data

print('Testing the resulting accuracy...')
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 2)
print('Test completed. Accuracy is % 2.2f.' %(test_acc))



########################
## Showcasing results ##
########################

# Take the array of prediction distributions given by the trained network.
predictions = model.predict(test_images)

# Let's first define a function to plot a given testing image together
# with a caption telling what we predict it to to be vs. what it actually is.

def plot_image(i,                   # Specifying on which image to plot
               predictions_array,   # Probability predictions of labels on a given input
               true_label,          # Actual label given by test data will be in true_label[i]
               img                  # The corresponding picture will be in img[i]
):
    # Setting the label and image data to the one we will be using, i.e.\ at index i
    true_label, img = true_label[i], img[i]

    # Pretty plotting formating
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    # Image will be in grayscale
    plt.imshow(img, cmap=plt.cm.binary)

    # Find which label maximizes the probability
    predicted_label = np.argmax(predictions_array)
    # Check if the predicition is correct and set a color accordingly
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    # Formatting the label text. Example case shuld be " Prediction, Certainty %, (real label) "
    plt.xlabel( "{} {:2.0f}% ({})".format(class_names[predicted_label],
                                          100*np.max(predictions_array),
                                          class_names[true_label],
                                          color = color
    )
    )



# Next we define a function to plot our prediction probabilities
# 
def plot_value_array(i,
                     predictions_array, # Probability predictions of labels on a given input
                     true_label         # Actual label given by test data will be in true_label[i]
):

    # Fix our data to the current index
    true_label = true_label[i]

    # Fancy plotting formatting
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color = "#777777")
    plt.ylim([0,1])

    # Finding the prediction probability maximizing label
    predicted_label = np.argmax(predictions_array)

    # Set first the color of all plots to red, and then override the correct predictions to blue.
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')



# Finally we generate a nice example of pictures, their corresponding
# estimates in probability form and if they were correct.

# Constants to alter grid dimensions.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols

# Generate the grid
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
# Populate the grid
for i in range(num_images): 
    plt.subplot(num_rows, 2*num_cols, 2*i+1)                
    plot_image(i, predictions[i], test_labels, test_images) 
    plt.subplot(num_rows, 2*num_cols, 2*i+2)                
    plot_value_array(i, predictions[i], test_labels)        
# Make it tight
plt.tight_layout()
# Show the grid
plt.show()
