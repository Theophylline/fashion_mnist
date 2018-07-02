VGG-like CNN for Fashion-MNIST

Fashion-MNIST is an alternative to the more commonly used MNIST dataset. It contains a set of 28x28 grayscale images, each of which is assigned with one of the following labels:
* T-shirt/top
* Trouser
* Pullover
* Dress
* Coat
* Sandal
* Shirt
* Sneaker
* Bag
* Ankle boot

Since fashion-MNIST is a more challenging image recognition task compared to MNIST, models will generally perform worse on this dataset. Below is a summary of a VGG-like CNN model that achieves 93.8% accuracy on the test set (note that the number of trainable parameters are decreased and dropout layers are added to avoid overfitting):
