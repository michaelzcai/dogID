# dogID
Application of Michael Nielsen's neural network http://neuralnetworksanddeeplearning.com/index.html to identify dog breed from images. Currently only two breeds (cairn and border collie) for sake of training time. Images from http://vision.stanford.edu/aditya86/ImageNetDogs/.

To use, unzip shortdogimages.zip and run
```
import dog_loader
training_data, validation_data, test_data = dog_loader.load_dogs_data()
import network
net = network.Network([224*224*3, 100, 2])
net.SGD(training_data, 20, 10, 0.5, test_data=test_data)
import dog_eval
dog_eval.test('../evaldogimages', net)
```
