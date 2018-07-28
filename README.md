# NN_Mnist
Tensor Flow based Neural Network - Reads numbers from 28x28pixels greyscale images, and also from Mnist database

# Train (Treinar)
Selects labeled examples from mnist.train... (All the saved weights and biases avaiable to be restored were trained 
with all the examples (55000) ,using a 100 sized batch)

# Accuracy (Precisão)
Tests the perfomance from mnist.test

# Test image (Testar imagem...)
Tests custom image. 
->Notes: 
-If the image is not in the same directory, specify it.
-Include the image extension. 
-Image must be 28x28 greyscale

# Test Mnist (Testar Mnist)
Select an example from mnist.test... This is made to cut off the jobs of accessing the image from the ImgMnist/Test folder. 
Simply use the index of example (eg: for '1_14.png', use 14). There are 10000 tests avaible, labeled from 0 to 9999
->Note: if, for some reason, you want to test from the train database, replace the line 66 
'test_image = mnist.test.images[index]' for this: 'test_image = mnist.train.images[index]'
(training data is labeled from 0 to 54999)

# Saving (Salvar)
In order to save time and not train the network every time you open the program, save it... 3 files will be generated.
->Note: This saved example will work with the specific structures of the network (same amout of layers, nodes in each 
layers, etc...), in order to keep things clear, I'm using the current "labeling" system:
'3lX_Y' -> the used network has 3 layes, being 784 fixed nodes in the input layer, X represents the amount of nodes in the
hidden layer, and Y the accuracy of the network (merely for keep things easier to find).
All the 3 generated files have this label as name, with varying extensions.

# Restore (Restaurar)
In order to save time and not train the network every time you open the program, restore it...
->Note: This restored example will work with the specific structures of the network (same amout of layers, nodes in each 
layers, etc...), in order to keep things clear, I'm using the current "labeling" system (which is used to restore the
network:
'3lX_Y' -> the used network has 3 layes, being 784 fixed nodes in the input layer, X represents the amount of nodes in the
hidden layer, and Y the accuracy of the network (merely for keep things easier to find).

# Clear (Limpar)
I have an issue with a clear screen. This option is for my OCD only.

