from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot = True)
import scipy.misc
import numpy as np
import sys

num=0

for i in range(10000):   
    tag = np.argmax(mnist.test.labels[i])
    nome = format(tag)+'_'+format(num)+'.png'
    test_image = mnist.test.images[i].reshape(28,28)    
    scipy.misc.toimage(test_image).save(format(tag)+'/'+nome)
    num+=1
