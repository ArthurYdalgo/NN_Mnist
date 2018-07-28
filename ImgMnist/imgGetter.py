from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot = True)
import scipy.misc
import numpy as np
import sys

num=0

#image to download
tag = int(sys.argv[1])


for i in range(10000):
    if(np.argmax(mnist.test.labels[i])==tag):
        nome = format(tag)+'_'+format(num)+'.png'
        test_image = mnist.test.images[i].reshape(28,28)
        if(len(sys.argv)>2):
            scipy.misc.toimage(test_image).save(sys.argv[2]+'/'+nome)
        else:
            scipy.misc.toimage(test_image).save(nome)

        num+=1
