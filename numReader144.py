from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot = True)

import tensorflow as tf
import numpy as np
from PIL import Image
import os.path
import sys



tf.set_random_seed(563)
#Inputs - definidos por uma matriz de n colunas(utilizando None)# e de 784 linhas, correspondentes a imagem de 28 x28 pixels
x = tf.placeholder(tf.float32, [None, 784])

# camada oculta

#w = tf.Variable(tf.random_normal([784, 10]))  #10
#w = tf.Variable(tf.random_normal([784, 15]))  #15
w = tf.Variable(tf.random_normal([784, 144]))  #144
#b = tf.Variable(tf.random_normal([10]))    #10
#b = tf.Variable(tf.random_normal([15]))   #15
b = tf.Variable(tf.random_normal([144]))   #144

m = tf.nn.sigmoid(tf.matmul(x, w) + b)

#w1 = tf.Variable(tf.random_normal([10, 10]))  #10
#w1 = tf.Variable(tf.random_normal([15, 10]))  #15
w1 = tf.Variable(tf.random_normal([144, 10]))  #30
b1 = tf.Variable(tf.random_normal([10]))

y = tf.nn.softmax(tf.matmul(m, w1) + b1)


# resultado esperado
y_ = tf.placeholder(tf.float32, [None, 10])

# função de custo
cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - y), reduction_indices = [1]))
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))


# Metodo do gradiente
#train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

# iniciar sessão e inicializar variaveis criadas
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

def treinar(qtd, batch):
    for _ in range(qtd):
        batch_xs, batch_ys = mnist.train.next_batch(batch)
        sess.run(train_step, feed_dict = {x: batch_xs , y_: batch_ys})


def precisao(): #Teste de modelo treinado
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('{0:.3f}%'.format(float(sess.run(accuracy, feed_dict = {x: mnist.test.images,y_: mnist.test.labels}))*100))


def testar_mnist_test(index):
    test_image = mnist.test.images[index]
    ti = np.array([test_image])
    feed_dict = {x: ti}    
    saida = sess.run(y, feed_dict)     

    #Mostrar hidden layer
    hidden = sess.run(m,feed_dict)      
    for i in range(hidden[0].size):
        hidden[0][i]*=255.0    
    hidden_img = Image.fromarray(hidden[0].reshape(12,12)) 
    base = 150
    wpercent = (base/float(hidden_img.size[0]))
    hsize = int((float(hidden_img.size[1])*float(wpercent)))
    hidden_img = hidden_img.resize((base,hsize),Image.ANTIALIAS)
    hidden_img.show()

    res = np.argmax(saida)
    ordered = np.argsort(saida[0])
                    
    segundo = ordered[8]
    terceiro = ordered[7]    
    print()                
    print('Esperado: '+format(np.argmax(mnist.test.labels[index])))
    print('Chute: {0}, com {1:.2f}% de certeza'.format(res,saida[0][res]*100))
    print('Outras opções...')
    print('Chute: {0}, com {1:.2f}% de certeza'.format(segundo,saida[0][segundo]*100))
    print('Chute: {0}, com {1:.2f}% de certeza'.format(terceiro,saida[0][terceiro]*100))

def testar(nomeArq):
    '''
    for i in range(30):
        print('testando caso {}...'.format(i))
        test_image = mnist.test.images[i]
        ti = np.array([test_image])
        feed_dict = {x: ti}
        saida = sess.run(y, feed_dict)        
        res = np.argmax(saida)
        
        print('Chutado:  '+format(res))
        print('esperado: '+format(np.argmax(mnist.test.labels[i])))
    '''    
    
    #with tf.gfile.Open(nomeArq, "rb") as f:
    #    test_image = extract_images(f)
    if(os.path.isfile(nomeArq)):        
        test_image = np.asarray(Image.open(nomeArq).getdata())
        test_image = test_image.astype(float)  
        if(test_image.size>784):
            print('Imagem invalida')
            return
        for i in range(test_image.size):
            test_image[i] = test_image[i]/255.0
        ti = np.array([test_image])
        #feed_dict = {x: test_image}
        feed_dict = {x: ti}

         #Mostrar hidden layer
        hidden = sess.run(m,feed_dict)      
        for i in range(hidden[0].size):
            hidden[0][i]*=255.0    
        hidden_img = Image.fromarray(hidden[0].reshape(12,12)) 
        base = 150
        wpercent = (base/float(hidden_img.size[0]))
        hsize = int((float(hidden_img.size[1])*float(wpercent)))
        hidden_img = hidden_img.resize((base,hsize),Image.ANTIALIAS)
        hidden_img.show()

        saida = sess.run(y, feed_dict)        
        res = np.argmax(saida)
        ordered = np.argsort(saida[0])
        #print(ordered)
        segundo = ordered[8]
        terceiro = ordered[7]    
        print()
        print("Camada de saida")
        print(saida[0])        
        print('Chute: {0}, com {1:.2f}% de certeza'.format(res,saida[0][res]*100))
        print('Outras opções...')
        print('Chute: {0}, com {1:.2f}% de certeza'.format(segundo,saida[0][segundo]*100))
        print('Chute: {0}, com {1:.2f}% de certeza'.format(terceiro,saida[0][terceiro]*100))      
    else:
        print('Imagem nao encontrada')


def main():
    esc=''
    while (esc != '0'):
        print('=== Reconhecedor de digitos ===')
        print('[1]-Treinar')
        print('[2]-Precisão')
        print('[3]-Testar imagem (28x28,GrayScale)')
        print('[4]-Testar Mnist(index)')
        print('[5]-Salvar')
        print('[6]-Restaurar')
        print('[7]-Limpar')
        print('[0]-Sair')        
        esc = input('Escolha: ')
        if (esc == '1'):            
            tamanho = int(input('Quantidade de testes (max=55000): '))
            if (tamanho > 55000 and tamanho < 1000):
                tamanho = 1000            
            batch = int(input('Tamanho do \'batch\': '))
            if ((batch > 1000 and batch < 10) or(batch > tamanho)):
                batch = 1000
            treinar(tamanho, batch)
            print ("\033c")
        if (esc == '2'):
            precisao()
            input('...')
            print ("\033c")
        if (esc == '3'):            
            nome = input('Nome do arquivo (com extensão): ')
            testar(nome)
        if (esc == '4'):
            index = int(input("Numero do caso: "))
            if(index <0 or index >=10000):
                print('Caso invalido')
            else:
                testar_mnist_test(index)
        if (esc == '5'):
            nome = input("Insira o nome do arquivo(extensão '.cktp' será adicionada automaticamente): ")
            nome = os.getcwd()+'/'+nome + '.cktp'
            saver.save(sess, nome)
            print ("\033c")            
        if (esc == '6'):
            nome = input("Insira o nome do arquivo (sem extensão): ")
            nome = os.getcwd()+'/'+nome + '.cktp'
            saver.restore(sess, nome)
            print ("\033c")            
        if (esc == '7'):
            print ("\033c")            


print ("\033c")
main()
