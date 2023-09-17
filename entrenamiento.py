import mnist_loader
import network


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data) #extraemos los datos en forma de lista
p1=training_data[0][0] #extraemos los datos de prueba
#así usamos las 784 entradas de los pixeles de la imagen y su etiqueta 
#correspondiente a cada imagen de forna [(pixeles,dígito),(pixeles,dígito),...]

#llamamos la func Network de la librería para darle estructura a la red 
net = network.Network([784, 30, 10]) #no de neuronas en la capa de entrada escondida y de salida
                                     

#usamos la funcion Stochastic Gradient Descent para empezar el proceso
#de entrenamiento de la red
#arg: datos de entrenamiento, no. de epocas, tamaño del mini-bach, datos de prueba
net.SGD(training_data, 40, 10, 0.07, test_data=test_data)