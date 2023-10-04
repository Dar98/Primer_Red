# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 21:20:44 2023

@author: diego
"""
import mnist_loader
import random
import numpy as np

#### Funciones que se usaran
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
    """ Esta función se utiliza para convertir un dígito (0 a 9) en un vector unitario de 10 dimensiones."""

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
    """Esta función calcula el valor de la función sigmoide para un valor z. Toma cualquier número real z como 
    entrada y devuelve un valor en el rango de 0 a 1."""

def sigmoid_prime(z):
    """Esta función calcula la derivada de la función sigmoide en un punto z."""
    return sigmoid(z)*(1-sigmoid(z))

#### Función de Costo
class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Este método calcula la función de costo de entropía cruzada para una salida 'a' y las etiquetas de clase 'y'."""
        
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
        """Para evitar problemas numéricos cuando 'a' o '1 - a' son cercanos a 0, se utiliza np.nan_to_num para reemplazar 
        los valores nulos por un valor pequeño, en este caso, 0.0. El resultado es la suma de los términos calculados para 
        cada muestra en el lote de entrenamiento."""

    @staticmethod
    def delta(z, a, y):
        """Este método calcula el gradiente de la función de costo con respecto a la salida a y las etiquetas reales y. El 
        gradiente se utiliza durante el proceso de retropropagación para ajustar los pesos y sesgos de la red neuronal."""
        
        return (a-y)
        """ Formula del gradiente."""


#### Red Neuronal
class Network(object):
    
    def __init__(self, sizes, cost=CrossEntropyCost):
        """Este método se llama cuando se crea una nueva instancia de la clase. """
        
        self.num_layers = len(sizes)
        """Se inicializa con la longitud de la lista sizes, lo que significa que almacena el número total de capas en la red."""
                                     
        self.sizes = sizes
        """Almacena la lista sizes para que esté disponible en toda la instancia de la clase."""
                           
        self.default_weight_initializer()
        """Es un método que inicializa los pesos y sesgos de la red neuronal de manera aleatoria. """
                                          
        self.cost=cost
        """Define a la función de costo"""     
        
    def default_weight_initializer(self):
        """Es utilizado para inicializar los pesos y sesgos de una red neuronal antes de comenzar el entrenamiento."""
        
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        """Este atributo se inicializa como una lista de vectores numpy aleatorios. Tomando valore de una distribución gaussiana."""
        
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        """se inicializa como una lista de matrices numpy aleatorias. Cada matriz representa los pesos de las conexiones entre 
        dos capas consecutivas de neuronas en la red."""   
        
    def feedforward(self, a):
        """Se utiliza para calcular la salida de la red neuronal dada una entrada a."""
        
        for b, w in zip(self.biases, self.weights):
            """Estas listas contienen los sesgos y los pesos de cada capa de la red neuronal, respectivamente. La función zip 
            combina elementos correspondientes de ambas listas en pares (b, w)."""
            
            a = sigmoid(np.dot(w, a)+b)
            """Para cada capa de la red, se realiza el producto escalar entre la matriz de pesos w y el vector de activaciones a 
            de la capa anterior. Esto representa la suma ponderada de las entradas a cada neurona en la capa actual. Luego, se le 
            suma el vector de sesgos b. El resultado se pasa a través de la función sigmoide (o alguna otra función de activación) 
            utilizando la función sigmoid. Esto calcula las activaciones de la capa actual."""

            
        #soft-max
        e=np.exp(a)
        se=np.sum(e)
        """Esto convierte las activaciones en una distribución de probabilidad."""
        
        return e/se
        """La función devuelve la salida depués de aplicar la capa soft max.""" 
        
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False,
            early_stopping_n = 0):
        """La función SGD (Stochastic Gradient Descent) es un método utilizado para entrenar una red neuronal mediante el 
        algoritmo de descenso de gradiente estocástico en mini-lotes (mini-batches). Esta función toma varios parámetros y 
        realiza el proceso de entrenamiento de la red neuronal durante un número especificado de épocas."""
        
        training_data = list(training_data)
        """Una lista de tuplas (x, y) que representan los datos de entrenamiento, donde 'x' es la entrada y 'y' es la salida deseada."""
        
        n = len(training_data)
        """Calcula el número de datos de entrenamiento."""

        if evaluation_data:
            evaluation_data = list(evaluation_data)
            n_data = len(evaluation_data)

        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        """La función primero inicializa listas vacías para almacenar el costo y la precisión en las épocas especificadas."""
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))

            print("Epoch %s training complete" % j)

            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(self.accuracy(evaluation_data), n_data))
        print("Terminó el entrenamiento")
        """Inicia un bucle que recorre el número de épocas especificadas. En cada época, se reorganizan aleatoriamente los 
        datos de entrenamiento para garantizar que el entrenamiento sea estocástico. Los datos de entrenamiento se dividen 
        en mini-lotes del tamaño especificado (mini_batch_size), y la red neuronal se actualiza utilizando cada mini-lote 
        a través del método update_mini_batch. Después de actualizar la red con todos los mini-lotes en una época, se calculan 
        y registran el costo y la precisión en los datos de entrenamiento y, si se especifica, en los datos de evaluación. 
        Si se observa una mejora en la precisión de la evaluación, se puede implementar una lógica de detención anticipada 
        (early stopping) si el parámetro early_stopping_n está configurado adecuadamente."""
        
        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy
        """La función finaliza después de completar todas las épocas y devuelve las listas que contienen los costos y la 
        precisión en las épocas correspondientes."""
        
    def update_mini_batch(self, mini_batch, eta, lmbda, n, momentum=None):
        """Actualizar los pesos y sesgos de una red neuronal mediante el uso del algoritmo de descenso de gradiente 
        estocástico (SGD) en un solo mini-lote de ejemplos de entrenamiento."""
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        """Se inicializan dos listas de gradientes, nabla_b y nabla_w, con las mismas dimensiones que los sesgos y los
        pesos de la red neuronal, respectivamente. Estas listas se utilizan para acumular los gradientes calculados durante
        el proceso de retropropagación (backpropagation) para todos los ejemplos en el mini-lote. """
        
        for x, y in mini_batch:
                delta_nabla_b, delta_nabla_w = self.backprop(x, y)
                nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        """" Se itera a través de cada ejemplo (x, y) en el mini-lote. Para cada ejemplo, se llama al método backprop para
        calcular los gradientes del costo con respecto a los sesgos y los pesos utilizando el algoritmo de retropropagación."""

        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        
    def backprop(self, x, y):
        """Calcula los gradientes de la función de costo con respecto a los sesgos (biases) y pesos (weights) de la red"""
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        """Inicializa nabla_b y nabla_w como listas de ceros para almacenar los gradientes de los sesgos y pesos, respectivamente."""
        
        # feedforward
        activation = x
        activations = [x] 
        zs = [] 
        """Crea una lista vacio donde guardara las activaciones."""
        
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
            
        """Comienza con la entrada x. Para cada capa de la red neuronal, calcula la suma ponderada de las entradas z y aplica 
         la función de activación sigmoide para obtener la activación de esa capa. Tanto z como la activación se almacenan en 
         zs y activations, respectivamente."""
            
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        """Calcula los gradientes para la capa de salida (la última capa) y almacénalos en nabla_b[-1] y nabla_w[-1]."""

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
        """Itera en sentido inverso a través de las capas desde la segunda capa hasta la primera capa oculta. Calcula el error delta 
        para cada capa propagando el error hacia atrás a través de la red. Calcula los gradientes de sesgos y pesos para cada capa y 
        almacénalos en nabla_b y nabla_w.
        Finalmente, devuelve una tupla (nabla_b, nabla_w) que contiene los gradientes calculados."""
        
    def accuracy(self, data, convert=False): #convert=False en caso de ser datos de entrenamiento y convert=True en caso de datos de prueba
        """Esta función toma un conjunto de datos (data) y compara las salidas predichas por la red neuronal con las etiquetas reales. 
        Dependiendo del valor de convert, puede realizar o no una conversión entre las representaciones de las etiquetas. Luego, cuenta 
        cuántas predicciones son correctas y devuelve la precisión como un número entero, que representa el número de entradas en el 
        conjunto de datos para las cuales la red neuronal produjo resultados correctos."""
            
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]

        result_accuracy = sum(int(x == y) for (x, y) in results)
        return result_accuracy
    
    def total_cost(self, data, lmbda, convert=False): #convert=False en caso de ser datos de entrenamiento y convert=True en caso de datos de prueba
        """esta función calcula el costo total (o pérdida) en un conjunto de datos dado. El costo se calcula para cada entrada en el 
        conjunto de datos utilizando la función de costo especificada en self.cost.fn. Dependiendo del valor de convert, puede realizar 
        o no una conversión entre las representaciones de las etiquetas."""
        
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
            cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights) # '**' - to the power of.
        return cost
    

#Activación de la red
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
"""Carga los conjuntos de datos MNIST y luego los convierte al conjunto de entrenamiento en una lista de Python."""

net = Network([784, 50, 10], cost=CrossEntropyCost) #no de neuronas en la capa de entrada escondida y de salida
net.SGD(training_data, 30, 10, 0.1,
    evaluation_data=test_data,
    lmbda = 0.1, # this is a regularization parameter
    monitor_evaluation_accuracy=True,
    monitor_training_accuracy=True)

#una vez entrenada podemos usar esto para probarla, (comentar el sgd primero)
# validation_data=list(validation_data)
# p1=validation_data[0][0]
# print(net.feedforward(p1))