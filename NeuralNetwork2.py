
import numpy as np
import random
import nnfs 
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data
#np.random.seed(0)

nnfs.init()

X,y=spiral_data(samples=100, classes=3)  # X feature , y label data or classification


class Layer_dense:
    
    def __init__(self, n_inputs,n_neurons):
        self.weights=0.1*np.random.randn (n_inputs,n_neurons) # made a random weights
        self.biases=np.zeros((1,n_neurons))   # made a biases 
        
    def forward(self, inputs):
        self.output=np.dot(inputs,self.weights)+self.biases
        
class Activation_ReLU:
    def forward(self,inputs):
        self.output=np.maximum(0,inputs)
        
class Activation_Softmax:
    def forward(self,inputs):
        exp_values=np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        Prob_values=exp_values/np.sum(exp_values,axis=1,keepdims=True)
        self.output=Prob_values


layer1=Layer_dense(2, 5 )
layer2=Layer_dense(5, 3)
Activation1=Activation_Softmax()
Activation2=Activation_Softmax()

layer1.forward(X)
print('After Activation')
Activation1.forward(layer1.output)
#layer 2
layer2.forward(Activation1.output)
Activation2.forward(layer2.output)

plt.polar(layer1.output,"*")
plt.show()
plt.polar(Activation1.output,"*")
plt.show()
plt.polar(layer2.output,"*")
plt.show()
plt.polar(Activation2.output,"*")
plt.show()
print(Activation1.output)
