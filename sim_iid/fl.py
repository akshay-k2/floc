import numpy as np

def softmax(x):
    x = np.array(x)
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return (e_x / e_x.sum(axis=1)[:,None])

def Relu(x):
    return np.maximum(0,x)

def dxRelu(x):
        return np.where(x >= 0, 1, 0)

def Forward(X,T,config):
    '''Adding the output matrix of every step of the process into a dictionary
    {"a1": [...], "z2": [...], "a2": [...], "z3": [...], ..... ,"h":[...]}'''
    X = np.matrix(X)
    m = X.shape[0]
    att = config[5] #activations
    #print(att)
    Thetas = T
    Forward_steps = {}
    Forward_steps['a1'] = X
    Lastlayer = int(len(config[4])) #nn_sizes
    
    for layer in range(1,Lastlayer):
        #print(att[layer-1])
        Forward_steps[f'z{layer+1}'] = np.dot(Forward_steps[f'a{layer}'], Thetas[f'T{layer}'])
        if att[layer-1] == 'softmax':
            Forward_steps[f'a{layer+1}'] = np.concatenate((np.ones([m,1]), softmax(Forward_steps[f'z{layer+1}'])), axis=1)
        elif att[layer-1] == 'relu':
                Forward_steps[f'a{layer+1}'] = np.concatenate((np.ones([m,1]), Relu(Forward_steps[f'z{layer+1}'])), axis=1)
        else:
            print('ERROR')

        
    h = Forward_steps.pop(f'a{Lastlayer}')
    Forward_steps['h'] = h[:,1:]
        
    return Forward_steps

def CostFunction(X,Y,T,config):
    '''Calculates the logistic cost function for every class and every row'''
    Thetas = T
    m = X.shape[0]
    Reg = config[3]
    soma_weights = 0
    for i in range(len(Thetas)):
        weights = Thetas[f'T{i+1}']
        weights[0] = 0
        soma_weights += np.sum(weights**2)
    Forward_dict = Forward(X,Thetas,config)
    h = Forward_dict['h']
    soma = np.sum((np.multiply(-Y , np.log(h)) - np.multiply((1-Y),(np.log(1-h)))))
    J = soma/m + (Reg/(2*m)) * soma_weights
    return J

def Gradients(X,Y,T,config):
    '''Calculates derivative of thetas w.r.t. cost function ans is organized
    inside a dictionary containing the deltas
    "delta<sum>" is the derivative of "T<num>"
    {...., "delta2": [...], "delta1": [...]}'''
    X = np.matrix(X)
    Y = np.matrix(Y)
    m = X.shape[0]
    Thetas = T
    n_layers = len(config[4]) # nn_sizes
    att = config[5] # activations
    Thetas_grad = []

    Forward_list = Forward(X,Thetas,config)
    deltas = {}
    deltas[f'delta{n_layers}'] = Forward_list['h'] - Y # delta4

    for i in range(n_layers-1,1,-1):# 3 ... 2
        if att[i-2] == 'relu':
            deltas[f'delta{i}'] = np.multiply((np.dot(deltas[f'delta{i+1}'],Thetas[f'T{i}'][1:].T)) , dxRelu(Forward_list[f'z{i}']))
        
        for c in range(len(deltas)):#0 ... 1 ... 2
            BigDelta = np.array(np.dot(deltas[f'delta{c+2}'].T, Forward_list[f'a{c+1}']))
            weights = Thetas[f'T{c+1}']
            weights[0] = 0 #Coluna dos bias vira 0's
            grad = np.array(BigDelta + (config[3] * weights.T))/m
            Thetas_grad.append(grad)

        return Thetas_grad #[T1_grad, T2_grad, T3_grad]

def accuracy(X,Y,Thetas,config):
    '''Percentage of correct classification'''
    Forward_list = Forward(X,Thetas,config)
    h = Forward_list['h']
    y_hat = np.argmax(h, axis=1)[:,None]
    y = np.argmax(Y, axis=1)[:,None]
    return np.mean(y_hat == y)
