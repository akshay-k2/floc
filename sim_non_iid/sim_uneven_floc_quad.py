import numpy as np
import time
import random
import fl

x_train = {}
y_train = {}
x_test  = {}
y_test  = {}

num_of_nodes = 4

x_train_merged  = np.load('even_np_datasets/central_x_train.npy')
y_train_merged  = np.load('even_np_datasets/central_y_train.npy')
x_test_merged  = np.load('even_np_datasets/central_x_test.npy')
y_test_merged  = np.load('even_np_datasets/central_y_test.npy')

m = x_train_merged.shape[0]
n = x_train_merged.shape[1] + 1
x_train_merged = np.concatenate((np.ones([m,1]),x_train_merged), axis=1) 

cat = np.zeros([m,10])
for ind, num in enumerate(y_train_merged):
    cat[ind][num] = 1
y_train_merged = cat

m = x_test_merged.shape[0]
n = x_test_merged.shape[1] + 1
x_test_merged = np.concatenate((np.ones([m,1]),x_test_merged), axis=1) 

cat = np.zeros([m,10])
for ind, num in enumerate(y_test_merged):
    cat[ind][num] = 1
y_test_merged = cat

#print('New Shapes:')
#print(f'- X Train 0: {x_train[0].shape[0]} x {x_train[0].shape[1]}')
#print(f'- Y train 0: {y_train[0].shape[0]} x {y_train[0].shape[1]}')
#print(f'- X Test 0: {x_test[0].shape[0]} x {x_test[0].shape[1]}')
#print(f'- Y Test 0: {y_test[0].shape[0]} x {y_test[0].shape[1]}')

def train_central(X, Y, x_test, y_test, T, config):
    # Function that organized all the processes, using the parameters imputed.
    # After <num> epochs, calculates the cost, the accuracy in train and test
    
    # Copy params
    Thetas = {}
    for key in T.keys():
        Thetas[key] = np.copy(T[key])

    m = X.shape[0]
    
    j_history = []
    sec1 = time.time()
    
    epochs = config[0]
    alpha = config[1]
    batch_size = config[2]
    
    if batch_size <= 0:
        b_size = m
        print(f'Using batch size: {b_size}..')
    elif isinstance(batch_size, int) and (1<= batch_size <= m):
        b_size = batch_size
    else:
        return 'ERROR IN BATCH_SIZE'

    for ep in range(epochs):
        a = np.array([0,b_size])
        num = 1 # Use a higher number if lot of epochs
        
        for i in range(m // b_size):
            inx = a + b_size*i
            grad_list = fl.Gradients(X[inx[0]:inx[1]], Y[inx[0]:inx[1]],Thetas,config)
            for g in range(len(grad_list)):
                Thetas[f'T{g+1}'] = Thetas[f'T{g+1}'] - alpha*np.array(grad_list[g]).T
        
        if (ep+1) % num == 0: #
            J = fl.CostFunction(X,Y,Thetas,config)
            j_history.append(J)
            accu_train = fl.accuracy(X,Y,Thetas,config)
            accu_test = fl.accuracy(x_test,y_test,Thetas,config)
            sec2 = time.time()
            time_spent = sec2 - sec1
            print(f'Epoch: {ep+1}; Cost: {J:.5f}: Accuracy Train: {accu_train:.5%}; Accuracy Test: {accu_test:.5%}; Time Spent: {time_spent:.2f} s')

    return Thetas

def train_node(T, config, subset_idx):
    # Function that organized all the processes, using the parameters imputed.
    # After <num> epochs, calculates the cost, the accuracy in train and test
    
    X = np.load('uneven_np_datasets/quad/quad_cluster_x_train'+str(subset_idx)+'.npy')
    m = X.shape[0]
    n = X.shape[1] + 1
    X = np.concatenate((np.ones([m,1]),X), axis=1) 

    Y = np.load('uneven_np_datasets/quad/quad_cluster_y_train'+str(subset_idx)+'.npy')
    cat = np.zeros([m,10])
    for ind, num in enumerate(Y):
        cat[ind][num] = 1
    Y = cat    

    # Copy params
    Thetas = {}
    for key in T.keys():
        Thetas[key] = np.copy(T[key])

    m = X.shape[0]
    
    j_history = []
    sec1 = time.time()
    
    epochs = config[0]
    alpha = config[1]
    batch_size = config[2]
    
    if batch_size <= 0:
        b_size = m
        print(f'Using batch size: {b_size}..')
    elif isinstance(batch_size, int) and (1<= batch_size <= m):
        b_size = batch_size
    else:
        return 'ERROR IN BATCH_SIZE'

    for ep in range(epochs):
        a = np.array([0,b_size])
        num = 10 # Use a higher number if lot of epochs
        
        for i in range(m // b_size):
            inx = a + b_size*i
            grad_list = fl.Gradients(X[inx[0]:inx[1]], Y[inx[0]:inx[1]],Thetas,config)
            for g in range(len(grad_list)):
                Thetas[f'T{g+1}'] = Thetas[f'T{g+1}'] - alpha*np.array(grad_list[g]).T
        
        if (ep+1) % num == 0: #
            J = fl.CostFunction(X,Y,Thetas,config)
            j_history.append(J)
            accu_train = fl.accuracy(X,Y,Thetas,config)
            #accu_test = fl.accuracy(x_test,y_test,Thetas,config)
            sec2 = time.time()
            time_spent = sec2 - sec1
            print(f'Node: {subset_idx+1}; Epoch: {ep+1}; Cost: {J:.5f}: Accuracy Train: {accu_train:.5%}; Time Spent: {time_spent:.2f} s')

    return Thetas

# Define NN architecture
structure = [784, 'relu', 100, 'softmax', 10]
nn_sizes = [x for x in structure if isinstance(x, int)]
activations = [x.lower() for x in structure if isinstance(x, str)]

epochs = 10
alpha = 0.1
Reg = 0.0
batch_size = 100

nn_config = [epochs, alpha, batch_size, Reg, nn_sizes, activations]

# Initialize weights
thetas = {}
for layer in range(len(nn_sizes)-1):
    thetas[f'T{layer+1}'] = np.random.randn(nn_sizes[layer]+1, nn_sizes[layer+1])/10

# Train model centrally with merged dataset
print('---------------------------------------------------------')
print('Train model centrally with merged dataset')
print('---------------------------------------------------------')
#trained_thetas = train_central(x_train_merged, y_train_merged, 
#                               x_test_merged, y_test_merged, 
#                               thetas, nn_config)

print('---------------------------------------------------------')
print('Train decentralized model in federated manner')
print('---------------------------------------------------------')
# Re-initialize weights for decentralized training
thetas = {}
for layer in range(len(nn_sizes)-1):
    thetas[f'T{layer+1}'] = np.random.randn(nn_sizes[layer]+1, nn_sizes[layer+1])/10

thetas_main = {}

#num_of_federated_rounds = 25
accu_test = 0.1
f = 0
#for f in range(num_of_federated_rounds):
while(accu_test < 0.96):
    print("Federated Learning: Round " + str(f+1))

    start = time.time()

    for i in range(num_of_nodes):
        thetas_main[i] = train_node(thetas, nn_config, i)

    end = time.time()
    time_spent = end - start

    # Update main model weights
    thetas_temp_T1 = thetas_main[0]['T1']
    thetas_temp_T2 = thetas_main[0]['T2']

    for i in range(1,num_of_nodes):
        thetas_temp_T1 += thetas_main[i]['T1']
        thetas_temp_T2 += thetas_main[i]['T2']

    thetas['T1'] = thetas_temp_T1/num_of_nodes
    thetas['T2'] = thetas_temp_T2/num_of_nodes
    
    accu_test = fl.accuracy(x_test_merged,y_test_merged,thetas,nn_config)
    print(f'Federated Round: {f+1}; Test Accuracy: {accu_test:.5%}; Time Spent: {time_spent:.2f} s')
    f+=1
