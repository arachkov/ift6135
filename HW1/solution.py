import numpy as np
import torch
import torchvision


def one_hot(y, n_classes=10):
    return np.eye(n_classes)[y]


def load_cifar10(root, flatten=False):
    """
    Usage example:
    > train_data, valid_data, test_data = load_cifar10("/data", flatten=True)
    > train_x, train_y = train_data
    where both train_x and train_y are numpy arrays
    train_x.shape == (40000, 3072) or train_x.shape == (40000, 3, 32, 32)
    train_y.shape == (40000, 10), one-hot format
    :param root: path where the cifar10 dataset will be downloaded, e.g. "/tmp/data/"
    :param flatten: When True, dataset is reshaped to (num_examples, 3072), otherwise shape is (num_examples, 3, 32, 32)
    :return: train, valid and test set in numpy arrays
    """
    transform = torchvision.transforms.ToTensor()

    train_dataset = torchvision.datasets.CIFAR10(root, train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root, train=False, transform=transform, download=True)

    # randomly split train into train/valid
    perm = np.random.RandomState(seed=1).permutation(
        range(len(train_dataset)))  # fix seed to have same split every time.
    x = [train_dataset[i][0] for i in perm]  # train_dataset.data[perm]
    y = [one_hot(train_dataset[i][1]) for i in perm]
    train_x, train_y = x[:40000], y[:40000]
    valid_x, valid_y = x[40000:], y[40000:]
    test_x = [test_dataset[i][0] for i in range(len(test_dataset))]
    test_y = [one_hot(test_dataset[i][1]) for i in range(len(test_dataset))]

    # convert to numpy arrays after stacking
    train_x = torch.stack(train_x).cpu().numpy()
    train_y = np.stack(train_y)
    valid_x = torch.stack(valid_x).cpu().numpy()
    valid_y = np.stack(valid_y)
    test_x = torch.stack(test_x).cpu().numpy()
    test_y = np.stack(test_y)

    if flatten:
        train_x = train_x.reshape(-1, 32 * 32 * 3)
        valid_x = valid_x.reshape(-1, 32 * 32 * 3)
        test_x = test_x.reshape(-1, 32 * 32 * 3)

    # Package everything
    train_data = train_x, train_y
    valid_data = valid_x, valid_y
    test_data = test_x, test_y

    return train_data, valid_data, test_data


class NN(object):
    def __init__(self,
                 hidden_dims=(784, 256),
                 epsilon=1e-6,
                 lr=0.01,
                 batch_size=64,
                 seed=1,
                 activation="relu",
                 data=None
                 ):

        self.hidden_dims = hidden_dims
        self.n_hidden = len(hidden_dims)
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.activation_str = activation
        self.epsilon = epsilon

        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': []}

        if data is None:
            # for testing, do NOT remove or modify
            self.train, self.valid, self.test = (
                (np.random.rand(400, 3072), one_hot(np.random.randint(0, 10, 400))),
                (np.random.rand(400, 3072), one_hot(np.random.randint(0, 10, 400))),
                (np.random.rand(400, 3072), one_hot(np.random.randint(0, 10, 400)))
        )
        else:
            self.train, self.valid, self.test = data


    def initialize_weights(self, dims):        
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = {}
        # self.weights is a dictionnary with keys W1, b1, W2, b2, ..., Wm, Bm where m - 1 is the number of hidden layers
        all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]
        for layer_n in range(1, self.n_hidden + 2):
            init_const = np.sqrt(6/(all_dims[layer_n-1]+all_dims[layer_n]))
            self.weights[f"W{layer_n}"] = np.random.uniform(low=-init_const, high=init_const, size=(all_dims[layer_n-1], all_dims[layer_n]))
            self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))

    def relu(self, x, grad=False):
        if grad:
            res = 1.0 * (x > 0)
        else:
            res = x * (x > 0)
        return res

    def sigmoid(self, x, grad=False):
        sigm_f = 1/(1+np.exp(-x))
        if grad:
            res = sigm_f*(1-sigm_f)
        else:
            res = sigm_f
        return res

    def tanh(self, x, grad=False):
        tanh_f = ((np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)))
        if grad:
            res= 1-tanh_f**2
        else:
            res=tanh_f
        return res

    def activation(self, x, grad=False):
        if self.activation_str == "relu":
            res = self.relu(x, grad)
        elif self.activation_str == "sigmoid":
            res = self.sigmoid(x, grad)
        elif self.activation_str == "tanh":
            res = self.tanh(x, grad)
        elif self.activation_str == "leakyrelu":
            res = self.leakyrelu(x, grad)
        else:
            raise Exception("invalid") 
        return res

    def softmax(self, x):
        # Remember that softmax(x-C) = softmax(x) when C is a constant.
        if len(np.shape(x))>1:
            batch_max = np.max(x, axis=1, keepdims=True)  
            res = np.exp(x-batch_max)/ np.sum(np.exp(x-batch_max), axis=1, keepdims=True)
            return res
        else:
            batch_max = np.max(x, axis=0, keepdims=True)  
            res = np.exp(x-batch_max)/ np.sum(np.exp(x-batch_max), axis=0, keepdims=True)
            return res

    def forward(self, x):
        cache = {"Z0": x}

        # cache is a dictionnary with keys Z0, A0, ..., Zm, Am where m - 1 is the number of hidden layers
        # Ai corresponds to the preactivation at layer i, Zi corresponds to the activation at layer i
        for layer_n in range(1, self.n_hidden + 2):
            cache[f"A{layer_n}"]=np.matmul(cache[f"Z{layer_n-1}"],self.weights[f"W{layer_n}"])+self.weights[f"b{layer_n}"]

            if layer_n<self.n_hidden+1:
                cache[f"Z{layer_n}"]=self.activation(cache[f"A{layer_n}"])
            else:
                cache[f"Z{layer_n}"]=self.softmax(cache[f"A{layer_n}"])

        return cache

    def backward(self, cache, labels):
        output = cache[f"Z{self.n_hidden + 1}"]
        grads = {}
        grads[f'dA{self.n_hidden+1}']= -(labels-output)

        for layer_k in range(self.n_hidden+1,0,-1):
            
            # calculate gradient per example in minibatch and take average of the minibatch

            grads[f"dW{layer_k}"] = np.matmul(np.transpose(cache[f"Z{layer_k-1}"]),grads[f"dA{layer_k}"])/output.shape[0] 

            grads[f'db{layer_k}']=np.mean(grads[f"dA{layer_k}"],axis=0,keepdims=True)

            if layer_k>1:
                grads[f'dZ{layer_k-1}']=np.matmul(grads[f"dA{layer_k}"],np.transpose(self.weights[f"W{layer_k}"]))
                grads[f'dA{layer_k-1}']= np.multiply(grads[f"dZ{layer_k-1}"],self.activation(cache[f"A{layer_k-1}"],grad=True))

        return grads

    def update(self, grads):

        for layer in range(1, self.n_hidden + 2):
            self.weights[f"W{layer}"] -= self.lr*grads[f"dW{layer}"]
            self.weights[f"b{layer}"] -= self.lr*grads[f"db{layer}"]

    # def one_hot(self, y, n_classes=None):
    #     n_classes = n_classes or self.n_classes
    #     return np.eye(n_classes)[y]

    def loss(self, prediction, labels):
        prediction[np.where(prediction < self.epsilon)] = self.epsilon
        prediction[np.where(prediction > 1 - self.epsilon)] = 1 - self.epsilon

        loss_matrix= -1*(labels*np.log(prediction))
        return loss_matrix.sum()/float(labels.shape[0])

    def compute_loss_and_accuracy(self, X, y):

        cache = self.forward(X)
        predictions = np.argmax(cache[f"Z{self.n_hidden + 1}"], axis=1)
        accuracy = np.mean(y == predictions)
        loss = self.loss(cache[f"Z{self.n_hidden + 1}"], y)
        return loss, accuracy, predictions

    def train_loop(self, n_epochs):
        X_train, y_train = self.train
        y_onehot = y_train
        dims = [X_train.shape[1], y_onehot.shape[1]]
        self.initialize_weights(dims)

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_onehot[self.batch_size * batch:self.batch_size * (batch + 1), :]

                cache = self.forward(minibatchX)
                grads = self.backward(cache, minibatchY)

                self.update(grads)                

            X_train, y_train = self.train
            train_loss, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train)            
            X_valid, y_valid = self.valid
            valid_loss, valid_accuracy, _ = self.compute_loss_and_accuracy(X_valid, y_valid)

            self.train_logs['train_accuracy'].append(train_accuracy)
            self.train_logs['validation_accuracy'].append(valid_accuracy)
            self.train_logs['train_loss'].append(train_loss)
            self.train_logs['validation_loss'].append(valid_loss)

        return self.train_logs

    def evaluate(self):
        X_test, y_test = self.test
        test_loss, test_accuracy, _ = self.compute_loss_and_accuracy(X_test, y_test)
        return test_loss, test_accuracy
