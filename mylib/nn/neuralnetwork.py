#Import
import numpy as np

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        self.W = []
        self.layers = layers
        self.alpha = alpha
        for i in np.arange(0, len(layers)-2): #cycling on all layers except the last two ones
            w = np.random.randn(layers[i]+1, layers[i+1]+1) #+1 is for bias trick but we didnt want it in the output layer
            self.W.append(w/np.sqrt(layers[i]))
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    #Return a string that rapresents the network
    def __repr__(self):
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoid_deriv(self, x): #x already passed into sigmoid funcion
        return x*(1-x)

    def fit(self, X, y, epochs=1000, displayUpdate=100):
        X = np.c_[X, np.ones(X.shape[0])]
        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X,y):
                self.fit_partial(x, target)
            if epoch==0 or (epoch+1)%displayUpdate==0:
                loss = self.calculate_loss(X,y)
                print("[INFO] epoch={}, loss={:.10f}".format(epoch+1, loss))

    def fit_partial(self, x, y):
        A = [np.atleast_2d(x)] #list of output activations
        #I) Feedforeward
        for layer in np.arange(0, len(self.W)):
            net = A[layer].dot(self.W[layer])
            out = self.sigmoid(net)
            A.append(out)
        #II) Backpropagation
        error = A[-1]-y
        D = [error*self.sigmoid_deriv(A[-1])] #list of deltas for chain rule
        for layer in np.arange(len(A)-2, 0, -1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)
        #reverse the delta's order NON FUNZIONA QUESTO COMANDO
        D[::-1]


        #III) Weight Update
        for layer in np.arange(0, len(self.W)):
            self.W[layer] += -self.alpha * A[layer].T.dot(D[len(A)-2-layer])

    def predict(self, X, addBias=True):
        p = np.atleast_2d(X)
        if addBias==True:
            p = np.c_[p, np.ones(p.shape[0])]
        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))
        return p

    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5*np.sum((predictions-targets)**2)
        return loss

