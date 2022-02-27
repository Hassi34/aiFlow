import numpy as np 

class Perceptron:
    def __init__(self, eta, epochs):
        self.weights = np.random.randn(3) * 1e-4
        print(f"Initial Weights before training \n{self.weights}") #small weight init
        self.eta = eta #learning rate
        self.epochs = epochs
    def activation_fuction(self, inputs, weights): #step function
        z = np.dot(inputs , weights) #z = W * X
        return np.where(z > 0 , 1 , 0 )
    def fit(self, X, y):
        self.X = X
        self.y = y 
        X_with_bias = np.c_[self.X, -np.ones((len(self.X), 1))] #concate bias with x
        print(f"X with bias: \n{X_with_bias}")
        for epoch in range(self.epochs):
            print("--"*10)
            print(f"For epoch {epoch}")
            print("--"*10)
            
            y_hat = self.activation_fuction(X_with_bias, self.weights) #Forward Propagation
            print(f"Predicted Value after forward pass: \n{y_hat}")
            self.error = self.y - y_hat
            print(f"Error: \n{self.error}")
            self.weights = self.weights + self.eta * (np.dot(X_with_bias.T, self.error)) #Backward Propagation
            print(f"Updated weights after epoch : \n{epoch} / {self.epochs} : \n{self.weights}")
            print("#"*100)
    def predict(self, X):
        X_with_bias  = np.c_[X, -np.ones((len(X), 1))]
        return self.activation_fuction(X_with_bias, self.weights)
    def total_loss(self):
        total_loss = np.sum(self.error)
        print(f"total loss : {total_loss}")
        return total_loss