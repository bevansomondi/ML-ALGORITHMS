import numpy as np

class LogisticRegression:
    def __init__(self, n_iters, learning_rate):
        self.n_iters = n_iters
        self.lr = learning_rate
        self.weights = None
        self.loss_history = []
        self.sigmoid = lambda z: 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        #self.weights = np.random.randn(X.shape[1] + 1).reshape(-1, 1)
        self.weights = np.zeros(X.shape[1] + 1).reshape(-1,1)
        

        #append ones to the feature vector to take care of the bias 
        X = np.append(X,np.ones((X.shape[0], 1)), axis=1)
        #make sure to change y to a 2D array 
        y = np.array(y, ndmin=2)

        self.update_weights(X, y)
        """
        for epoch in range(self.n_iters):
            self.update_weights(X, y)
            self.loss_history.append(self.cost(X,y))
            print(f"Epoch: {epoch + 1 }/{self.n_iters} cost: {self.loss_history[epoch]}")
        """
        
        

    def update_weights(self, X, y):
        diff = np.inf

        while diff > .01:
            pred = self.sigmoid(np.dot(X, self.weights))
            error = pred - y.T
            #print(y)
            #print(pred)

            dl_dw = (1 / float(len(X))) * np.dot(X.T, error) #don't sum here because the dot product already takes care of that
            #update the weights 
            diff = np.abs(dl_dw).sum()
            print(diff)
            self.weights -= self.lr * dl_dw 
            self.loss_history.append(self.cost(X,y))
    
        

    def cost(self, X, y):
        epsilon = 1e-9
        predicted_prob = self.sigmoid(np.dot(X, self.weights))
        
        cost0 = y.T * np.log(predicted_prob + epsilon)
        cost1 = (1-y).T * np.log(1-predicted_prob + epsilon)

        return -np.mean(cost0 + cost1)
    
    def predict(self, X, threshold = .5):
        X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
        y_hat = self.sigmoid(np.dot(X, self.weights))
        predicted_cls = [1 if y > threshold else 0 for y in y_hat]
        
        return np.array(predicted_cls), y_hat
    
    def evaluate(self, y, yhat):
        tp, tn, fp, fn = 0.0, 0.0, 0.0, 0.0
        for i in range(len(y)):
            if y[i] == 1 and yhat[i] == 1:
                tp += 1

            elif y[i] == 1 and yhat[i] == 0:
                fn += 1

            elif y[i] == 0 and yhat[i] == 0:
                tn += 1

            elif y[i] == 0 and yhat[i] == 1:
                fp += 1

            precision = tp / (tp + fp)
            recal = tp / (tp + fn)
            f1_score = 2 * precision * recal / (precision + recal)

            return precision, recal, f1_score



        
        
        
       

        
       

        
