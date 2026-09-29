import numpy as np
from ucimlrepo import fetch_ucirepo


class LinearRegression:
    def __init__(self, X, y):
        self.X = X                # Design matrix (feature matrix)
        self.y = y                # Ground truth target values
        self.n = y.shape[0]       # Number of training samples
        self.d = X.shape[1]       # Number of features (dimension of weight vector w)


    # Compute MSE (Mean Squared Error) loss
    def compute_loss(self, w, b):
        y_pred = self.X @ w + b   # Calculate predicted values: y_hat = Xw + b
        mse = np.mean(np.square(y_pred - self.y)) # Compute mean squared error
        return mse                # Return MSE loss value

    # Compute gradients of loss with respect to weight w and bias b
    def compute_gradient(self, w, b):
        y_pred = self.X @ w + b   # Calculate predicted values
        dw = (2.0 / self.n) * self.X.T @ (y_pred - self.y) # Gradient for weight vector w
        db = (2.0 / self.n) * np.sum(y_pred - self.y)      # Gradient for bias term b
        return dw, db             # Return gradients dw and db


# Define Gradient Descent optimizer class
class GradientDescent:
    def __init__(self, A=0.005, epochs=15000):
        self.A = A                # A represents learning rate for parameter update
        self.epochs = epochs      # Total number of training iterations

    # lf: abbreviation of loss function
    def optimize(self, lf):
        w = np.zeros(lf.d)        # Initialize weight vector w to all zeros
        b = 0.0                   

        # Iterate for fixed epochs
        for epoch in range(self.epochs):
            dw, db = lf.compute_gradient(w, b) # Get gradients from loss function object(lf)
            # Update parameters using gradient descent rule
            w = w - self.A * dw
            b = b - self.A * db

            # Print loss information every 200 epochs
            if epoch % 200 == 0:
                current_loss = lf.compute_loss(w, b) # Calculate current MSE loss from loss function
                print(f"Epoch {epoch:5d} | Loss = {current_loss:.4f}")
        return w, b                


if __name__ == "__main__":
    # Load UCI Wine Quality dataset (red wine)
    #   Original dataset: https://archive.ics.uci.edu/dataset/186/wine+quality
    wine = fetch_ucirepo(id=186)
    X = wine.data.features.values  # feature matrix
    y = wine.data.targets.values.ravel() # target values
    print("Dataset name:", wine.metadata.name)
    print("Number of samples:", X.shape[0])
    print("Number of features:", X.shape[1])
    # Standardize features (z-score)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)


   
    np.random.seed(42)                 # Set random seed for reproducible result
    total_samples = X.shape[0]         # Get total number of data samples
    shuffled_index = np.random.permutation(total_samples) # Generate shuffled index array
    train_ratio = 0.8                  # Define 80% data for training
    train_size = int(total_samples * train_ratio) # Calculate number of training samples
    train_idx = shuffled_index[:train_size] # Get indices for training set
    test_idx = shuffled_index[train_size:]  

    # Combine original four lines into two lines for concise view
    X_train, y_train = X[train_idx, :], y_raw[train_idx] # Extract training features and labels
    X_test, y_test = X[test_idx, :], y_raw[test_idx]     # Extract test features and labels

    lf = LinearRegression(X_train, y_train) # Create loss function(lf) object with training data
    optimizer = GradientDescent(A=0.01, epochs=8000) # Create gradient descent optimizer, A is learning rate
    w_opt, b_opt = optimizer.optimize(lf)   

    print("=== Training Finished ===")
    print("Optimal weights w:", w_opt)         # Print trained weight vector
    print("Optimal bias b:", b_opt)           # Print trained bias term

    train_loss = lf.compute_loss(w_opt, b_opt) # Evaluate loss on training set
    print("Training set final loss:", train_loss)

    test_model = LinearRegression(X_test, y_test) # Create loss function object for test dataset
    test_loss = test_model.compute_loss(w_opt, b_opt) # Evaluate loss on test set
    print("Test set final loss:", test_loss)
