import numpy as np

class MachineLearningModel():
    def __init__(self, func, cost_function):
        self.func = func # linear, sigmoid, etc.
        self.cost_function = cost_function # mse, binary-crossentropy, etc.
        self.weights = None
        self.bias = None
    def train(self, X, y, lr, epochs):
        if self.weights is None:
            # each column in weights matrix represents a neuron
            # each row matches a feature of X
            # num of rows must match num of columns in X
            # (each column in X represents a feature)
            num_of_features = X.shape[1]
            # as it's a ML model, only one neuron will be created
            num_of_neurons = 1
            self.weights = np.zeros(
                (num_of_features, num_of_neurons), dtype='float64',
            )
        if self.bias is None:
            # each element represents a bias of a neuron
            num_of_neurons = 1
            self.bias = np.zeros(num_of_neurons, dtype='float64')

        self.weights, self.bias = gradient_descent(
            X, y, self.weights, self.bias, self.func, lr, epochs
        )
    def predict(self, X):
        return predict(X, self.weights, self.func, bias_vector=self.bias)
    def cost(self, X, y):
        if self.cost_function == 'mse':
            cost = mean_squared_error(
                X, y, self.weights, self.bias, self.func
            )
        return cost

def mean_squared_error(X, y, weights, bias_vector, func):
    """
    Mean Squared Error cost function for layers.

    Args:
        X (np.array): Input matrix.
            Each row represents an input/example.
            Each column represents a feature.
        y (np.array): Targets vector.
            Each element represents a target.
        weights (np.array): Weights matrix.
            Each column represents a neuron.
        bias_vector (np.array): Bias vector.
            Each element represents a bias.
        func (str): Activation function.
            'linear', 'polynomial', 'sigmoid'.

    Returns:
        float: Error.
    """    
    m = len(y)
    predictions = predict(X, weights, func, bias_vector=bias_vector)
    error = (1/m) * np.sum(np.power(predictions - y, 2))
    return error

def gradient_descent(X, y, layer_weights, bias_vector, func, lr, epochs):
    m = len(y)
    rows, columns = layer_weights.shape[:2]
    X_rows = X.shape[0]

    layer_weights = layer_weights.copy()
    X = X.copy()

    # create bias weights for each neuron (merge bias vector into weights matrix)
    layer_weights = np.insert(layer_weights, 0, bias_vector, 0)
    # create a row full of ones (which will be used by the bias weights)
    X = np.insert(X, 0, np.ones(X_rows), 1)
    X_transposed = X.T

    for iteration in range(epochs):
        # vectorized form of gradient descent        
        layer_weights -= lr * (1/m) * X_transposed.dot(predict(X, layer_weights, func) - y)

    # separate bias vector and layer weights
    bias_vector = layer_weights[0,:]
    layer_weights = np.delete(layer_weights, 0, 0)

    return layer_weights, bias_vector

def predict(X, layer_weights, func, bias_vector=None):
    """
    Args:
        X (np.array): Input matrix.
            Each row represents an input/example.
            Each column represents a feature.
        layer_weights (np.array): Layer weights matrix.
            Each column represents a neuron.
        func (str): Activation function.
            'linear', 'polynomial', 'sigmoid'.
        bias_vector (np.array, optional): Bias vector.
            Each element represents a bias.
            If layer_weights has the bias vector as a row (like in Gradient Descent), 
            bias_vector has to be None.

    Returns:
        np.ndarray: Prediction.
    """
    # logits
    if func in ['linear', 'polynomial', 'sigmoid']:
        z = np.dot(X, layer_weights)
        if bias_vector:
            z += bias_vector

    # prediction
    if func in ['linear', 'polynomial']:
        prediction = z
    elif func in ['sigmoid']:
        prediction = 1/(1 + np.e**(-z))

    return prediction

if __name__ == '__main__':
    print('\n\n-------- TEST 1.1 Gradient Descent without Machine Learning Model --------\n')

    # 1000 USD per square meter # feature 1
    # 5000 USD per bedroom # feature 2
    # 40000 USD bias # bias

    # each row represents an example/input
    # each column represents a feature
    X_train = np.array([
        [20, 4], # 2 features
        [50, 5],
        [30, 3],
        [40, 6],
        [10, 1],
        [15, 2],
        [10, 3],
        [30, 5],
        [70, 2],
        [90, 1],
        [85, 3],
        [90, 4],
        [100, 3],
        [60, 4],
    ], dtype='float64')

    target_layer_weights_without_normalization = np.array([
        [1000, 5000], # neuron 1 (1 neuron linear model)
    ], dtype='float64').T # each column represents a neuron

    target_bias_without_normalization = np.array([40000], dtype='float64')

    print(f'X train (without normalization):\n{X_train}')
    print(f'Target weights (without normalization):\n{target_layer_weights_without_normalization}')
    print(f'Target bias (without normalization):\n{target_bias_without_normalization}')

    y_train = predict(
        X_train, target_layer_weights_without_normalization, 
        'linear', bias_vector=target_bias_without_normalization,
    )
    print(f'y train (without normalization):\n{y_train}')

    # Normalization
    # 1st feature
    first_feature_range = np.max(X_train[:, 0]) - np.min(X_train[:, 0])
    X_train[:, 0] /= first_feature_range

    # 2nd feature
    second_feature_range = np.max(X_train[:, 1]) - np.min(X_train[:, 1])
    X_train[:, 1] /= second_feature_range

    # targets
    y_range = np.max(y_train) - np.min(y_train)
    y_train /= y_range

    # Initialize weights and bias
    layer_weights = np.array([
        [0, 0],
    ], dtype='float64').T

    bias_vector = np.array([0], dtype='float64')

    # Gradient Descent
    fit_weights, fit_bias = gradient_descent(
        X_train, y_train, layer_weights, bias_vector, 'linear', 0.001, 10000,
    )
    print(f'Weights after Gradient Descent:\n{fit_weights}\nBias after Gradient Descent:\n{fit_bias}')

    error = mean_squared_error(X_train, y_train, fit_weights, fit_bias, 'linear')
    print(f'Mean Squared Error: {error}')

    # Validation
    X_validation = np.array([
        [10, 8],
        [200, 70],
        [30, 8],
        [60, 3],
        [120, 4],
    ], dtype='float64')

    y_validation = predict(
        X_validation, target_layer_weights_without_normalization, 
        'linear', bias_vector=target_bias_without_normalization,
    )

    # Normalization
    X_validation[:, 0] /= first_feature_range
    X_validation[:, 1] /= second_feature_range

    y_validation /= y_range

    prediction = predict(X_validation, fit_weights, 'linear', bias_vector=fit_bias)

    # remove normalization
    prediction *= y_range
    y_validation_without_normalization = y_validation * y_range

    print(f"Expected prediction:\n{y_validation_without_normalization}")
    print(f"Predicted:\n{prediction}")


    print('\n\n-------- TEST 1.2 Machine Learning Model --------\n')

    model = MachineLearningModel('linear', 'mse')

    model.train(X_train, y_train, 0.001, 10000)
    print(f'Weights after training:\n{model.weights}\nBias after training:\n{model.bias}')

    error = model.cost(X_train, y_train)
    print(f'Mean Squared Error (train): {error}')

    # Validation
    prediction = model.predict(X_validation)

    # remove normalization
    prediction *= y_range

    print(f"Expected prediction:\n{y_validation_without_normalization}")
    print(f"Predicted:\n{prediction}")

    error = model.cost(X_validation, y_validation)
    print(f'Mean Squared Error (validation): {error}')
