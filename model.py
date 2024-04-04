import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = None
        self.mean = None
        self.var = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Initialize arrays to store mean and variance for each feature and class
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.class_probs = np.zeros(n_classes)

        # Compute mean and variance for each class and feature
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.class_probs[idx] = len(X_c) / float(n_samples)

    def predict(self, X):
        preds = [self._predict(x) for x in X]
        return np.array(preds)

    def _predict(self, x):
        posteriors = []

        # Calculate posterior probability for each class
        for idx, c in enumerate(self.classes):
            prior = np.log(self.class_probs[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        # Return the class with the highest posterior probability
        return self.classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)

    # Shuffle the data
    idx = np.random.permutation(len(X))
    X_shuffle = X[idx]
    y_shuffle = y[idx]

    # Split the data into train and test sets
    split_index = int(len(X) * (1 - test_size))
    X_train = X_shuffle[:split_index]
    y_train = y_shuffle[:split_index]
    X_test = X_shuffle[split_index:]
    y_test = y_shuffle[split_index:]

    return X_train, X_test, y_train, y_test