// Code Snippets
const svmFromScratch = `import numpy as np
import cvxopt
from cvxopt import matrix, solvers

class SVMQP:
    def __init__(self, epsilon=1e-6, C=1.0, kernel='rbf', gamma=0.1):
        self.epsilon = epsilon
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.support_vectors = None
        self.support_vector_labels = None
        self.alphas = None
        self.b = 0
        
    def rbf_kernel(self, x1, x2):
        return np.exp(-self.gamma * np.sum((x1 - x2) ** 2))
    
    def compute_kernel_matrix(self, X):
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                if self.kernel == 'rbf':
                    K[i, j] = self.rbf_kernel(X[i], X[j])
                else:
                    K[i, j] = np.dot(X[i], X[j])  # Linear kernel
        return K
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Convert labels to +1/-1
        y_binary = np.where(y > 0, 1, -1)
        
        # Compute the kernel matrix
        K = self.compute_kernel_matrix(X)
        
        # Set up the QP problem
        P = matrix(np.outer(y_binary, y_binary) * K)
        q = matrix(-np.ones(n_samples))
        
        # Inequality constraints: 0 <= alpha <= C
        G = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
        h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))
        
        # Equality constraint: sum(alpha_i * y_i) = 0
        A = matrix(y_binary.astype(float)).trans()
        b = matrix(0.0)
        
        # Solve the QP problem
        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)
        
        # Extract alphas from the solution
        alphas = np.array(solution['x']).flatten()
        
        # Find support vectors (alphas > epsilon)
        sv_indices = np.where(alphas > self.epsilon)[0]
        self.alphas = alphas[sv_indices]
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y_binary[sv_indices]
        
        # Calculate bias term b
        self.b = 0
        for i in range(len(sv_indices)):
            self.b += self.support_vector_labels[i]
            self.b -= np.sum(self.alphas * self.support_vector_labels * K[sv_indices[i], sv_indices])
        self.b /= len(sv_indices)
        
    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            s = 0
            for alpha, sv_y, sv in zip(self.alphas, self.support_vector_labels, self.support_vectors):
                if self.kernel == 'rbf':
                    s += alpha * sv_y * self.rbf_kernel(X[i], sv)
                else:
                    s += alpha * sv_y * np.dot(X[i], sv)
            y_pred[i] = s
        return np.sign(y_pred + self.b)`;

const logisticRegressionFromScratch = `import numpy as np
from scipy.optimize import minimize

class MyLogisticRegression:
    def __init__(self, lr=0.01, max_iter=1000, tol=1e-4, C=1.0):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.C = C  # Regularization parameter
        self.theta = None
        self.classes = None
        
    def softmax(self, z):
        # Avoid numerical overflow by subtracting the maximum
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
    def one_hot_encode(self, y):
        self.classes = np.unique(y)
        n_samples = len(y)
        n_classes = len(self.classes)
        y_encoded = np.zeros((n_samples, n_classes))
        for i, class_label in enumerate(self.classes):
            y_encoded[:, i] = (y == class_label).astype(int)
        return y_encoded
        
    def compute_loss(self, theta, X, y_encoded, C):
        theta_reshaped = theta.reshape(X.shape[1], y_encoded.shape[1])
        z = X @ theta_reshaped
        y_pred = self.softmax(z)
        
        # Cross-entropy loss
        loss = -np.sum(y_encoded * np.log(y_pred + 1e-10)) / len(y_encoded)
        
        # L2 regularization (excluding bias)
        # Assuming the first column of X is the bias term
        reg = 0.5 * C * np.sum(theta_reshaped[1:, :] ** 2)
        
        return loss + reg
        
    def compute_gradient(self, theta, X, y_encoded, C):
        theta_reshaped = theta.reshape(X.shape[1], y_encoded.shape[1])
        z = X @ theta_reshaped
        y_pred = self.softmax(z)
        
        # Gradient of cross-entropy loss
        grad = X.T @ (y_pred - y_encoded) / len(y_encoded)
        
        # Gradient of L2 regularization (excluding bias)
        reg_grad = np.zeros_like(theta_reshaped)
        reg_grad[1:, :] = C * theta_reshaped[1:, :]
        
        return (grad + reg_grad).flatten()
        
    def fit(self, X, y):
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        y_encoded = self.one_hot_encode(y)
        
        n_features = X_with_bias.shape[1]
        n_classes = y_encoded.shape[1]
        
        # Initialize parameters
        initial_theta = np.zeros(n_features * n_classes)
        
        # Optimize using BFGS
        result = minimize(
            fun=self.compute_loss,
            x0=initial_theta,
            args=(X_with_bias, y_encoded, self.C),
            method='BFGS',
            jac=self.compute_gradient,
            options={'gtol': self.tol, 'maxiter': self.max_iter}
        )
        
        self.theta = result.x.reshape(n_features, n_classes)
        
    def predict_proba(self, X):
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        z = X_with_bias @ self.theta
        return self.softmax(z)
        
    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes[np.argmax(proba, axis=1)]`;

const svmSklearn = `from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Preprocess data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define model with optimal hyperparameters 
model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    class_weight='balanced',
    decision_function_shape='ovr',
    random_state=42,
    probability=False
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score, f1_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')
weighted_f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1-Score: {macro_f1:.4f}")
print(f"Weighted F1-Score: {weighted_f1:.4f}")
print(classification_report(y_test, y_pred))`;

const logisticRegressionSklearn = `from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Preprocess data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define model with optimal hyperparameters
model = LogisticRegression(
    multi_class='multinomial',
    solver='saga',
    C=0.01,
    penalty='l1',
    max_iter=1000,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score, f1_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')
weighted_f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1-Score: {macro_f1:.4f}")
print(f"Weighted F1-Score: {weighted_f1:.4f}")
print(classification_report(y_test, y_pred))`;

const xgboostClassifier = `import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Preprocess data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to DMatrix format for better performance
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Define parameters
params = {
    'objective': 'multi:softmax',  # Multi-class classification
    'num_class': 5,  # 5 classes
    'eval_metric': 'mlogloss',
    'tree_method': 'hist',  # Enable GPU acceleration
    # 'predictor': 'gpu_predictor',
    'learning_rate': 0.05,
    'max_depth': 7,
    # 'n_estimators': 1000,
    "device": 'cuda',
    "max_bin": 512,
    "num_parallel_tree": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "nthread":4,
    "seed": 42
}

# Train the model
num_round = 1000
model = xgb.train(
    params,
    dtrain,
    num_round,
    verbose_eval=100
)

# Make predictions
y_pred = model.predict(dtest)

# Evaluate the model
from sklearn.metrics import accuracy_score, f1_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')
weighted_f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1-Score: {macro_f1:.4f}")
print(f"Weighted F1-Score: {weighted_f1:.4f}")
print(classification_report(y_test, y_pred))

# Visualize feature importance
import matplotlib.pyplot as plt

xgb.plot_importance(model, max_num_features=20)
plt.title('Top 20 Feature Importance in XGBoost Model')
plt.tight_layout()
plt.show()`;

const smoteCode = `from imblearn.over_sampling import SMOTE
from collections import Counter

# Before SMOTE
print('Original dataset shape:', Counter(y_train))

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# After SMOTE
print('Resampled dataset shape:', Counter(y_train_resampled))`;

const fftCode = `import numpy as np
import matplotlib.pyplot as plt

# Select a sample from class 0 (Normal heartbeat)
sample_idx = np.where(y_train == 0)[0][0]
ecg_signal = X_train[sample_idx]

# Compute FFT
fft_signal = np.abs(np.fft.fft(ecg_signal))
freq = np.fft.fftfreq(len(ecg_signal))

# Plot time domain vs frequency domain
plt.figure(figsize=(12, 6))

# Time domain plot
plt.subplot(1, 2, 1)
plt.plot(ecg_signal)
plt.title('ECG Signal (Time Domain)')
plt.xlabel('Sample point')
plt.ylabel('Amplitude')

# Frequency domain plot
plt.subplot(1, 2, 2)
plt.plot(freq[:len(freq)//2], fft_signal[:len(freq)//2])
plt.title('ECG Signal (Frequency Domain)')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.show()`;

const gradientBoostingFromScratch = `from tqdm.notebook import tqdm

class DecisionTree:
    """A simple regression decision tree used as a weak learner."""
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, g, h):
        """Fit the tree to gradients g using Hessians h for leaf values."""
        self.tree = self._build_tree(X, g, h, depth=0)

    def _build_tree(self, X, g, h, depth):
        """Recursively build the tree by finding the best splits."""
        if depth >= self.max_depth or len(X) <= 1:
            # Leaf node: value is sum of gradients / sum of Hessians
            value = np.sum(g) / np.sum(h) if np.sum(h) != 0 else 0
            return {'value': value}
        else:
            # Find the best split
            best_feature, best_threshold, best_gain = self._find_best_split(X, g)
            if best_gain <= 0:
                # No improvement: make a leaf
                value = np.sum(g) / np.sum(h) if np.sum(h) != 0 else 0
                return {'value': value}
            else:
                left_idx = X[:, best_feature] < best_threshold
                right_idx = ~left_idx
                left_tree = self._build_tree(X[left_idx], g[left_idx], h[left_idx], depth + 1)
                right_tree = self._build_tree(X[right_idx], g[right_idx], h[right_idx], depth + 1)
                return {'feature': best_feature, 'threshold': best_threshold,
                        'left': left_tree, 'right': right_tree}

    def _find_best_split(self, X, g):
        """Find the best split by maximizing variance reduction of gradients."""
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        n_features = X.shape[1]
        for feature in range(n_features):
            values = X[:, feature]
            thresholds = np.unique(values)
            for threshold in thresholds:
                left_idx = values < threshold
                right_idx = ~left_idx
                if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                    continue
                g_left = g[left_idx]
                g_right = g[right_idx]
                # Variance reduction as gain
                var_left = np.var(g_left) * len(g_left)
                var_right = np.var(g_right) * len(g_right)
                var_total = np.var(g) * len(g)
                gain = var_total - (var_left + var_right)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold, best_gain

    def predict(self, X):
        """Predict values for all samples in X."""
        return np.array([self._predict_sample(x, self.tree) for x in X])

    def _predict_sample(self, x, node):
        """Recursively predict for a single sample."""
        if 'value' in node:
            return node['value']
        else:
            if x[node['feature']] < node['threshold']:
                return self._predict_sample(x, node['left'])
            else:
                return self._predict_sample(x, node['right'])

class MyGradientBoosting:
    """Gradient boosting classifier for multi-class problems."""
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, n_classes=5):
        """
        Initialize the gradient boosting model.
        
        Parameters:
        - n_estimators: Number of boosting iterations
        - learning_rate: Step size for updates
        - max_depth: Maximum depth of decision trees
        - n_classes: Number of classes (fixed at 5)
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_classes = n_classes
        self.trees = [[] for _ in range(n_classes)]  # Trees for each class
        self.initial_scores = None

    def fit(self, X, y):
        """
        Fit the model to training data X and labels y.
        
        Parameters:
        - X: Array of shape (n_samples, 187) with continuous features
        - y: Array of shape (n_samples,) with labels in {0, 1, 2, 3, 4}
        """
        # Compute initial scores based on class frequencies
        class_freq = np.bincount(y, minlength=self.n_classes) / len(y)
        self.initial_scores = np.log(class_freq + 1e-9)  # Avoid log(0)
        F = np.repeat(self.initial_scores[np.newaxis, :], X.shape[0], axis=0)

        for _ in tqdm(range(self.n_estimators)):
            # Compute current probabilities
            p = self._softmax(F)
            # Convert labels to one-hot encoding
            y_onehot = np.eye(self.n_classes)[y]
            # Compute gradients and Hessians
            g = y_onehot - p  # Negative gradient of log-loss
            h = p * (1 - p)   # Diagonal approximation of Hessian
            # Fit one tree per class
            for k in range(self.n_classes):
                tree = DecisionTree(max_depth=self.max_depth)
                tree.fit(X, g[:, k], h[:, k])
                self.trees[k].append(tree)
                # Update scores with tree prediction
                F[:, k] += self.learning_rate * tree.predict(X)

    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        - X: Array of shape (n_samples, 187) with continuous features
        
        Returns:
        - Array of shape (n_samples,) with predicted labels in {0, 1, 2, 3, 4}
        """
        # Initialize scores with initial values
        scores = np.repeat(self.initial_scores[np.newaxis, :], X.shape[0], axis=0)
        # Add contributions from all trees
        for k in range(self.n_classes):
            for tree in self.trees[k]:
                scores[:, k] += self.learning_rate * tree.predict(X)
        # Compute probabilities and predict class
        p = self._softmax(scores)
        return np.argmax(p, axis=1)

    def _softmax(self, scores):
        """Compute softmax probabilities from scores."""
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
`

const randomForestFromScratch = `from tqdm.notebook import tqdm
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature        # Feature index to split on
        self.threshold = threshold    # Threshold value for the split
        self.left = left              # Left child node
        self.right = right            # Right child node
        self.value = value            # Predicted class (for leaf nodes)

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, max_features=None, n_classes=5):
        self.max_depth = max_depth              # Maximum depth of the tree
        self.min_samples_split = min_samples_split  # Minimum samples required to split
        self.max_features = max_features        # Number of features to consider at each split
        self.n_classes = n_classes              # Number of classes (5 in your case)
        self.root = None                        # Root node of the tree

    def fit(self, X, y, sample_weight=None):
        """Fit the decision tree to the data."""
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        self.n_features = X.shape[1]  # Number of features (187)
        if self.max_features == 'sqrt':
            self.max_features = int(np.sqrt(self.n_features))
        self.root = self._build_tree(np.arange(len(y)), 0, X, y, sample_weight)

    def _build_tree(self, indices, depth, X, y, sample_weight):
        """Recursively build the decision tree."""
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           len(indices) < self.min_samples_split or \
           len(np.unique(y[indices])) == 1:
            total_weight_per_class = np.bincount(y[indices], weights=sample_weight[indices], minlength=self.n_classes)
            value = np.argmax(total_weight_per_class)
            return Node(value=value)

        # Find the best split
        feature_subset = np.random.choice(self.n_features, self.max_features, replace=False)
        best_decrease = -np.inf
        best_feature = None
        best_threshold = None

        # Compute parent node impurity
        total_weight_per_class = np.bincount(y[indices], weights=sample_weight[indices], minlength=self.n_classes)
        total_weight = np.sum(total_weight_per_class)
        G_parent = 1 - np.sum((total_weight_per_class / total_weight)**2) if total_weight > 0 else 0

        for feature in feature_subset:
            sorted_idx = np.argsort(X[indices, feature])
            sorted_indices = indices[sorted_idx]
            left_weight_per_class = np.zeros(self.n_classes)
            right_weight_per_class = total_weight_per_class.copy()

            for k in range(len(indices) - 1):
                sample_k = sorted_indices[k]
                class_k = y[sample_k]
                weight_k = sample_weight[sample_k]
                left_weight_per_class[class_k] += weight_k
                right_weight_per_class[class_k] -= weight_k

                if X[sorted_indices[k], feature] < X[sorted_indices[k+1], feature]:
                    W_left = np.sum(left_weight_per_class)
                    W_right = np.sum(right_weight_per_class)
                    if W_left > 0 and W_right > 0:
                        G_left = 1 - np.sum((left_weight_per_class / W_left)**2) if W_left > 0 else 0
                        G_right = 1 - np.sum((right_weight_per_class / W_right)**2) if W_right > 0 else 0
                        G_split = (W_left / total_weight) * G_left + (W_right / total_weight) * G_right
                        impurity_decrease = G_parent - G_split
                        if impurity_decrease > best_decrease:
                            best_decrease = impurity_decrease
                            best_feature = feature
                            best_threshold = (X[sorted_indices[k], feature] + X[sorted_indices[k+1], feature]) / 2

        if best_feature is not None:
            mask = X[indices, best_feature] <= best_threshold
            left_indices = indices[mask]
            right_indices = indices[~mask]
            left_child = self._build_tree(left_indices, depth + 1, X, y, sample_weight)
            right_child = self._build_tree(right_indices, depth + 1, X, y, sample_weight)
            return Node(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)
        else:
            total_weight_per_class = np.bincount(y[indices], weights=sample_weight[indices], minlength=self.n_classes)
            value = np.argmax(total_weight_per_class)
            return Node(value=value)

    def predict(self, X):
        """Predict classes for input samples."""
        return np.array([self._predict_sample(self.root, x) for x in X])

    def _predict_sample(self, node, x):
        """Recursively predict for a single sample."""
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_sample(node.left, x)
        else:
            return self._predict_sample(node.right, x)

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, max_features='sqrt', class_weight=None, n_classes=5):
        self.n_estimators = n_estimators        # Number of trees
        self.max_depth = max_depth              # Maximum depth of each tree
        self.min_samples_split = min_samples_split  # Minimum samples to split
        self.max_features = max_features        # Features to consider per split
        self.class_weight = class_weight        # Class weights (None, 'balanced', or dict)
        self.n_classes = n_classes              # Number of classes (5)
        self.trees = []                         # List of decision trees

    def fit(self, X, y):
        """Fit the random forest to the data."""
        n_samples = len(y)

        # Compute sample weights based on class_weight
        if self.class_weight is None:
            sample_weight = np.ones(n_samples)
        elif self.class_weight == 'balanced':
            classes, counts = np.unique(y, return_counts=True)
            weights = n_samples / (len(classes) * counts)
            sample_weight = np.array([weights[c] for c in y])
        elif isinstance(self.class_weight, dict):
            sample_weight = np.array([self.class_weight[c] for c in y])
        else:
            raise ValueError("class_weight must be None, 'balanced', or a dictionary")

        # Build each tree
        for _ in tqdm(range(self.n_estimators)):
            boot_indices = np.random.choice(n_samples, n_samples, replace=True)
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                n_classes=self.n_classes
            )
            tree.fit(X[boot_indices], y[boot_indices], sample_weight[boot_indices])
            self.trees.append(tree)

    def predict(self, X):
        """Predict classes for input samples."""
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        preds = np.zeros(len(X), dtype=int)
        for i in range(len(X)):
            counts = np.bincount(all_preds[:, i], minlength=self.n_classes)
            preds[i] = np.argmax(counts)
        return preds
`

const randomForestClassifier = `from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Preprocess data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
clf.fit(X_train_resampled, y_train_resampled)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score, f1_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')
weighted_f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1-Score: {macro_f1:.4f}")
print(f"Weighted F1-Score: {weighted_f1:.4f}")
print(classification_report(y_test, y_pred))
`

const gradientBoostingClassifier = `from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# Preprocess data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
clf.fit(X_train_resampled, y_train_resampled)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score, f1_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')
weighted_f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1-Score: {macro_f1:.4f}")
print(f"Weighted F1-Score: {weighted_f1:.4f}")
print(classification_report(y_test, y_pred))
`

export {
	svmFromScratch,
	logisticRegressionFromScratch,
	svmSklearn,
	logisticRegressionSklearn,
	xgboostClassifier,
	smoteCode,
	fftCode,
  //this under is just show-case, we didn't use it in the projet
  gradientBoostingFromScratch,
  randomForestFromScratch,
  randomForestClassifier,
  gradientBoostingClassifier
}