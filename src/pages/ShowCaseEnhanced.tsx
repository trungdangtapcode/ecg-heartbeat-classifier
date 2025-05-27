import { useState } from 'react';
import useEmblaCarousel from 'embla-carousel-react';
import Autoplay from 'embla-carousel-autoplay';
import CodeTypingEffect from '@/components/TypingCodeEffect';
import AlgorithmCards from '@/components/AlgorithmCards';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { motion } from 'framer-motion';

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
    C=10.0,
    gamma='scale',
    class_weight='balanced',
    decision_function_shape='ovo',
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
    'objective': 'multi:softmax',
    'num_class': 5,
    'learning_rate': 0.05,
    'max_depth': 7,
    'min_child_weight': 1,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'device': 'cuda' # For GPU acceleration
}

# Train the model
num_round = 500
model = xgb.train(
    params,
    dtrain,
    num_round,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=50,
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

// Algorithm Details for AlgorithmCards component
const algorithms = [
  {
    name: "Support Vector Machine",
    type: "Classification Algorithm",
    description: "SVM is a powerful algorithm that works by finding the optimal hyperplane that separates different classes with the maximum margin. For non-linear ECG data, we use the RBF kernel to map data to a higher-dimensional space where it becomes linearly separable.",
    mathFormulation: "For binary classification, SVM solves this optimization problem:\n\nmin (1/2)||w||² + C∑ξᵢ\ns.t. yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ\n     ξᵢ ≥ 0\n\nFor multi-class, we use One-vs-Rest strategy, training K binary classifiers (one per class).\n\nThe RBF kernel function is: K(x₁,x₂) = exp(-γ||x₁-x₂||²)",
    complexity: {
      training: "O(n²) to O(n³) where n is the number of training samples",
      inference: "O(nsv × d) where nsv is the number of support vectors and d is the dimensionality"
    },
    codeSnippet: {
      library: svmSklearn,
      scratch: svmFromScratch
    },
    illustration: "/plot/hyperparameter_search/accuracy_svmhyper.png"
  },
  {
    name: "Logistic Regression",
    type: "Classification Algorithm",
    description: "Logistic Regression is a linear model for classification that uses the sigmoid (binary) or softmax (multi-class) function to estimate class probabilities. It's simple, interpretable, and serves as a good baseline for many classification problems.",
    mathFormulation: "For multi-class classification with K classes, the probability of class k is:\n\np(y=k|x) = exp(wₖᵀx) / ∑ₖ exp(wₖᵀx)\n\nThe model is trained by minimizing the negative log-likelihood:\n\nL(w) = -∑ᵢ∑ₖ yᵢₖ log(p(yᵢ=k|xᵢ)) + λ||w||²\n\nwhere λ is the regularization parameter (λ = 1/C)",
    complexity: {
      training: "O(n × d × i) where n is sample size, d is dimensions, i is iterations",
      inference: "O(d × K) where d is dimensions and K is number of classes"
    },
    codeSnippet: {
      library: logisticRegressionSklearn,
      scratch: logisticRegressionFromScratch
    },
    illustration: "/plot/sklearn_logistic_coef.png"
  },
  {
    name: "XGBoost",
    type: "Ensemble Algorithm",
    description: "XGBoost (Extreme Gradient Boosting) is an optimized implementation of gradient boosting that sequentially builds an ensemble of weak tree-based models. Each new model corrects the errors of the previous ones, leading to high performance on structured data like ECG signals.",
    mathFormulation: "XGBoost minimizes the objective function:\n\nObj = ∑ᵢ l(yᵢ, ŷᵢ) + ∑ₖ Ω(fₖ)\n\nwhere l is a differentiable loss function, and Ω is a regularization term:\n\nΩ(f) = γT + (1/2)λ||w||²\n\nFor tree f with weights w, T is the number of leaves, γ and λ are regularization parameters.",
    complexity: {
      training: "O(n × d × log(n) × K × B) for n samples, d features, K classes, B boosting rounds",
      inference: "O(D × B) for D total depth of all trees and B boosting rounds"
    },
    codeSnippet: {
      library: xgboostClassifier,
      scratch: "# XGBoost is a complex algorithm optimized through years of research.\n# A from-scratch implementation would be extensive.\n# For learning purposes, we can use the library implementation\n# and examine its internals via documentation and publications."
    },
    illustration: "/plot/benchmarkplot/roc_curve_XGBoost.png"
  }
];

// Main ShowCase Component
const ShowCase = () => {
  const [emblaRef, emblaApi] = useEmblaCarousel({ loop: true }, [Autoplay({ delay: 5000 })]);
  const [typingEnabled, setTypingEnabled] = useState(true);
  const [activeTab, setActiveTab] = useState("code");

  const scrollPrev = () => emblaApi && emblaApi.scrollPrev();
  const scrollNext = () => emblaApi && emblaApi.scrollNext();
  const toggleTyping = () => setTypingEnabled(prev => !prev);

  return (
    <div className="w-full min-h-screen bg-[#0d0d0d] text-[#facc15] py-12 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Header Section */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl md:text-5xl font-extrabold mb-4 drop-shadow-md bg-gradient-to-r from-yellow-400 via-yellow-300 to-yellow-500 bg-clip-text text-transparent">
            Heart Rhythm Classification
          </h1>
          <p className="text-xl text-amber-200 max-w-3xl mx-auto">
            Exploring machine learning algorithms for ECG heartbeat classification: library-based and custom implementations
          </p>
        </motion.div>

        {/* Tabs Navigation */}
        <Tabs 
          defaultValue="code" 
          className="w-full mb-12"
          value={activeTab}
          onValueChange={setActiveTab}
        >
          <TabsList className="grid grid-cols-3 max-w-md mx-auto bg-[#222] mb-6">
            <TabsTrigger value="code" className="data-[state=active]:bg-amber-600 text-white">
              Model Implementation
            </TabsTrigger>
            <TabsTrigger value="algorithms" className="data-[state=active]:bg-amber-600 text-white">
              Algorithms
            </TabsTrigger>
            <TabsTrigger value="preprocessing" className="data-[state=active]:bg-amber-600 text-white">
              Preprocessing
            </TabsTrigger>
          </TabsList>

          {/* Code Implementation Tab */}
          <TabsContent value="code" className="mt-4">
            <div className="flex items-center justify-between w-full mb-8">
              <h2 className="ml-4 md:ml-10 text-3xl font-bold drop-shadow-md bg-gradient-to-r from-yellow-400 via-yellow-300 to-yellow-500 bg-clip-text text-transparent">
                Model Implementation
              </h2>
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium">Typing Effect</span>
                <label className="relative inline-flex items-center cursor-pointer mr-4 md:mr-8">
                  <input 
                    type="checkbox" 
                    className="sr-only peer" 
                    checked={typingEnabled}
                    onChange={toggleTyping}
                  />
                  <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-[#facc15]"></div>
                </label>
              </div>
            </div>

            <div className="embla w-full overflow-hidden rounded-lg shadow-lg border border-[#333]" ref={emblaRef}>
              <div className="min-h-60 embla__container flex">
                <div className="embla__slide flex-[0_0_100%] min-w-0 bg-[#1a1a1a] p-4 rounded-lg">
                  <CodeTypingEffect title="SVM from Scratch" code={svmFromScratch} typingEnabled={typingEnabled} />
                </div>
                <div className="embla__slide flex-[0_0_100%] min-w-0 bg-[#1a1a1a] p-4 rounded-lg">
                  <CodeTypingEffect title="Logistic Regression from Scratch" code={logisticRegressionFromScratch} typingEnabled={typingEnabled} />
                </div>
                <div className="embla__slide flex-[0_0_100%] min-w-0 bg-[#1a1a1a] p-4 rounded-lg">
                  <CodeTypingEffect title="SVM (scikit-learn)" code={svmSklearn} typingEnabled={typingEnabled} />
                </div>
                <div className="embla__slide flex-[0_0_100%] min-w-0 bg-[#1a1a1a] p-4 rounded-lg">
                  <CodeTypingEffect title="Logistic Regression (scikit-learn)" code={logisticRegressionSklearn} typingEnabled={typingEnabled} />
                </div>
                <div className="embla__slide flex-[0_0_100%] min-w-0 bg-[#1a1a1a] p-4 rounded-lg">
                  <CodeTypingEffect title="XGBoost Classifier" code={xgboostClassifier} typingEnabled={typingEnabled} />
                </div>
              </div>
            </div>

            <div className="flex justify-center gap-4 mt-6">
              <button
                className="bg-[#facc15] hover:bg-yellow-400 text-black font-bold py-2 px-5 rounded-lg transition duration-300 shadow-md"
                onClick={scrollPrev}
              >
                Previous
              </button>
              <button
                className="bg-[#facc15] hover:bg-yellow-400 text-black font-bold py-2 px-5 rounded-lg transition duration-300 shadow-md"
                onClick={scrollNext}
              >
                Next
              </button>
            </div>
          </TabsContent>

          {/* Algorithms Tab */}
          <TabsContent value="algorithms" className="mt-4">
            <AlgorithmCards algorithms={algorithms} />
          </TabsContent>

          {/* Preprocessing Tab */}
          <TabsContent value="preprocessing" className="mt-4">
            <div className="max-w-5xl mx-auto">
              <h2 className="text-3xl font-bold mb-8 text-center drop-shadow-md bg-gradient-to-r from-yellow-400 via-yellow-300 to-yellow-500 bg-clip-text text-transparent">
                Data Preprocessing Techniques
              </h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
                <div className="bg-[#1a1a1a] p-6 rounded-lg border border-[#333] shadow-lg">
                  <h3 className="text-xl font-semibold text-[#facc15] mb-4">SMOTE: Handling Class Imbalance</h3>
                  <p className="text-gray-300 mb-4">
                    Synthetic Minority Over-sampling Technique generates synthetic samples for minority classes to balance the class distribution. 
                    This helps the model learn patterns from all classes equally rather than favoring the majority class.
                  </p>
                  <div className="bg-[#0d0d0d] rounded-lg overflow-x-auto">
                    <pre className="p-4 text-sm text-gray-300 font-mono">{smoteCode}</pre>
                  </div>
                </div>
                
                <div className="bg-[#1a1a1a] p-6 rounded-lg border border-[#333] shadow-lg">
                  <h3 className="text-xl font-semibold text-[#facc15] mb-4">Fast Fourier Transform (FFT)</h3>
                  <p className="text-gray-300 mb-4">
                    FFT transforms the ECG signal from time domain to frequency domain, revealing patterns that might not be apparent in the original signal. 
                    This transformation can help machine learning models identify periodic characteristics essential for heartbeat classification.
                  </p>
                  <div className="bg-[#0d0d0d] rounded-lg overflow-x-auto">
                    <pre className="p-4 text-sm text-gray-300 font-mono">{fftCode}</pre>
                  </div>
                </div>
              </div>
              
              <div className="bg-[#1a1a1a] p-6 rounded-lg border border-[#333] shadow-lg">
                <h3 className="text-xl font-semibold text-[#facc15] mb-4">Preprocessing Pipeline</h3>
                <ol className="list-decimal pl-6 space-y-4 text-gray-300">
                  <li>
                    <strong className="text-amber-200">Data Loading:</strong> Load the ECG signals and their corresponding labels from the MIT-BIH Arrhythmia Database files.
                  </li>
                  <li>
                    <strong className="text-amber-200">Feature Engineering:</strong> Extract time-domain features from the ECG signals and/or transform signals using FFT to capture frequency-domain features.
                  </li>
                  <li>
                    <strong className="text-amber-200">Data Normalization:</strong> Apply Z-score normalization using StandardScaler so that all features have a mean of 0 and variance of 1.
                  </li>
                  <li>
                    <strong className="text-amber-200">Class Imbalance Handling:</strong> Apply SMOTE to generate synthetic samples for minority classes or use class weights to assign higher importance to minority classes.
                  </li>
                  <li>
                    <strong className="text-amber-200">Train-Test Split:</strong> Use the pre-defined train/test split from the dataset to evaluate model performance on unseen data.
                  </li>
                  <li>
                    <strong className="text-amber-200">Model Training:</strong> Train various models (SVM, Logistic Regression, XGBoost) on the preprocessed data with hyperparameter tuning.
                  </li>
                  <li>
                    <strong className="text-amber-200">Performance Evaluation:</strong> Evaluate models using metrics appropriate for imbalanced classification: accuracy, precision, recall, and F1-score.
                  </li>
                </ol>
                
                <div className="mt-8 text-center">
                  <img 
                    src="/plot/fft.png" 
                    alt="ECG Signal Transformation with FFT" 
                    className="max-w-full mx-auto rounded-lg border border-[#555]" 
                  />
                  <p className="mt-2 text-sm text-gray-400">Transformation of an ECG signal from time domain to frequency domain using Fast Fourier Transform</p>
                </div>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default ShowCase;
