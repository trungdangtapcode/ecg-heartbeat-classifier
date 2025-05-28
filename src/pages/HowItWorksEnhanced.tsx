import { useRef } from 'react';
import { motion, useInView } from 'framer-motion';
import ResultsTable from '@/components/ResultsTable';
import VisualizationGallery from '@/components/VisualizationGallery';
import DataPreprocessing from '@/components/DataPreprocessing';
import ModelComparison from '@/components/ModelComparison';

// Enhanced and expanded report content with more detailed information
const reportContent = {
  introduction: {
    problem: "Analyzing electrocardiogram (ECG) data is crucial in diagnosing heart diseases, but this process is time-consuming and requires high expertise when done manually. The problem posed is to build an automated system for classifying heartbeats using machine learning to support faster and more accurate diagnosis. This project explores and compares the effectiveness of three popular classification algorithms: Support Vector Machine (SVM), Logistic Regression, and Gradient Boosting (XGBoost) on real heartbeat data, including both using libraries and reimplementing these algorithms from scratch.",
    dataset: "The dataset used in this project is the heartbeat dataset from Kaggle, linked to the MIT-BIH Arrhythmia Database. The dataset is pre-split into training ('mitbih_train.csv') with 87,553 samples and test ('mitbih_test.csv') with 21,891 samples. Each sample has 187 features (of type float64) representing data points on the ECG signal, with the final column being the label indicating the heart rhythm type, classified into 5 classes: Normal, Fusion, Ventricular, Supraventricular, and Unknown. The dataset exhibits significant class imbalance, with some classes having a much lower proportion.",
    objectives: [
      "Apply and evaluate classification algorithms: Implement and evaluate the effectiveness of three machine learning algorithms: Support Vector Machine (SVM), Logistic Regression, and Gradient Boosting (with the XGBoost variant) on real heartbeat data.",
      "Compare two implementation methods: Implement and use the algorithms in two ways: using existing machine learning libraries (such as scikit-learn, XGBoost) and reimplementing the algorithms from basic principles. Compare the results, performance, and understanding of the algorithms from these two approaches.",
      "Master the classification problem-solving process: Practice skills through the steps of a typical machine learning project, including: loading and exploring data, data preprocessing (normalization, handling class imbalance), model training, evaluating results using appropriate metrics (Accuracy, F1-score).",
      "Analyze and discuss results: Compare the performance between different algorithms as well as between the library versions and the from-scratch implementations. From there, draw conclusions about the advantages and disadvantages of each model and implementation method in the context of the heartbeat classification problem."
    ]
  },
  theoretical: {
    preprocessing: {
      scaling: "Data normalization brings features to the same scale, which is necessary for many algorithms. Z-score normalization is a common method, transforming data according to the formula zi = (xi - μ) / σ. In this project, StandardScaler from scikit-learn is used.",
      imbalance: "Imbalanced data (where the minority class has fewer samples than the majority class) causes the model to be biased towards the majority class. Oversampling techniques like SMOTE (Synthetic Minority Over-sampling Technique) are used to generate synthetic samples for minority classes to balance the class distribution. Class weight techniques are also applied by setting 'class_weight=balanced' parameter, assigning higher weights to minority classes.",
      fft: "Fast Fourier Transform (FFT) is used to transform the data from the time domain to the frequency domain, which can help machine learning models identify periodic characteristics in ECG signals. FFT can reveal patterns that might not be apparent in the original time-domain signal."
    },
    algorithms: {
      svm: "SVM classifies by finding the optimal separating hyperplane. For non-linear ECG data, the RBF kernel is used to map data to a higher-dimensional space. The regularization parameter C controls the trade-off between a large margin and classification error. Multi-class SVM uses One-vs-Rest or One-vs-One strategies. The project implements SVM with RBF kernel using the SVC library and also from scratch via Quadratic Programming.",
      logistic: "Logistic Regression is a linear model for classification, using a link function (Sigmoid for binary, Softmax for multi-class) to estimate probabilities. For the multi-class heartbeat problem, Softmax Regression is applied. The model is trained by minimizing the loss function (Negative Log-Likelihood with L2 regularization) directly using gradient-based methods.",
      gradient: "Gradient Boosting is an ensemble technique that sequentially builds models from weak learners (decision trees), correcting errors based on gradients. XGBoost (Extreme Gradient Boosting) is an optimized implementation of Gradient Boosting, outstanding in speed and performance due to improvements like regularization, handling missing values, and parallel computing. The project uses XGBoost for heartbeat classification, leveraging the power of boosting techniques."
    },
    metrics: "Evaluating the classification model on the test data uses appropriate metrics, especially for imbalanced data. The two main metrics are: Accuracy - The ratio of correct predictions to the total number of samples. Easy to understand but can be misleading with imbalanced data (Accuracy = Number of correct predictions / Total number of samples); F1-score - The harmonic mean of Precision and Recall, suitable for imbalanced data (F1-score = 2 × (Precision × Recall) / (Precision + Recall)). For multi-class problems, Macro F1-score (average F1 of each class) and Weighted F1-score (average F1 weighted by the number of samples) are used. The project uses both Accuracy and F1-score (macro and weighted) for a comprehensive evaluation."
  },
  methods: {
    preprocessing: {
      text: "This section describes the dataset and the steps to prepare the data for model training. The heartbeat dataset is taken from the files mitbih_train.csv and mitbih_test.csv, containing 87,553 and 21,891 samples respectively. All attributes are of type float64. The initial data is loaded into pandas DataFrames. After checking for missing values (none detected), the data is split into features X and labels y. X_train has shape (87553, 187), y_train has shape (87553,), X_test has shape (21891, 187), and y_test has shape (21891,).",
      steps: [
        {
          title: "Data Normalization",
          description: "Z-score normalization using StandardScaler is applied separately to X_train and X_test so that features have mean 0 and variance 1.",
          technical: "scaler = StandardScaler()\nX_train = scaler.fit_transform(X_train)\nX_test = scaler.transform(X_test)",
          image: "/illustration/standard_scaler.png"
        },
        {
          title: "Handling Class Imbalance", 
          description: "Two techniques are used: SMOTE and Class Weight. SMOTE generates synthetic samples for minority classes, increasing the training set to 362,350 samples. Class Weight assigns higher weights to minority classes using the 'balanced' parameter.",
          technical: "smote = SMOTE(random_state=42)\nX_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)"
        },
        {
          title: "Feature Transformation",
          description: "Fast Fourier Transform (FFT) is applied to transform data from time domain to frequency domain, helping identify periodic characteristics in ECG signals.",
          technical: "X_train_fft = np.abs(np.fft.fft(X_train))\nX_test_fft = np.abs(np.fft.fft(X_test))",
          image: "/illustration/fft.png"
        },
        {
          title: "Train-Test Split",
          description: "The dataset is already pre-split into training and test sets. The test set retains the original class distribution for realistic evaluation.",
          technical: "# No additional code needed as the dataset was provided pre-split"
        }
      ]
    },
    implementation: {
      libraries: {
        gradient: "The Gradient Boosting Classifier model is implemented using GradientBoostingClassifier from scikit-learn with parameters n_estimators=3, learning_rate=0.05, max_depth=5, and subsample=0.8. The model is trained on the preprocessed dataset with different approaches: SMOTE, Class Weight, and FFT.",
        xgboost: "The XGBoost model is implemented using the xgboost library with parameters objective='multi:softmax', num_class=5, learning_rate=0.05, max_depth=7, tree_method='hist', and device='cuda'. The training data is converted to the optimized DMatrix format for XGBoost. Grid search is used for hyperparameter tuning.",
        svm: "The SVM model is implemented using SVC from scikit-learn with RBF kernel (kernel='rbf'), C=1.0, and gamma='scale'. The model is trained on data processed with SMOTE, Class Weight, and FFT approaches.",
        logistic: "The Logistic Regression model is implemented using LogisticRegression from scikit-learn with parameters multi_class='multinomial', solver='saga', penalty='l1', and C=0.01. This model is trained on various processed datasets for comparison."
      },
      scratch: {
        svm: "The custom SVM implementation (SVMQP class) focuses on binary classification with RBF kernel, solving the Quadratic Programming problem for the dual form using cvxopt. The implementation includes kernel matrix computation, QP solving, support vector identification, bias calculation, and prediction. For multi-class classification, it's integrated with scikit-learn's OneVsRestClassifier. Due to computational constraints, this model was trained on a random subset of 10,000 samples.",
        logistic: "The custom Logistic Regression implementation (MyLogisticRegression class) handles multi-class classification using Softmax Regression. It optimizes the Negative Log-Likelihood with L2 regularization using scipy.optimize.minimize with the BFGS method. It includes one-hot encoding, loss and gradient computation, and probability/class prediction methods. This model was trained on the complete training dataset after preprocessing."
      }
    },
    environment: "The experiments were conducted on Kaggle Notebooks, providing computational resources and libraries for machine learning tasks. Configuration includes: Platform - Kaggle Notebooks; Hardware - GPU for training acceleration, especially XGBoost; OS - Linux; Python - 3.10; Libraries - numpy, pandas, sklearn, imblearn (SMOTE), cvxopt, scipy, xgboost. GPU usage significantly reduces training time for complex models and large datasets post-oversampling."
  },
  results: {
    library: {
      text: "This section presents the performance evaluation results of the classification models implemented using popular machine learning libraries on the test dataset. The metrics used are Accuracy, Precision, Recall and F1-score, as explained in the Theoretical Background section.",
      table: [
        { model: "Gradient Boosting", accuracy: "0.74", macroF1: "0.53", macroPrecision: "0.50", macroRecall: "0.69" },
        { model: "XGBoost", accuracy: "0.90", macroF1: "0.78", macroPrecision: "0.74", macroRecall: "0.90" },
        { model: "SVM", accuracy: "0.94", macroF1: "0.77", macroPrecision: "0.72", macroRecall: "0.91"  },
        { model: "Logistic Regression", accuracy: "0.67", macroF1: "0.48", macroPrecision: "0.45", macroRecall: "0.77" }
      ]
    },
    scratch: {
      text: "This section presents the performance comparison between the custom implementations and library-based implementations of the classification models.",
      table: [
        { model: "SVM (Library)", accuracy: "0.94", macroF1: "0.77", weightedF1: "0.94" },
        { model: "SVM (Scratch)", accuracy: "0.90", macroF1: "0.72", weightedF1: "0.89" },
        { model: "Logistic Regression (Library)", accuracy: "0.67", macroF1: "0.48", weightedF1: "0.64" },
        { model: "Logistic Regression (Scratch)", accuracy: "0.63", macroF1: "0.45", weightedF1: "0.61" }
      ]
    },
    fft: {
      text: "This section compares the performance of models trained on raw ECG signals versus those trained on FFT-transformed data.",
      table: [
        { model: "XGBoost)", accuracy: "0.90", macroF1: "0.78", weightedF1: "0.90" },
        { model: "XGBoost (FFT)", accuracy: "0.92", macroF1: "0.81", weightedF1: "0.92" },
        { model: "SVM", accuracy: "0.94", macroF1: "0.77", weightedF1: "0.94" },
        { model: "SVM (FFT)", accuracy: "0.95", macroF1: "0.80", weightedF1: "0.95" }
      ]
    },
    hyperparameter: {
      text: "Grid search was performed to find optimal hyperparameters for each model. The results below show how different parameter settings affected performance.",
      svm: {
        C: [0.1, 1, 10, 100],
        gamma: ['scale', 'auto', 0.001, 0.01, 0.1],
        bestParams: { C: 10, gamma: 'scale' }
      },
      logistic: {
        C: [0.001, 0.01, 0.1, 1],
        penalty: ['l1', 'l2'],
        bestParams: { C: 0.01, penalty: 'l1' }
      },
      xgboost: {
        learning_rate: [0.01, 0.05, 0.1],
        max_depth: [3, 5, 7],
        bestParams: { learning_rate: 0.05, max_depth: 7 }
      }
    }
  },  visualizations: [
    {
      src: "/plot/fft.png",
      title: "Time Domain to Frequency Domain",
      description: "Visualization of an ECG signal in both time domain (original) and frequency domain (after FFT transformation). The FFT transformation reveals periodic patterns that might not be visible in the original signal.",
      alt: "FFT transformation of ECG signal"
    },
    {
      src: "/plot/feature_importance/features_importance_mi.png",
      title: "Feature Importance - Mutual Information",
      description: "Top 20 most important features identified by mutual information analysis for heart rhythm classification. This helps identify which specific parts of the ECG signal contribute most to accurate classification.",
      alt: "Feature importance chart using mutual information"
    },
    {
      src: "/plot/feature_importance/features_importance_rf.png",
      title: "Feature Importance - Random Forest",
      description: "Important features identified by Random Forest feature importance, showing which ECG signal features have the highest predictive power. Random Forest provides a different perspective on feature importance compared to statistical methods.",
      alt: "Feature importance chart from Random Forest"
    },
	  {
      src: "/plot/feature_importance/features_importance_f_score.png",
      title: "Feature Importance - ANOVA",
      description: "Important features identified by the ANOVA F-score, highlighting which ECG signal features best separate class means. ",
      alt: "Feature importance chart from ANOVA F-Score"
    },
    {
      src: "/plot/feature_importance/features_importance_chi2.png",
      title: "Feature Importance - Chi2",
      description: "Important features identified by the Chi-squared test, showing which ECG signal features have the strongest statistical association with the target variable",
      alt: "Feature importance chart from ANOVA Chi-square"
    },
    {
      src: "/plot/correlation_matrix.png",
      title: "Correlation Matrix",
      description: "Correlation matrix  shows the pairwise linear relationships between different time-dependent variables (or between lagged versions of the same variable).",
      alt: "Correlation matrix for time series data"
    },
    {
      src: "/plot/hyperparam_search/accuracy_svmhyper.png",
      title: "SVM Hyperparameter Tuning - Accuracy",
      description: "Effect of different C and gamma values on SVM accuracy. Higher values of C lead to more complex models that may overfit, while appropriate gamma values help capture the right level of detail in the data.",
      alt: "SVM hyperparameter tuning for accuracy"
    },
    {
      src: "/plot/hyperparam_search/f1score_svmhyper.png",
      title: "SVM Hyperparameter Tuning - F1 Score",
      description: "Impact of SVM hyperparameters on F1-score, which is particularly important for imbalanced datasets like ECG classification. The plot shows how different combinations of C and gamma affect the balance between precision and recall.",
      alt: "SVM hyperparameter tuning for F1-score"
    },
    {
      src: "/plot/hyperparam_search/lr_acc.png",
      title: "Logistic Regression Hyperparameter Tuning",
      description: "Accuracy results from hyperparameter tuning for Logistic Regression. The plot shows how regularization strength (C) and penalty type affect model performance on ECG classification.",
      alt: "Logistic Regression hyperparameter tuning"
    },
    {
      src: "/plot/benchmarkplot/roc_curve_XGBoost.png",
      title: "ROC Curves for XGBoost",
      description: "ROC curves showing the true positive rate against the false positive rate for different thresholds for each heart rhythm class. The area under the curve (AUC) indicates how well the model distinguishes between classes.",
      alt: "XGBoost ROC curve"
    },
    {
      src: "/plot/benchmarkplot/roc_curve_SVC.png",
      title: "ROC Curves for SVM",
      description: "ROC curves for the SVM model across different heart rhythm classes. These curves demonstrate the superior performance of SVM in distinguishing between normal heartbeats and various arrhythmias.",
      alt: "SVM ROC curve"
    },
    {
      src: "/plot/benchmarkplot/precision_recall_curve_XGBoost.png",
      title: "Precision-Recall Curves for XGBoost",
      description: "Precision-Recall curves for XGBoost model, which are particularly useful for evaluating performance on imbalanced datasets. The curves show how the model handles the trade-off between precision and recall for each class.",
      alt: "XGBoost Precision-Recall curve"
    },
    {
      src: "/plot/sklearn_logistic_coef.png",
      title: "Logistic Regression Coefficients",
      description: "Visualization of the learned coefficients in the scikit-learn logistic regression model, showing how different features contribute to the classification decision for each heart rhythm class.",
      alt: "Logistic regression coefficients"
    },
    {
      src: "/plot/my_logistic_coef_bfgs_step.gif",
      title: "Custom Logistic Regression Training",
      description: "Animated visualization of how coefficients evolve during the training of the custom logistic regression implementation using the BFGS optimization method. Shows how the model converges to find optimal coefficient values over iterations.",
      alt: "Animated training of logistic regression"
    },
    {
      src: "/plot/benchmarkplot/calibration_curve_XGBoost.png",
      title: "Probability Calibration for XGBoost",
      description: "Calibration curves showing how well the predicted probabilities of the XGBoost model match the actual outcomes. Well-calibrated probabilities are essential for making reliable risk assessments in clinical applications.",
      alt: "XGBoost calibration curve"
    },
    {
      src: "/plot/FOL_outlier.png",
      title: "ECG Outlier Detection",
      description: "Visualization of outliers detected in the ECG dataset using First-Order-Lowpass (FOL) filtering. Identifying outliers is important for data cleaning and understanding anomalous heart rhythms that may require special attention.",
      alt: "ECG outlier detection"
    },
    {
      src: "/plot/benchmarkplot/facet_histogram_XGBoost.png",
      title: "Class Probability Distributions - XGBoost",
      description: "Faceted histograms showing the distribution of predicted probabilities for each heart rhythm class using XGBoost. These distributions illustrate how confidently the model assigns probabilities to different classes.",
      alt: "XGBoost probability distributions per class"
    }
  ],
  modelComparison: {
    library: [
      {
        name: "Support Vector Machine (SVM)",
        type: "library",
        description: "SVM with RBF kernel from scikit-learn that achieved the highest accuracy (94%) among all tested models.",
        advantages: [
          "Best overall performance on the test dataset",
          "Handles non-linear relationships in ECG data well",
          "Robust to overfitting with proper hyperparameter tuning"
        ],
        limitations: [
          "Slower training time compared to other models",
          "Memory-intensive for large datasets",
          "Less interpretable than some other models"
        ],
        implementationDetails: "Implemented using SVC from scikit-learn with RBF kernel, C=10.0, and gamma='scale'. Trained on data processed with SMOTE for handling class imbalance."
      },
      {
        name: "XGBoost",
        type: "library",
        description: "Optimized gradient boosting implementation that achieved excellent results with relatively fast training times.",
        advantages: [
          "Second-best performance with 90% accuracy",
          "Faster training than SVM, especially with GPU acceleration",
          "Built-in feature importance for interpretability"
        ],
        limitations: [
          "More complex model with many hyperparameters",
          "Can overfit on small datasets without proper regularization",
          "Requires more memory than simple models"
        ],
        implementationDetails: "Implemented using the xgboost library with objective='multi:softmax', num_class=5, learning_rate=0.05, max_depth=7, using CUDA for GPU acceleration."
      },
      {
        name: "Logistic Regression",
        type: "library",
        description: "Linear model for multi-class classification using the softmax function.",
        advantages: [
          "Fastest training and inference times",
          "Most interpretable model with clear feature coefficients",
          "Low memory requirements"
        ],
        limitations: [
          "Lowest accuracy (67%) among tested models",
          "Cannot model complex non-linear relationships",
          "Less effective on imbalanced data"
        ],
        implementationDetails: "Implemented using LogisticRegression from scikit-learn with multi_class='multinomial', solver='saga', penalty='l1', and C=0.01."
      }
    ],
    scratch: [
      {
        name: "SVM (From Scratch)",
        type: "scratch",
        description: "Custom implementation of SVM using Quadratic Programming for solving the dual form optimization problem.",
        advantages: [
          "Provides deep understanding of the SVM algorithm",
          "Competitive performance (90% accuracy) compared to library version (94%)",
          "Flexibility to modify the optimization process"
        ],
        limitations: [
          "Training limited to a subset of 10,000 samples due to computational constraints",
          "Slower implementation compared to optimized libraries",
          "More complex implementation requiring mathematical background"
        ],
        implementationDetails: "Implemented in the SVMQP class, solving the Quadratic Programming problem for dual SVM using cvxopt. Used RBF kernel and was integrated with OneVsRestClassifier for multi-class classification."
      },
      {
        name: "Logistic Regression (From Scratch)",
        type: "scratch",
        description: "Custom implementation of multi-class logistic regression using softmax and gradient-based optimization.",
        advantages: [
          "Deep understanding of logistic regression principles",
          "Close performance (63% accuracy) to library version (67%)",
          "Direct control over optimization process"
        ],
        limitations: [
          "Slower convergence compared to specialized solvers",
          "More complex implementation requiring calculus knowledge",
          "Less numerically stable without specialized techniques"
        ],
        implementationDetails: "Implemented in the MyLogisticRegression class, optimizing Negative Log-Likelihood with L2 regularization using scipy.optimize.minimize with BFGS method."
      }
    ]
  },
  conclusion: {
    summary: "In this study, we explored heart rhythm classification using three different algorithms. The SVM model achieved the highest accuracy (94%) and F1-score (77%) on the test set, followed by XGBoost (90% accuracy, 78% F1-score). Logistic Regression had lower performance but was still effective for certain classes. The FFT transformation improved model performance across all algorithms, indicating that frequency domain features are valuable for ECG classification. The from-scratch implementations performed slightly worse than their library counterparts but provided valuable insights into the underlying algorithms.",
    limitations: [
      "The custom SVM implementation was trained on a smaller subset due to computational constraints",
      "Class imbalance remains a challenge despite mitigation strategies",
      "The evaluation metrics might not fully capture clinical significance of different types of errors"
    ],
    futureWork: [
      "Explore deep learning approaches like Convolutional Neural Networks for automatic feature extraction",
      "Implement real-time classification for continuous monitoring",
      "Incorporate domain expert knowledge for better feature engineering",
      "Extend the approach to more specific ECG abnormalities detection"
    ]
  }
};

const HowItWorksEnhanced = () => {
  // Create refs for each section to track visibility
  const introductionRef = useRef(null);
  const theoreticalRef = useRef(null);
  const methodsRef = useRef(null);
  const resultsRef = useRef(null);
  const conclusionRef = useRef(null);

  // Use useInView to detect when each section is in view
  const isIntroductionInView = useInView(introductionRef, { once: false, margin: '-100px' });
  const isTheoreticalInView = useInView(theoreticalRef, { once: false, margin: '-100px' });
  const isMethodsInView = useInView(methodsRef, { once: false, margin: '-100px' });
  const isResultsInView = useInView(resultsRef, { once: false, margin: '-100px' });
  const isConclusionInView = useInView(conclusionRef, { once: false, margin: '-100px' });

  return (
    <div className="bg-[#363636] text-gray-200 min-h-screen grow w-full">
      <header className="bg-[#363636] p-6 text-center">
        <h1 className="text-4xl font-bold text-[#FFD700]">How It Works</h1>
      </header>
      <nav className="sticky top-0 bg-[#2D2D2D] p-4 shadow z-20">
        <ul className="flex justify-center space-x-6">
          <li><a href="#introduction" className="text-[#FFD700] font-medium hover:text-white transition-colors">Introduction</a></li>
          <li><a href="#theoretical" className="text-[#FFD700] font-medium hover:text-white transition-colors">Theoretical Background</a></li>
          <li><a href="#methods" className="text-[#FFD700] font-medium hover:text-white transition-colors">Experimental Methods</a></li>
          <li><a href="#results" className="text-[#FFD700] font-medium hover:text-white transition-colors">Results</a></li>
          <li><a href="#conclusion" className="text-[#FFD700] font-medium hover:text-white transition-colors">Conclusion</a></li>
        </ul>
      </nav>
      <main className="px-4">
        <motion.section
          ref={introductionRef}
          initial={{ opacity: 0, y: 50 }}
          animate={isIntroductionInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 50 }}
          transition={{ duration: 0.5 }}
          id="introduction"
          className="max-w-6xl mx-auto my-8 p-6 bg-[#404040] rounded-lg shadow-lg border-l-4 border-[#FFD700]"
        >
          <h2 className="text-3xl font-semibold text-[#FFD700]">Introduction</h2>
          <div className="w-12 h-1 bg-[#FFD700] mt-2 mb-6"></div>
          
          <h3 className="text-xl font-medium mt-6 mb-3 text-[#FFD700]">Problem Introduction</h3>
          <p className="leading-relaxed">{reportContent.introduction.problem}</p>
          
          <h3 className="text-xl font-medium mt-6 mb-3 text-[#FFD700]">Dataset Introduction</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            <div className="bg-[#333] p-4 rounded-lg border border-[#555] shadow-md col-span-2">
              <p className="leading-relaxed">{reportContent.introduction.dataset}</p>
            </div>
            <div className="bg-[#333] p-4 rounded-lg border border-[#555] shadow-md">
              <h4 className="font-medium text-[#FFD700] mb-2">Dataset Statistics</h4>
              <ul className="list-disc pl-5 space-y-1">
                <li>Training samples: <span className="font-mono">87,553</span></li>
                <li>Testing samples: <span className="font-mono">21,891</span></li>
                <li>Feature dimensions: <span className="font-mono">187</span></li>
                <li>Classes: <span className="font-mono">5</span></li>
                <li>Data type: <span className="font-mono">float64</span></li>
              </ul>
            </div>
          </div>
          
          <h3 className="text-xl font-medium mt-6 mb-3 text-[#FFD700]">Project Objectives</h3>
          <ul className="list-disc pl-5 leading-relaxed">
            {reportContent.introduction.objectives.map((obj, index) => (
              <li key={index} className="mb-2">{obj}</li>
            ))}
          </ul>
        </motion.section>

        <motion.section
          ref={theoreticalRef}
          initial={{ opacity: 0, y: 50 }}
          animate={isTheoreticalInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 50 }}
          transition={{ duration: 0.5 }}
          id="theoretical"
          className="max-w-6xl mx-auto my-8 p-6 bg-[#404040] rounded-lg shadow-lg border-l-4 border-[#FFD700]"
        >
          <h2 className="text-3xl font-semibold text-[#FFD700]">Theoretical Background</h2>
          <div className="w-12 h-1 bg-[#FFD700] mt-2 mb-6"></div>
          
          <h3 className="text-xl font-medium mt-6 mb-4 text-[#FFD700]">Data Preprocessing</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <div className="bg-[#333] p-4 rounded-lg border border-[#555] shadow-md">
              <h4 className="font-medium text-[#FFD700] mb-2">Data Scaling</h4>
              <p className="leading-relaxed">{reportContent.theoretical.preprocessing.scaling}</p>
            </div>
            <div className="bg-[#333] p-4 rounded-lg border border-[#555] shadow-md">
              <h4 className="font-medium text-[#FFD700] mb-2">Handling Imbalanced Data</h4>
              <p className="leading-relaxed">{reportContent.theoretical.preprocessing.imbalance}</p>
            </div>
          </div>
          <div className="bg-[#333] p-4 rounded-lg border border-[#555] shadow-md mb-6">
            <h4 className="font-medium text-[#FFD700] mb-2">Fast Fourier Transform (FFT)</h4>
            <p className="leading-relaxed">{reportContent.theoretical.preprocessing.fft}</p>
          </div>
          
          <h3 className="text-xl font-medium mt-8 mb-4 text-[#FFD700]">Classification Algorithms</h3>
          <div className="space-y-4 mb-6">
            <div className="bg-[#333] p-4 rounded-lg border border-[#555] shadow-md">
              <h4 className="font-medium text-[#FFD700] mb-2">Support Vector Machine (SVM)</h4>
              <p className="leading-relaxed">{reportContent.theoretical.algorithms.svm}</p>
            </div>
            <div className="bg-[#333] p-4 rounded-lg border border-[#555] shadow-md">
              <h4 className="font-medium text-[#FFD700] mb-2">Logistic Regression</h4>
              <p className="leading-relaxed">{reportContent.theoretical.algorithms.logistic}</p>
            </div>
            <div className="bg-[#333] p-4 rounded-lg border border-[#555] shadow-md">
              <h4 className="font-medium text-[#FFD700] mb-2">Gradient Boosting and XGBoost</h4>
              <p className="leading-relaxed">{reportContent.theoretical.algorithms.gradient}</p>
            </div>
          </div>
          
          <h3 className="text-xl font-medium mt-8 mb-3 text-[#FFD700]">Evaluation Metrics</h3>
          <div className="bg-[#333] p-4 rounded-lg border border-[#555] shadow-md">
            <p className="leading-relaxed">{reportContent.theoretical.metrics}</p>
          </div>
        </motion.section>

        <motion.section
          ref={methodsRef}
          initial={{ opacity: 0, y: 50 }}
          animate={isMethodsInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 50 }}
          transition={{ duration: 0.5 }}
          id="methods"
          className="max-w-6xl mx-auto my-8 p-6 bg-[#404040] rounded-lg shadow-lg border-l-4 border-[#FFD700]"
        >
          <h2 className="text-3xl font-semibold text-[#FFD700]">Experimental Methods</h2>
          <div className="w-12 h-1 bg-[#FFD700] mt-2 mb-6"></div>
          
          <h3 className="text-xl font-medium mt-6 mb-4 text-[#FFD700]">Data Description and Preprocessing</h3>
          <p className="leading-relaxed mb-6">{reportContent.methods.preprocessing.text}</p>
          
          <DataPreprocessing steps={reportContent.methods.preprocessing.steps} />
          
          <h3 className="text-xl font-medium mt-10 mb-4 text-[#FFD700]">Model Implementation</h3>
          
          <div className="mb-6">
            <h4 className="text-lg font-medium mt-6 mb-4 text-amber-300">Using Libraries</h4>
            <div className="space-y-4">
              <div className="bg-[#333] p-4 rounded-lg border border-[#555] shadow-md">
                <h5 className="font-medium text-amber-200 mb-2">Gradient Boosting</h5>
                <p className="leading-relaxed">{reportContent.methods.implementation.libraries.gradient}</p>
              </div>
              <div className="bg-[#333] p-4 rounded-lg border border-[#555] shadow-md">
                <h5 className="font-medium text-amber-200 mb-2">XGBoost</h5>
                <p className="leading-relaxed">{reportContent.methods.implementation.libraries.xgboost}</p>
              </div>
              <div className="bg-[#333] p-4 rounded-lg border border-[#555] shadow-md">
                <h5 className="font-medium text-amber-200 mb-2">Support Vector Machine</h5>
                <p className="leading-relaxed">{reportContent.methods.implementation.libraries.svm}</p>
              </div>
              <div className="bg-[#333] p-4 rounded-lg border border-[#555] shadow-md">
                <h5 className="font-medium text-amber-200 mb-2">Logistic Regression</h5>
                <p className="leading-relaxed">{reportContent.methods.implementation.libraries.logistic}</p>
              </div>
            </div>
          </div>
          
          <div className="mb-6">
            <h4 className="text-lg font-medium mt-8 mb-4 text-amber-300">From-Scratch Implementation</h4>
            <div className="space-y-4">
              <div className="bg-[#333] p-4 rounded-lg border border-[#555] shadow-md">
                <h5 className="font-medium text-amber-200 mb-2">Support Vector Machine</h5>
                <p className="leading-relaxed">{reportContent.methods.implementation.scratch.svm}</p>
              </div>
              <div className="bg-[#333] p-4 rounded-lg border border-[#555] shadow-md">
                <h5 className="font-medium text-amber-200 mb-2">Logistic Regression</h5>
                <p className="leading-relaxed">{reportContent.methods.implementation.scratch.logistic}</p>
              </div>
            </div>
          </div>
          
          <h3 className="text-xl font-medium mt-10 mb-3 text-[#FFD700]">Experimental Environment</h3>
          <div className="bg-[#333] p-4 rounded-lg border border-[#555] shadow-md">
            <p className="leading-relaxed">{reportContent.methods.environment}</p>
          </div>
        </motion.section>

        <motion.section
          ref={resultsRef}
          initial={{ opacity: 0, y: 50 }}
          animate={isResultsInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 50 }}
          transition={{ duration: 0.5 }}
          id="results"
          className="max-w-6xl mx-auto my-8 p-6 bg-[#404040] rounded-lg shadow-lg border-l-4 border-[#FFD700]"
        >
          <h2 className="text-3xl font-semibold text-[#FFD700]">Results and Discussion</h2>
          <div className="w-12 h-1 bg-[#FFD700] mt-2 mb-6"></div>
          
          {/* Library Models Results */}
          <ResultsTable 
            title="Evaluation Results of Library Models" 
            description={reportContent.results.library.text}
            results={reportContent.results.library.table} 
          />
          
          {/* From-Scratch vs Library Comparison */}
          <ResultsTable 
            title="Comparison: Library vs From-Scratch Implementations" 
            description={reportContent.results.scratch.text}
            results={reportContent.results.scratch.table} 
          />
          
          {/* FFT Transformation Results */}
          <ResultsTable 
            title="Effect of FFT Transformation on Model Performance" 
            description={reportContent.results.fft.text}
            results={reportContent.results.fft.table} 
          />

          {/* Hyperparameter Tuning Results */}
          <div className="mt-8 mb-6">
            <h3 className="text-xl font-medium mb-3 text-[#FFD700]">Hyperparameter Tuning Results</h3>
            <p className="mb-4 text-gray-300">{reportContent.results.hyperparameter.text}</p>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-[#333] p-4 rounded-lg border border-[#555] shadow-md">
                <h4 className="font-medium text-amber-200 mb-3">SVM Hyperparameters</h4>
                <div className="mb-3">
                  <p className="text-sm font-medium text-gray-300 mb-1">C values tested:</p>
                  <div className="flex flex-wrap gap-2">
                    {reportContent.results.hyperparameter.svm.C.map((val, idx) => (
                      <span key={idx} className="bg-[#444] px-2 py-1 rounded text-xs">
                        {val}
                      </span>
                    ))}
                  </div>
                </div>
                <div className="mb-3">
                  <p className="text-sm font-medium text-gray-300 mb-1">Gamma values tested:</p>
                  <div className="flex flex-wrap gap-2">
                    {reportContent.results.hyperparameter.svm.gamma.map((val, idx) => (
                      <span key={idx} className="bg-[#444] px-2 py-1 rounded text-xs">
                        {val}
                      </span>
                    ))}
                  </div>
                </div>                <div className="mt-4 pt-3 border-t border-[#555]">
                  <p className="text-sm font-medium text-amber-200">Best parameters:</p>
                  <p className="font-mono text-sm mt-1">
                    C = {reportContent.results.hyperparameter.svm.bestParams.C}, 
                    gamma = '{reportContent.results.hyperparameter.svm.bestParams.gamma}'
                  </p>
                  <div className="mt-2 text-xs text-gray-400">
                    <p>Based on multiple metrics from plots: accuracy_svmhyper.png, f1score_svmhyper.png, precision_svmhyper.png, and recall_svmhyper.png</p>
                  </div>
                </div>
              </div>
              
              <div className="bg-[#333] p-4 rounded-lg border border-[#555] shadow-md">
                <h4 className="font-medium text-amber-200 mb-3">Logistic Regression Hyperparameters</h4>
                <div className="mb-3">
                  <p className="text-sm font-medium text-gray-300 mb-1">C values tested:</p>
                  <div className="flex flex-wrap gap-2">
                    {reportContent.results.hyperparameter.logistic.C.map((val, idx) => (
                      <span key={idx} className="bg-[#444] px-2 py-1 rounded text-xs">
                        {val}
                      </span>
                    ))}
                  </div>
                </div>
                <div className="mb-3">
                  <p className="text-sm font-medium text-gray-300 mb-1">Penalty types tested:</p>
                  <div className="flex flex-wrap gap-2">
                    {reportContent.results.hyperparameter.logistic.penalty.map((val, idx) => (
                      <span key={idx} className="bg-[#444] px-2 py-1 rounded text-xs">
                        {val}
                      </span>
                    ))}
                  </div>
                </div>                <div className="mt-4 pt-3 border-t border-[#555]">
                  <p className="text-sm font-medium text-amber-200">Best parameters:</p>
                  <p className="font-mono text-sm mt-1">
                    C = {reportContent.results.hyperparameter.logistic.bestParams.C}, 
                    penalty = '{reportContent.results.hyperparameter.logistic.bestParams.penalty}'
                  </p>
                  <div className="mt-2 text-xs text-gray-400">
                    <p>Results visualized in lr_acc.png, lr_f1.png, lr_precision.png, and lr_recall.png</p>
                  </div>
                </div>
              </div>
              
              <div className="bg-[#333] p-4 rounded-lg border border-[#555] shadow-md">
                <h4 className="font-medium text-amber-200 mb-3">XGBoost Hyperparameters</h4>
                <div className="mb-3">
                  <p className="text-sm font-medium text-gray-300 mb-1">Learning rate values tested:</p>
                  <div className="flex flex-wrap gap-2">
                    {reportContent.results.hyperparameter.xgboost.learning_rate.map((val, idx) => (
                      <span key={idx} className="bg-[#444] px-2 py-1 rounded text-xs">
                        {val}
                      </span>
                    ))}
                  </div>
                </div>
                <div className="mb-3">
                  <p className="text-sm font-medium text-gray-300 mb-1">Max depth values tested:</p>
                  <div className="flex flex-wrap gap-2">
                    {reportContent.results.hyperparameter.xgboost.max_depth.map((val, idx) => (
                      <span key={idx} className="bg-[#444] px-2 py-1 rounded text-xs">
                        {val}
                      </span>
                    ))}
                  </div>
                </div>                <div className="mt-4 pt-3 border-t border-[#555]">
                  <p className="text-sm font-medium text-amber-200">Best parameters:</p>
                  <p className="font-mono text-sm mt-1">
                    learning_rate = {reportContent.results.hyperparameter.xgboost.bestParams.learning_rate}, 
                    max_depth = {reportContent.results.hyperparameter.xgboost.bestParams.max_depth}
                  </p>
                  <div className="mt-2 text-xs text-gray-400">
                    <p>Additional parameters used: objective='multi:softmax', num_class=5, tree_method='hist', device='cuda'</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Visualizations Gallery */}
          <div className="mt-10">
            <h3 className="text-xl font-medium mb-4 text-[#FFD700]">Visualizations</h3>
            <VisualizationGallery images={reportContent.visualizations} />
          </div>
          
          {/* Model Comparison */}
          <div className="mt-10">
            <h3 className="text-xl font-medium mb-4 text-[#FFD700]">Model Comparison</h3>
            <ModelComparison 
              models={[
                ...reportContent.modelComparison.library,
                ...reportContent.modelComparison.scratch
              ]} 
            />
          </div>
        </motion.section>
        
        <motion.section
          ref={conclusionRef}
          initial={{ opacity: 0, y: 50 }}
          animate={isConclusionInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 50 }}
          transition={{ duration: 0.5 }}
          id="conclusion"
          className="max-w-6xl mx-auto my-8 p-6 bg-[#404040] rounded-lg shadow-lg border-l-4 border-[#FFD700]"
        >
          <h2 className="text-3xl font-semibold text-[#FFD700]">Conclusion</h2>
          <div className="w-12 h-1 bg-[#FFD700] mt-2 mb-6"></div>
          
          <h3 className="text-xl font-medium mt-6 mb-3 text-[#FFD700]">Summary</h3>
          <p className="leading-relaxed mb-6">{reportContent.conclusion.summary}</p>
            <h3 className="text-xl font-medium mt-6 mb-3 text-[#FFD700]">Key Insights from Visualizations</h3>
          <div className="bg-[#333] p-4 rounded-lg border border-[#555] shadow-md mb-6">
            <ul className="list-disc pl-5 space-y-2 text-gray-300">
              <li>
                <strong className="text-[#FFD700]">Feature Importance:</strong> From the feature importance visualizations (features_importance_mi.png and features_importance_rf.png), we observed that specific segments of the ECG signal (particularly around QRS complexes) contribute significantly more to classification accuracy.
              </li>
              <li>
                <strong className="text-[#FFD700]">Calibration Analysis:</strong> The calibration curves (calibration_curve_XGBoost.png, calibration_curve_SVC.png) revealed that SVM tends to produce less calibrated probability outputs compared to XGBoost, which has implications for reliability in clinical settings.
              </li>
              <li>
                <strong className="text-[#FFD700]">Model Performance:</strong> The ROC and precision-recall curves clearly illustrate that SVM and XGBoost significantly outperform basic Gradient Boosting and Logistic Regression across all heart rhythm classes.
              </li>
              <li>
                <strong className="text-[#FFD700]">Frequency Domain Benefits:</strong> FFT transformation plots demonstrate how certain patterns become more distinguishable in the frequency domain, explaining the consistent performance improvement when using FFT-transformed data.
              </li>
              <li>
                <strong className="text-[#FFD700]">Hyperparameter Sensitivity:</strong> The hyperparameter tuning plots for SVM and Logistic Regression reveal that these models are particularly sensitive to their regularization parameters (C), with sharp performance drops at suboptimal values.
              </li>
            </ul>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-[#333] p-4 rounded-lg border border-[#555] shadow-md">
              <h3 className="text-lg font-medium mb-3 text-[#FFD700]">Limitations</h3>
              <ul className="list-disc pl-5 space-y-2">
                {reportContent.conclusion.limitations.map((limitation, index) => (
                  <li key={index} className="text-gray-300">{limitation}</li>
                ))}
              </ul>
            </div>
            
            <div className="bg-[#333] p-4 rounded-lg border border-[#555] shadow-md">
              <h3 className="text-lg font-medium mb-3 text-[#FFD700]">Future Work</h3>
              <ul className="list-disc pl-5 space-y-2">
                {reportContent.conclusion.futureWork.map((work, index) => (
                  <li key={index} className="text-gray-300">{work}</li>
                ))}
              </ul>
            </div>
          </div>
        </motion.section>
      </main>
    </div>
  );
};

export default HowItWorksEnhanced;
