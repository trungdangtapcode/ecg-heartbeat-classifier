import { useState } from 'react';
import useEmblaCarousel from 'embla-carousel-react';
import Autoplay from 'embla-carousel-autoplay';
import CodeTypingEffect from '@/components/TypingCodeEffect';
import AlgorithmCards from '@/components/AlgorithmCards';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { motion } from 'framer-motion';
import BenchmarkScore from '@/components/BenchmarkScore';
import {
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
} from '@/components/PythonCodeSnippets';


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
    illustration: "/illustration/svm_rbf.png"
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
    illustration: "/illustration/lr_bfgs.png"
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
    illustration: "/illustration/xgboost.png"
  }
];

// Main ShowCase Component
const ShowCase = () => {
  const [emblaRef, emblaApi] = useEmblaCarousel({ loop: true }, [Autoplay({ delay: 5000 })]);
  const [typingEnabled, setTypingEnabled] = useState(true);
  const [activeTab, setActiveTab] = useState("code");
//   const [benchmarkViz, setBenchmarkViz] = useState<'roc' | 'precision-recall' | 'calibration'>('roc');
//   const [selectedModel, setSelectedModel] = useState<'XGBoost' | 'SVM' | 'Logistic'>('XGBoost');

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
                <div className="embla__slide flex-[0_0_100%] min-w-0 bg-[#1a1a1a] p-4 rounded-lg">
                  <CodeTypingEffect title="Gradient Boosting from Scratch" code={gradientBoostingFromScratch} typingEnabled={typingEnabled} />
                </div>
                <div className="embla__slide flex-[0_0_100%] min-w-0 bg-[#1a1a1a] p-4 rounded-lg">
                  <CodeTypingEffect title="Random Forest from Scratch" code={randomForestFromScratch} typingEnabled={typingEnabled} />
                </div>
                <div className="embla__slide flex-[0_0_100%] min-w-0 bg-[#1a1a1a] p-4 rounded-lg">
                  <CodeTypingEffect title="Gradient Boosting (scikit-learn)" code={gradientBoostingClassifier} typingEnabled={typingEnabled} />
                </div>
                <div className="embla__slide flex-[0_0_100%] min-w-0 bg-[#1a1a1a] p-4 rounded-lg">
                  <CodeTypingEffect title="Random Forest (scikit-learn)" code={randomForestClassifier} typingEnabled={typingEnabled} />
                </div>
              </div>
            </div>

            <div className="flex justify-center gap-4 mt-6">
              <button
                className="bg-[#facc15] hover:bg-yellow-400 text-white font-bold py-2 px-5 rounded-lg transition duration-300 shadow-md"
                onClick={scrollPrev}
              >
                Previous
              </button>
              <button
                className="bg-[#facc15] hover:bg-yellow-400 text-white font-bold py-2 px-5 rounded-lg transition duration-300 shadow-md"
                onClick={scrollNext}
              >
                Next
              </button>
            </div>
            <BenchmarkScore/>
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
                    <pre className="p-4 text-sm text-gray-300 font-mono text-left">{smoteCode}</pre>
                  </div>
                </div>
                
                <div className="bg-[#1a1a1a] p-6 rounded-lg border border-[#333] shadow-lg">
                  <h3 className="text-xl font-semibold text-[#facc15] mb-4">Fast Fourier Transform (FFT)</h3>
                  <p className="text-gray-300 mb-4">
                    FFT transforms the ECG signal from time domain to frequency domain, revealing patterns that might not be apparent in the original signal. 
                    This transformation can help machine learning models identify periodic characteristics essential for heartbeat classification.
                  </p>
                  <div className="bg-[#0d0d0d] rounded-lg overflow-x-auto">
                    <pre className="p-4 text-sm text-gray-300 font-mono text-left">{fftCode}</pre>
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
