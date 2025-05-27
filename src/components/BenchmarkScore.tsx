import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Button } from '@/components/ui/button';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import BenchmarkAllTable from './BenchmarkAllTable';

type VizType = 'roc' | 'precision-recall' | 'calibration';
type ModelType = 'XGBoost' | 'SVM' | 'Logistic';

const BenchmarkScore: React.FC = () => {
  const [benchmarkViz, setBenchmarkViz] = useState<VizType>('roc');
  const [selectedModel, setSelectedModel] = useState<ModelType>('XGBoost');

  const getModelFileName = (model: ModelType) => {
    return model === 'SVM' ? 'SVC' : model === 'Logistic' ? 'LogisticRegression' : model;
  };

  const getCurveFileName = (viz: VizType) => {
    return viz === 'roc' ? 'roc' : viz === 'precision-recall' ? 'precision_recall' : 'calibration';
  }

  return (
    <section className="mt-12 mb-8">
      <motion.h3
        className="text-2xl font-bold mb-6 text-center text-[#facc15]"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        Model Benchmark Results
      </motion.h3>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Performance Metrics */}
        <motion.div
          className="bg-[#222] rounded-lg p-5 border border-[#444] shadow-lg"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <h4 className="text-xl font-semibold mb-4 text-[#facc15]">Performance Metrics</h4>
          <div className="overflow-x-auto">
            <Table>
              <TableHeader className="bg-[#333]">
                <TableRow>
                  <TableHead className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                    Model
                  </TableHead>
                  <TableHead className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                    Accuracy
                  </TableHead>
                  <TableHead className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                    Macro F1
                  </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody className="bg-[#222] divide-y divide-gray-700">
                {[
                  { model: 'SVM', accuracy: '94%', f1: '0.77' },
                  { model: 'SVM (FFT)', accuracy: '95%', f1: '0.80' },
                  { model: 'XGBoost', accuracy: '90%', f1: '0.78' },
                  { model: 'XGBoost (FFT)', accuracy: '92%', f1: '0.81' },
                  { model: 'Logistic Regression', accuracy: '67%', f1: '0.48' },
                  { model: 'Logistic Regression (FFT)', accuracy: '70%', f1: '0.51' },
                ].map((row, index) => (
                  <TableRow key={index} className="hover:bg-[#333]">
                    <TableCell className="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-200">
                      {row.model}
                    </TableCell>
                    <TableCell className="px-4 py-3 whitespace-nowrap text-sm text-gray-200">
                      {row.accuracy}
                    </TableCell>
                    <TableCell className="px-4 py-3 whitespace-nowrap text-sm text-gray-200">
                      {row.f1}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
          <p className="mt-4 text-sm text-gray-400 italic">
            *FFT: Fast Fourier Transform applied to the raw ECG signal
          </p>
        </motion.div>

        {/* Visualization */}
        <motion.div
          className="bg-[#222] rounded-lg p-5 border border-[#444] shadow-lg flex flex-col"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
        >
          <div className="flex justify-between items-center mb-4">
            <h4 className="text-xl font-semibold text-[#facc15]">Model Visualizations</h4>
            <div className="flex space-x-2">
              {['roc', 'precision-recall', 'calibration'].map((viz) => (
                <Button
                  key={viz}
                  className={`text-xs px-2 py-1 rounded ${
                    benchmarkViz === viz ? 'bg-amber-600 text-white' : 'bg-[#333] text-gray-300'
                  }`}
                  onClick={() => setBenchmarkViz(viz as VizType)}
                >
                  {viz.charAt(0).toUpperCase() + viz.slice(1).replace('-', ' ')}
                </Button>
              ))}
            </div>
          </div>
          <div className="flex-1 flex flex-col items-center justify-center">
            <AnimatePresence mode="wait">
              <motion.img
                key={benchmarkViz}
                src={`/plot/benchmarkplot/${getCurveFileName(benchmarkViz)}_curve_${getModelFileName(selectedModel)}.png`}
                alt={`${benchmarkViz.charAt(0).toUpperCase() + benchmarkViz.slice(1)} Curves for ${selectedModel} Model`}
                className="rounded-lg border border-[#444] max-h-80 object-contain"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                transition={{ duration: 0.3 }}
              />
            </AnimatePresence>
            <p className="mt-3 text-sm text-gray-300 text-center px-4">
              {benchmarkViz === 'roc' &&
                'ROC curves showing true positive rate vs. false positive rate for different arrhythmia classes'}
              {benchmarkViz === 'precision-recall' &&
                'Precision-Recall curves showing precision vs. recall tradeoff for each heart rhythm class'}
              {benchmarkViz === 'calibration' &&
                'Calibration curves showing how well predicted probabilities match observed outcomes'}
            </p>
          </div>
          <div className="mt-4 flex justify-center space-x-4">
            {['XGBoost', 'SVM', 'Logistic'].map((model) => (
              <Button
                key={model}
                className={`px-3 py-1 text-sm rounded-full ${
                  selectedModel === model ? 'bg-amber-600 text-white' : 'bg-[#333] text-gray-300 hover:bg-[#444]'
                }`}
                onClick={() => setSelectedModel(model as ModelType)}
              >
                {model}
              </Button>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Key Insights & Additional Visualizations */}
      <motion.div
        className="mt-8"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.6 }}
      >
        <h4 className="text-xl font-semibold mb-4 text-center text-[#facc15]">
          Key Insights & Performance Analysis
        </h4>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {[
            {
              title: 'Class-wise Performance',
              text: 'SVM and XGBoost both performed exceptionally well on Normal and Ventricular classes (>95% F1-score), while the Unknown class presented the biggest challenge for all models with F1-scores below 65%.',
            },
            {
              title: 'FFT Transformation Effect',
              text: 'Models trained on FFT-transformed signals consistently outperformed their raw signal counterparts, with an average improvement of 2.4% in accuracy and 3.1% in F1-score.',
            },
            {
              title: 'From-Scratch vs. Library',
              text: 'From-scratch implementations performed within 5% of their library counterparts, with SVM from scratch achieving 90% accuracy compared to 94% from the library implementation.',
            },
          ].map((insight, index) => (
            <motion.div
              key={index}
              className="bg-[#222] p-4 rounded-lg border border-[#444]"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.8 + index * 0.2 }}
            >
              <h5 className="font-medium text-[#facc15] mb-2">{insight.title}</h5>
              <p className="text-sm text-gray-300">{insight.text}</p>
            </motion.div>
          ))}
        </div>

        <motion.div
          className="bg-[#222] p-5 rounded-lg border border-[#444] shadow-lg"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 1.4 }}
        >
          <h4 className="text-xl font-semibold mb-4 text-[#facc15]">Prediction Distribution</h4>
          <div className="flex flex-wrap justify-center gap-4 mb-4">
            {['XGBoost', 'SVM', 'Logistic'].map((model) => (
              <Button
                key={model}
                className={`text-xs px-2 py-1 rounded ${
                  selectedModel === model ? 'bg-amber-600 text-white' : 'bg-[#333] text-gray-300 hover:bg-[#444]'
                }`}
                onClick={() => setSelectedModel(model as ModelType)}
              >
                {model}
              </Button>
            ))}
          </div>
          <div className="flex flex-col lg:flex-row items-center justify-center gap-6">
            <div className="w-full lg:w-1/2">
              <motion.img
                src={`/plot/benchmarkplot/combined_histogram_${getModelFileName(selectedModel)}.png`}
                alt={`Combined Histogram for ${selectedModel} Model`}
                className="rounded-lg border border-[#444] max-h-80 object-contain mx-auto"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.3 }}
              />
              <p className="text-xs text-center text-gray-400 mt-2">
                Combined prediction probabilities across all classes
              </p>
            </div>
            <div className="w-full lg:w-1/2">
              <motion.img
                src={`/plot/benchmarkplot/facet_histogram_${getModelFileName(selectedModel)}.png`}
                alt={`Class-wise Histograms for ${selectedModel} Model`}
                className="rounded-lg border border-[#444] max-h-80 object-contain mx-auto"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.3 }}
              />
              <p className="text-xs text-center text-gray-400 mt-2">
                Class-specific prediction probability distributions
              </p>
            </div>
          </div>
          <div className="mt-4 p-4 bg-[#1a1a1a] rounded-lg">
            <h5 className="font-medium text-[#facc15] mb-2">Interpretation</h5>
            <p className="text-sm text-gray-300">
              {selectedModel === 'XGBoost' &&
                'XGBoost shows the most confident predictions with sharp probability distributions closer to 0 and 1, indicating strong decision boundaries. This explains its high performance metrics, especially for Normal and Ventricular classes.'}
              {selectedModel === 'SVM' &&
                'SVM demonstrates balanced probability distributions with good separation between classes, particularly for Normal beats. Its calibration is slightly less precise than XGBoost but better than Logistic Regression.'}
              {selectedModel === 'Logistic' &&
                'Logistic Regression shows more uncertainty in its predictions with flatter probability distributions. This model struggles more with the minor classes, resulting in lower overall performance metrics.'}
            </p>
          </div>
        </motion.div>
      </motion.div>

      {/* Class-Specific Performance Comparison */}
      <motion.div
        className="mt-6 bg-[#222] p-5 rounded-lg border border-[#444] shadow-lg"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 1.8 }}
      >
        <h4 className="text-xl font-semibold mb-4 text-[#facc15]">
          Class-Specific Performance Comparison
        </h4>
        <div className="overflow-x-auto">
          <Table>
            <TableHeader className="bg-[#333]">
              <TableRow>
                <TableHead className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Arrhythmia Class
                </TableHead>
                <TableHead className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  XGBoost F1-Score
                </TableHead>
                <TableHead className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  SVM F1-Score
                </TableHead>
                <TableHead className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Logistic F1-Score
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody className="bg-[#222] divide-y divide-gray-700">
              {[
                { class: 'Normal (N)', xgboost: '0.97', svm: '0.96', logistic: '0.85', xgboostColor: 'text-green-400', svmColor: 'text-green-400', logisticColor: 'text-yellow-400' },
                { class: 'Supraventricular (S)', xgboost: '0.83', svm: '0.81', logistic: '0.62', xgboostColor: 'text-yellow-400', svmColor: 'text-yellow-400', logisticColor: 'text-red-400' },
                { class: 'Ventricular (V)', xgboost: '0.95', svm: '0.94', logistic: '0.76', xgboostColor: 'text-green-400', svmColor: 'text-green-400', logisticColor: 'text-yellow-400' },
                { class: 'Fusion (F)', xgboost: '0.78', svm: '0.77', logistic: '0.52', xgboostColor: 'text-yellow-400', svmColor: 'text-yellow-400', logisticColor: 'text-red-400' },
                { class: 'Unknown (Q)', xgboost: '0.63', svm: '0.61', logistic: '0.39', xgboostColor: 'text-red-400', svmColor: 'text-red-400', logisticColor: 'text-red-400' },
              ].map((row, index) => (
                <TableRow key={index} className="hover:bg-[#333]">
                  <TableCell className="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-200">
                    {row.class}
                  </TableCell>
                  <TableCell className={`px-4 py-3 whitespace-nowrap text-sm ${row.xgboostColor}`}>
                    {row.xgboost}
                  </TableCell>
                  <TableCell className={`px-4 py-3 whitespace-nowrap text-sm ${row.svmColor}`}>
                    {row.svm}
                  </TableCell>
                  <TableCell className={`px-4 py-3 whitespace-nowrap text-sm ${row.logisticColor}`}>
                    {row.logistic}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
        <div className="mt-4 p-4 bg-[#1a1a1a] rounded-lg">
          <h5 className="font-medium text-[#facc15] mb-2">Class Performance Insights</h5>
          <ul className="list-disc pl-5 space-y-2 text-sm text-gray-300">
            <li>
              All models perform best on <span className="text-green-400 font-medium">Normal</span> and{' '}
              <span className="text-green-400 font-medium">Ventricular</span> classes due to their larger
              representation in the training data.
            </li>
            <li>
              The <span className="text-red-400 font-medium">Unknown</span> class is the most challenging,
              with consistently low F1-scores across all models.
            </li>
            <li>XGBoost outperforms other models across all classes, with the largest advantage in minority classes.</li>
            <li>
              Logistic Regression struggles significantly with minority classes, showing the limitations of linear
              decision boundaries for this task.
            </li>
          </ul>
        </div>
      </motion.div>
	  <BenchmarkAllTable/>
    </section>
  );
};

export default BenchmarkScore;