import { motion } from 'framer-motion';

interface DataProcessingStep {
  title: string;
  description: string;
  technical?: string;
  image?: string;
}

interface DataPreprocessingProps {
  steps: DataProcessingStep[];
}

const DataPreprocessing = ({ steps }: DataPreprocessingProps) => {  return (
    <div className="space-y-6 sm:space-y-8 my-4 sm:my-6">
      {steps.map((step, index) => (
        <motion.div
          key={index}
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: index * 0.1 }}
          className="bg-[#404040] rounded-lg p-4 sm:p-6 border-l-4 border-[#FFD700] shadow-md"
        >
          <div className="flex flex-col md:flex-row gap-4 md:gap-6">
            <div className={`${step.image ? 'md:w-2/3' : 'w-full'}`}>
              <div className="flex items-center mb-3">
                <span className="flex items-center justify-center w-6 h-6 rounded-full bg-[#FFD700] text-black text-xs font-bold mr-2">
                  {index + 1}
                </span>
                <h3 className="text-base sm:text-lg font-medium text-[#FFD700]">
                  {step.title}
                </h3>
              </div>
              <p className="text-sm sm:text-base text-gray-300 mb-3 sm:mb-4">{step.description}</p>
              {step.technical && (
                <div className="bg-[#333] p-3 sm:p-4 rounded border border-[#555] mt-3">
                  <h4 className="text-xs sm:text-sm font-medium text-gray-200 mb-2">Technical Implementation:</h4>
                  <p className="text-gray-300 text-xs sm:text-sm">{step.technical}</p>
                </div>
              )}
            </div>
            {step.image && (
              <div className="mt-4 md:mt-0 md:w-1/3">
                <div className="bg-[#333] p-3 sm:p-4 rounded-lg overflow-hidden shadow">
                  <img 
                    src={step.image} 
                    alt={`Visualization for ${step.title}`} 
                    className="w-full h-auto rounded object-contain max-h-[200px] mx-auto"
                  />
                </div>
              </div>
            )}
          </div>
        </motion.div>
      ))}
    </div>
  );
};

export default DataPreprocessing;
