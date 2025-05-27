import { motion } from 'framer-motion';
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

interface ModelInfo {
  name: string;
  type: string;  // Allow any string value, but typically 'library' or 'scratch'
  description: string;
  advantages: string[];
  limitations: string[];
  implementationDetails: string;
}

interface ModelComparisonProps {
  models: ModelInfo[];
  onSelectModel?: (model: ModelInfo) => void;
}

const ModelComparison = ({ models, onSelectModel }: ModelComparisonProps) => {
  const libraryModels = models.filter(model => model.type === 'library');
  const scratchModels = models.filter(model => model.type === 'scratch');

  const cardVariants = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0, transition: { duration: 0.5 } },
    hover: { scale: 1.03, boxShadow: "0 10px 15px rgba(0,0,0,0.2)" }
  };
    return (
    <div className="mt-6 sm:mt-8">
      <div className="mb-8">
        <h3 className="text-lg sm:text-xl font-medium mb-3 sm:mb-4 text-[#FFD700]">Library-based Implementations</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
          {libraryModels.map((model, index) => (
            <motion.div 
              key={index}
              variants={cardVariants}
              initial="initial"
              animate="animate"
              whileHover="hover"
              transition={{ delay: index * 0.1 }}
            >
              <Card className="bg-[#444] border-[#555] overflow-hidden h-full">
                <div className="h-2 bg-gradient-to-r from-yellow-400 to-amber-600"></div>
                <CardContent className="p-4 sm:p-6">
                  <h4 className="text-base sm:text-lg font-semibold text-[#FFD700] mb-2 sm:mb-3">{model.name}</h4>
                  <p className="text-gray-300 text-xs sm:text-sm mb-3 sm:mb-4">{model.description}</p>
                  
                  <div className="mb-3 sm:mb-4">
                    <h5 className="text-xs sm:text-sm font-medium text-gray-200 mb-1 sm:mb-2">Advantages:</h5>
                    <ul className="list-disc pl-4 sm:pl-5 text-gray-300 text-xs sm:text-sm space-y-0.5 sm:space-y-1">
                      {model.advantages.map((advantage, i) => (
                        <li key={i}>{advantage}</li>
                      ))}
                    </ul>
                  </div>
                    <div className="mb-3 sm:mb-4">
                    <h5 className="text-xs sm:text-sm font-medium text-gray-200 mb-1 sm:mb-2">Limitations:</h5>
                    <ul className="list-disc pl-4 sm:pl-5 text-gray-300 text-xs sm:text-sm space-y-0.5 sm:space-y-1">
                      {model.limitations.map((limitation, i) => (
                        <li key={i}>{limitation}</li>
                      ))}
                    </ul>
                  </div>
                  
                  {onSelectModel && (
                    <Button 
                      onClick={() => onSelectModel(model)}
                      variant="secondary" 
                      className="mt-1 sm:mt-2 w-full bg-[#555] hover:bg-[#666] text-white text-xs sm:text-sm py-1 px-2 h-auto sm:h-9"
                    >
                      View Details
                    </Button>
                  )}
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      </div>

      <div>
        <h3 className="text-lg sm:text-xl font-medium mb-3 sm:mb-4 text-[#FFD700]">From-Scratch Implementations</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 sm:gap-6">
          {scratchModels.map((model, index) => (
            <motion.div 
              key={index}
              variants={cardVariants}
              initial="initial"
              animate="animate"
              whileHover="hover"
              transition={{ delay: index * 0.1 }}
            >              <Card className="bg-[#444] border-[#555] overflow-hidden h-full">
                <div className="h-2 bg-gradient-to-r from-yellow-600 to-amber-800"></div>
                <CardContent className="p-4 sm:p-6">
                  <h4 className="text-base sm:text-lg font-semibold text-[#FFD700] mb-2 sm:mb-3">{model.name}</h4>
                  <p className="text-gray-300 text-xs sm:text-sm mb-3 sm:mb-4">{model.description}</p>
                  
                  <div className="mb-3 sm:mb-4">
                    <h5 className="text-xs sm:text-sm font-medium text-gray-200 mb-1 sm:mb-2">Implementation Details:</h5>
                    <p className="text-gray-300 text-xs sm:text-sm">{model.implementationDetails}</p>
                  </div>
                  
                  <div className="mb-3 sm:mb-4">
                    <h5 className="text-xs sm:text-sm font-medium text-gray-200 mb-1 sm:mb-2">Advantages:</h5>
                    <ul className="list-disc pl-4 sm:pl-5 text-gray-300 text-xs sm:text-sm space-y-0.5 sm:space-y-1">
                      {model.advantages.map((advantage, i) => (
                        <li key={i}>{advantage}</li>
                      ))}
                    </ul>
                  </div>
                  
                  <div className="mb-3 sm:mb-4">
                    <h5 className="text-xs sm:text-sm font-medium text-gray-200 mb-1 sm:mb-2">Limitations:</h5>
                    <ul className="list-disc pl-4 sm:pl-5 text-gray-300 text-xs sm:text-sm space-y-0.5 sm:space-y-1">
                      {model.limitations.map((limitation, i) => (
                        <li key={i}>{limitation}</li>
                      ))}
                    </ul>
                  </div>
                    {onSelectModel && (
                    <Button 
                      onClick={() => onSelectModel(model)}
                      variant="secondary" 
                      className="mt-1 sm:mt-2 w-full bg-[#555] hover:bg-[#666] text-white text-xs sm:text-sm py-1 px-2 h-auto sm:h-9"
                    >
                      View Details
                    </Button>
                  )}
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ModelComparison;
