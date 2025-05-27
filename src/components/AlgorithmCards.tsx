import { useState } from 'react';
import { motion } from 'framer-motion';
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardFooter, 
  CardHeader, 
  CardTitle 
} from "@/components/ui/card";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";

interface AlgorithmProps {
  algorithms: {
    name: string;
    type: string;
    description: string;
    mathFormulation: string;
    complexity: {
      training: string;
      inference: string;
    };
    codeSnippet: {
      library: string;
      scratch: string;
    };
    illustration?: string;
  }[];
}

const AlgorithmCards = ({ algorithms }: AlgorithmProps) => {
  const [selectedAlgorithm, setSelectedAlgorithm] = useState(algorithms[0]);
  return (
    <div className="my-8 sm:my-12 px-2 sm:px-4">
      <h2 className="text-2xl sm:text-3xl font-bold text-center mb-6 sm:mb-8 text-[#FFD700]">Classification Algorithms</h2>
      
      {/* Mobile dropdown for algorithm selection */}
      <div className="block sm:hidden mb-6">
        <label htmlFor="algorithm-select" className="block text-sm font-medium text-gray-300 mb-2">
          Select Algorithm:
        </label>
        <select
          id="algorithm-select"
          value={selectedAlgorithm.name}
          onChange={(e) => {
            const selected = algorithms.find(algo => algo.name === e.target.value);
            if (selected) setSelectedAlgorithm(selected);
          }}
          className="w-full bg-[#333] text-gray-200 p-3 rounded-md border border-gray-700 focus:outline-none focus:ring-2 focus:ring-[#FFD700]"
        >
          {algorithms.map((algo) => (
            <option key={algo.name} value={algo.name}>
              {algo.name} ({algo.type})
            </option>
          ))}
        </select>
      </div>
      
      {/* Desktop/tablet card grid */}
      <div className="hidden sm:grid grid-cols-2 lg:grid-cols-3 gap-4 md:gap-6 mb-8 md:mb-10">
        {algorithms.map((algo, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: index * 0.1 }}
          >
            <Card 
              className={`cursor-pointer h-full transition-all duration-300 overflow-hidden 
              ${selectedAlgorithm.name === algo.name 
                ? 'border-[#FFD700] bg-gradient-to-b from-[#3A3A3A] to-[#333]' 
                : 'border-gray-700 bg-[#333] hover:border-gray-500'}`}
              onClick={() => setSelectedAlgorithm(algo)}
            >
              <div className={`h-1.5 ${selectedAlgorithm.name === algo.name ? 'bg-[#FFD700]' : 'bg-gray-700'}`}></div>
              <CardHeader className="pb-2 p-4 sm:p-6">
                <div className="flex justify-between items-center">
                  <Badge variant="outline" className="bg-[#444] text-xs font-normal text-gray-300 border-gray-600">
                    {algo.type}
                  </Badge>
                </div>
                <CardTitle className="mt-2 text-xl text-gray-100">{algo.name}</CardTitle>
              </CardHeader>
              <CardContent className="text-sm text-gray-300">
                <p className="line-clamp-3">{algo.description}</p>
              </CardContent>
              <CardFooter className="pt-0">
                <Button 
                  variant="ghost" 
                  size="sm" 
                  className={`text-xs ${selectedAlgorithm.name === algo.name ? 'text-[#FFD700]' : 'text-gray-400'}`}
                >
                  {selectedAlgorithm.name === algo.name ? 'Selected' : 'Select'}
                </Button>
              </CardFooter>
            </Card>
          </motion.div>
        ))}
      </div>      <motion.div
        key={selectedAlgorithm.name}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
        className="bg-[#333] rounded-lg border border-gray-700 overflow-hidden shadow-lg"
      >
        <div className="p-4 sm:p-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-4">
            <div>
              <h3 className="text-xl sm:text-2xl font-bold text-[#FFD700] mb-1">{selectedAlgorithm.name}</h3>
              <span className="inline-block bg-[#444] px-2 py-0.5 rounded text-xs font-medium text-gray-300 border-gray-600">
                {selectedAlgorithm.type}
              </span>
            </div>
          </div>
          
          <p className="text-sm sm:text-base text-gray-300 mb-6">{selectedAlgorithm.description}</p>
          
          <div className="mb-6">
            <h4 className="text-lg font-medium text-[#FFD700] mb-3">Mathematical Formulation</h4>
            <div className="bg-[#2B2B2B] p-3 sm:p-4 rounded border border-gray-700 overflow-x-auto">
              <p className="text-gray-300 whitespace-pre-wrap font-mono text-xs sm:text-sm">
                {selectedAlgorithm.mathFormulation}
              </p>
            </div>
          </div>

          <div className="mb-6">
            <h4 className="text-lg font-medium text-[#FFD700] mb-3">Computational Complexity</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-[#2B2B2B] p-3 sm:p-4 rounded border border-gray-700">
                <h5 className="text-sm font-medium text-gray-200 mb-2">Training:</h5>
                <p className="text-xs sm:text-sm text-gray-300 font-mono">{selectedAlgorithm.complexity.training}</p>
              </div>
              <div className="bg-[#2B2B2B] p-3 sm:p-4 rounded border border-gray-700">
                <h5 className="text-sm font-medium text-gray-200 mb-2">Inference:</h5>
                <p className="text-xs sm:text-sm text-gray-300 font-mono">{selectedAlgorithm.complexity.inference}</p>
              </div>
            </div>
          </div>          {selectedAlgorithm.illustration && (
            <div className="mb-6">
              <h4 className="text-lg font-medium text-[#FFD700] mb-3">Visual Explanation</h4>
              <div className="bg-[#2B2B2B] p-3 sm:p-4 rounded border border-gray-700 flex justify-center">
                <img 
                  src={selectedAlgorithm.illustration} 
                  alt={`${selectedAlgorithm.name} illustration`} 
                  className="max-w-full max-h-[200px] sm:max-h-[300px] object-contain"
                />
              </div>
            </div>
          )}

          <div>
            <h4 className="text-lg font-medium text-[#FFD700] mb-3">Implementation</h4>
            <Tabs defaultValue="library" className="w-full">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="library" className="text-xs sm:text-sm">Library Implementation</TabsTrigger>
                <TabsTrigger value="scratch" className="text-xs sm:text-sm">From Scratch</TabsTrigger>
              </TabsList>
              <TabsContent value="library" className="mt-4">
                <div className="bg-[#1a1a1a] p-3 sm:p-4 rounded border border-gray-700">
                  <div className="overflow-x-auto">
                    <pre className="text-gray-300 font-mono text-xs sm:text-sm whitespace-pre">
                      {selectedAlgorithm.codeSnippet.library}
                    </pre>
                  </div>
                </div>
              </TabsContent>
              <TabsContent value="scratch" className="mt-4">
                <div className="bg-[#1a1a1a] p-3 sm:p-4 rounded border border-gray-700">
                  <div className="overflow-x-auto">
                    <pre className="text-gray-300 font-mono text-xs sm:text-sm whitespace-pre">
                      {selectedAlgorithm.codeSnippet.scratch}
                    </pre>
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default AlgorithmCards;
