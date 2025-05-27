import { useState } from 'react';
import { motion } from 'framer-motion';

interface VisualizationProps {
  images: {
    src: string;
    title: string;
    description: string;
    alt: string;
  }[];
}

const VisualizationGallery = ({ images }: VisualizationProps) => {
  const [selectedImageIndex, setSelectedImageIndex] = useState(0);
  return (
    <div className="mt-8 mb-12">
      <div className="flex flex-col md:flex-row gap-6">
        {/* Mobile-first selection for small screens */}
        <div className="block md:hidden w-full mb-4">
          <div className="bg-[#333] p-4 rounded-lg border border-[#555] shadow-md">
            <h3 className="text-lg font-medium text-[#FFD700] mb-4">Select Visualization</h3>
            <select 
              value={selectedImageIndex}
              onChange={(e) => setSelectedImageIndex(parseInt(e.target.value))}
              className="w-full bg-[#444] text-gray-200 p-2 rounded-md border border-[#555] focus:outline-none focus:ring-2 focus:ring-[#FFD700]"
            >
              {images.map((image, index) => (
                <option key={index} value={index}>
                  {image.title}
                </option>
              ))}
            </select>
          </div>
        </div>
        
        {/* Main visualization area */}
        <div className="w-full md:w-2/3">
          <motion.div
            key={selectedImageIndex}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
            className="bg-[#333] p-3 sm:p-4 rounded-lg border border-[#555] shadow-md"
          >
            <div className="aspect-w-16 aspect-h-9 overflow-hidden rounded-md">
              <img
                src={images[selectedImageIndex].src}
                alt={images[selectedImageIndex].alt}
                className="object-contain w-full max-h-[60vh]"
              />
            </div>
            <h3 className="mt-4 text-lg font-medium text-[#FFD700]">
              {images[selectedImageIndex].title}
            </h3>
            <p className="mt-2 text-sm sm:text-base text-gray-300">
              {images[selectedImageIndex].description}
            </p>
          </motion.div>
        </div>
        
        {/* Desktop sidebar for visualization selection */}
        <div className="hidden md:block w-full md:w-1/3">
          <div className="bg-[#333] p-4 rounded-lg border border-[#555] shadow-md sticky top-4">
            <h3 className="text-lg font-medium text-[#FFD700] mb-4">Visualizations</h3>
            <div className="space-y-2 max-h-[400px] overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-[#555]">
              {images.map((image, index) => (
                <button
                  key={index}
                  onClick={() => setSelectedImageIndex(index)}
                  className={`w-full text-left p-2 rounded-md transition-colors ${
                    selectedImageIndex === index
                      ? "bg-[#FFD700] text-yellow-400 font-medium"
                      : "bg-[#444] text-gray-200 hover:bg-[#555]"
                  }`}
                >
                  {image.title}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VisualizationGallery;
