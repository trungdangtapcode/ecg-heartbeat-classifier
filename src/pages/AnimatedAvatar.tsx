// components/AnimatedAvatar.tsx
"use client";

import { useEffect, useState } from "react";
import { motion, useAnimation } from "framer-motion";
import { cn } from "@/lib/utils"; // assuming you're using shadcn utils

type Props = {
  src?: string;
  size?: number;
};

const clipShapes = [
  `polygon(
    50% 0%, 79.4% 9.6%, 95.1% 34.6%, 95.1% 65.4%, 
    79.4% 90.4%, 50% 100%, 20.6% 90.4%, 4.9% 65.4%, 
    4.9% 34.6%, 20.6% 9.6%
  )`,
  `polygon(
    30% 0%, 70% 0%, 100% 30%, 100% 70%, 
    70% 100%, 30% 100%, 0% 70%, 0% 30%
  )`,
];

export const AnimatedAvatar = ({ src = "/wtm_about.png", size = 150 }: Props) => {
  const [index, setIndex] = useState(0);
  const controls = useAnimation();

  useEffect(() => {
    const interval = setInterval(() => {
      setIndex((prev) => (prev + 1) % clipShapes.length);
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    controls.start({
      clipPath: clipShapes[index],
      transition: { duration: 1.5, ease: "easeInOut" },
    });
  }, [index, controls]);

  return (
    <motion.div
      animate={controls}
      className={cn(
        "rounded-full p-[5px]",
        "bg-gradient-to-br from-black via-yellow-800 to-yellow-600",
        "shadow-lg hover:shadow-yellow-500/30 transition-transform",
        "hover:scale-105",
      )}
      style={{ width: size, height: size }}
    >
      <div className="w-full h-full overflow-hidden rounded-full">
        <img
          src={src}
          alt="avatar"
          className="w-full h-full object-cover rounded-full"
        />
      </div>
    </motion.div>
  );
};

export default AnimatedAvatar;
