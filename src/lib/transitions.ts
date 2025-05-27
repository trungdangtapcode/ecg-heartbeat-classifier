// Animation variants for common components
export const fadeIn = {
  initial: { opacity: 0 },
  animate: { opacity: 1 },
  exit: { opacity: 0 },
};

export const fadeInUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -20 },
};

export const fadeInDown = {
  initial: { opacity: 0, y: -20 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: 20 },
};

export const fadeInLeft = {
  initial: { opacity: 0, x: -20 },
  animate: { opacity: 1, x: 0 },
  exit: { opacity: 0, x: 20 },
};

export const fadeInRight = {
  initial: { opacity: 0, x: 20 },
  animate: { opacity: 1, x: 0 },
  exit: { opacity: 0, x: -20 },
};

export const staggerContainer = {
  animate: {
    transition: {
      staggerChildren: 0.1
    }
  }
};

export const slideIn = {
  initial: { opacity: 0, x: -20 },
  animate: { opacity: 1, x: 0 },
  exit: { opacity: 0, x: -20 },
};

export const scaleUp = {
  initial: { scale: 0.8, opacity: 0 },
  animate: { scale: 1, opacity: 1 },
  exit: { scale: 0.8, opacity: 0 },
};

// Transition settings
export const easing = [0.6, -0.05, 0.01, 0.99];

export const fadeInTransition = {
  duration: 0.6,
  ease: easing
};

export const staggerTransition = {
  duration: 0.3,
  ease: easing,
  staggerChildren: 0.1
};

export const springTransition = {
  type: "spring",
  stiffness: 100,
  damping: 15
};

// For scrolling animations
export const scrollVariants = {
  hidden: { opacity: 0, y: 30 },
  visible: { 
    opacity: 1, 
    y: 0,
    transition: {
      duration: 0.6,
      ease: easing
    }
  }
};

// Animation utilities
export const createDelayedVariants = (baseVariants: any, delayStep: number = 0.1) => {
  return (index: number) => ({
    ...baseVariants,
    animate: {
      ...baseVariants.animate,
      transition: {
        ...(baseVariants.animate?.transition || {}),
        delay: index * delayStep
      }
    }
  });
};
