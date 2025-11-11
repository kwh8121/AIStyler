import { motion } from 'motion/react';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

const sizeClasses = {
  sm: 'w-8 h-8 border-2',
  md: 'w-12 h-12 border-3',
  lg: 'w-16 h-16 border-4'
};

export function LoadingSpinner({ size = 'md', className = '' }: LoadingSpinnerProps) {
  return (
    <motion.div
      className={`${sizeClasses[size]} border-gray-200 border-t-blue-500 rounded-full ${className}`}
      animate={{ rotate: 360 }}
      transition={{
        duration: 1,
        repeat: Infinity,
        ease: "linear"
      }}
    />
  );
}