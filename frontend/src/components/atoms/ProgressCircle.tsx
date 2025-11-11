import { motion } from 'motion/react';
import { LucideIcon } from 'lucide-react';

interface ProgressCircleProps {
  progress: number;
  icon?: LucideIcon;
  size?: 'sm' | 'md' | 'lg';
  gradientId?: string;
  fromColor?: string;
  toColor?: string;
}

const sizeConfig = {
  sm: { viewBox: 80, radius: 35, strokeWidth: 6, iconSize: 'w-6 h-6', textSize: 'text-lg' },
  md: { viewBox: 120, radius: 50, strokeWidth: 8, iconSize: 'w-8 h-8', textSize: 'text-2xl' },
  lg: { viewBox: 160, radius: 70, strokeWidth: 10, iconSize: 'w-10 h-10', textSize: 'text-3xl' }
};

export function ProgressCircle({
  progress,
  icon: Icon,
  size = 'md',
  gradientId = 'gradient',
  fromColor = '#3b82f6',
  toColor = '#2563eb'
}: ProgressCircleProps) {
  const config = sizeConfig[size];
  const circumference = 2 * Math.PI * config.radius;
  
  return (
    <div className={`relative ${size === 'sm' ? 'w-20 h-20' : size === 'md' ? 'w-32 h-32' : 'w-40 h-40'}`}>
      <svg 
        className="transform -rotate-90" 
        viewBox={`0 0 ${config.viewBox} ${config.viewBox}`}
      >
        <circle
          cx={config.viewBox / 2}
          cy={config.viewBox / 2}
          r={config.radius}
          stroke="#e5e7eb"
          strokeWidth={config.strokeWidth}
          fill="none"
        />
        <motion.circle
          cx={config.viewBox / 2}
          cy={config.viewBox / 2}
          r={config.radius}
          stroke={`url(#${gradientId})`}
          strokeWidth={config.strokeWidth}
          fill="none"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - progress / 100)}
          transition={{ duration: 0.5, ease: "easeOut" }}
        />
        <defs>
          <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor={fromColor} />
            <stop offset="100%" stopColor={toColor} />
          </linearGradient>
        </defs>
      </svg>
      
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        {Icon && (
          <motion.div
            animate={{ scale: [1, 1.1, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            <Icon className={`${config.iconSize} text-gray-700 mb-2`} />
          </motion.div>
        )}
        <span className={`${config.textSize} font-bold text-gray-800`}>{progress}%</span>
      </div>
    </div>
  );
}