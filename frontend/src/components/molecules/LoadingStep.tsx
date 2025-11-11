import { motion } from 'motion/react';
import { CheckCircle } from 'lucide-react';
import { LucideIcon } from 'lucide-react';

export interface LoadingStepData {
  name: string;
  icon: LucideIcon;
  description: string;
  progress: number;
  color: string;
}

interface LoadingStepProps {
  step: LoadingStepData;
  index: number;
  currentStep: number;
  isTimeline?: boolean;
}

export function LoadingStep({ step, index, currentStep, isTimeline = false }: LoadingStepProps) {
  const isCompleted = index < currentStep;
  const isActive = index === currentStep;
  
  const stepIndicatorClass = isCompleted 
    ? 'bg-green-100 text-green-600' 
    : isActive 
    ? `bg-gradient-to-r ${step.color} text-white`
    : 'bg-gray-100 text-gray-400';

  if (!isTimeline) {
    return (
      <motion.div
        key={index}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <h3 className="text-xl font-semibold text-gray-800 mb-1">
          {step.name}
        </h3>
        <p className="text-gray-600">{step.description}</p>
      </motion.div>
    );
  }

  return (
    <div className="flex items-center">
      <div className="flex items-center flex-1">
        <motion.div
          className={`w-10 h-10 rounded-full flex items-center justify-center mr-4 ${stepIndicatorClass}`}
          animate={{
            scale: isActive ? [1, 1.05, 1] : 1
          }}
          transition={{ duration: 1, repeat: isActive ? Infinity : 0 }}
        >
          {isCompleted ? (
            <CheckCircle className="w-5 h-5" />
          ) : (
            <step.icon className="w-5 h-5" />
          )}
        </motion.div>

        <div className="flex-1">
          <div className="flex justify-between items-center">
            <span className={`font-medium ${
              index <= currentStep ? 'text-gray-800' : 'text-gray-400'
            }`}>
              {step.name}
            </span>
            {isCompleted && (
              <span className="text-green-600 text-sm font-medium">완료</span>
            )}
            {isActive && (
              <span className="text-blue-600 text-sm font-medium">진행 중</span>
            )}
          </div>
          <p className={`text-sm mt-1 ${
            index <= currentStep ? 'text-gray-600' : 'text-gray-400'
          }`}>
            {step.description}
          </p>
        </div>
      </div>
    </div>
  );
}