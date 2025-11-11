import { Badge } from '../ui/badge';

export type StatusType = 'AI' | '번역' | '복원';

interface StatusBadgeProps {
  type: StatusType;
  className?: string;
}

const statusColors: Record<StatusType, string> = {
  'AI': 'bg-blue-100 text-blue-700',
  '번역': 'bg-amber-100 text-amber-700',
  '복원': 'bg-green-100 text-green-700'
};

export function StatusBadge({ type, className = '' }: StatusBadgeProps) {
  return (
    <Badge 
      variant="secondary" 
      className={`text-xs ${statusColors[type]} ${className}`}
    >
      {type}
    </Badge>
  );
}