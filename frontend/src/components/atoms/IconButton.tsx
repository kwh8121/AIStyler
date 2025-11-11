import * as React from 'react';
import { Button } from '../ui/button';
import { LucideIcon } from 'lucide-react';

interface IconButtonProps {
  icon: LucideIcon;
  onClick?: React.MouseEventHandler<HTMLButtonElement>;
  variant?: 'default' | 'destructive' | 'outline' | 'secondary' | 'ghost' | 'link';
  size?: 'default' | 'sm' | 'lg' | 'icon';
  className?: string;
  disabled?: boolean;
  loading?: boolean;
  loadingIcon?: LucideIcon;
  children?: React.ReactNode;
  'aria-label'?: string;
}

export function IconButton({
  icon: Icon,
  onClick,
  variant = 'ghost',
  size = 'sm',
  className = '',
  disabled = false,
  loading = false,
  loadingIcon: LoadingIcon,
  children,
  'aria-label': ariaLabel,
  ...props
}: IconButtonProps) {
  const DisplayIcon = loading && LoadingIcon ? LoadingIcon : Icon;
  
  return (
    <Button
      variant={variant}
      size={size}
      onClick={onClick}
      disabled={disabled || loading}
      className={className}
      aria-label={ariaLabel}
      {...props}
    >
      <DisplayIcon className={`${children ? 'mr-2' : ''} ${loading ? 'animate-spin' : ''} w-4 h-4`} />
      {children}
    </Button>
  );
}
