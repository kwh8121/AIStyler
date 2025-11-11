import { LucideIcon } from 'lucide-react';

interface HeaderSectionProps {
  icon: LucideIcon;
  title: string;
  subtitle?: string;
  iconClassName?: string;
  titleClassName?: string;
  subtitleClassName?: string;
  containerClassName?: string;
}

export function HeaderSection({
  icon: Icon,
  title,
  subtitle,
  iconClassName = "w-10 h-10 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl flex items-center justify-center",
  titleClassName = "text-2xl font-semibold bg-gradient-to-r from-gray-900 to-gray-600 bg-clip-text text-transparent",
  subtitleClassName = "text-sm text-gray-500",
  containerClassName = "flex items-center gap-3"
}: HeaderSectionProps) {
  return (
    <div className={containerClassName}>
      <div className={iconClassName}>
        <Icon className="w-5 h-5 text-white" />
      </div>
      <div>
        <h1 className={titleClassName}>
          {title}
        </h1>
        {subtitle && <p className={subtitleClassName}>{subtitle}</p>}
      </div>
    </div>
  );
}