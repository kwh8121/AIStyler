import { Textarea } from '../ui/textarea';
import { Badge } from '../ui/badge';
import { IconButton } from '../atoms/IconButton';
import { LucideIcon, Check } from 'lucide-react';

interface TextEditorProps {
  value: string;
  onChange?: (value: string) => void;
  placeholder?: string;
  label: string;
  icon: LucideIcon;
  readonly?: boolean;
  isPreviewMode?: boolean;
  previewBadgeText?: string;
  actionButton?: {
    icon: LucideIcon;
    onClick: () => void;
    ariaLabel: string;
  };
  className?: string;
  editorClassName?: string;
  isCopied?: boolean;
}

export function TextEditor({
  value,
  onChange,
  placeholder,
  label,
  icon: Icon,
  readonly = false,
  isPreviewMode = false,
  previewBadgeText = "미리보기 모드",
  actionButton,
  className = "",
  editorClassName = "min-h-[350px] resize-none",
  isCopied = false
}: TextEditorProps) {
  return (
    <div className={`space-y-4 ${className}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Icon className="w-5 h-5 text-blue-600" />
          <label className="font-semibold text-gray-900">{label}</label>
        </div>
        {isPreviewMode && (
          <Badge variant="secondary" className="bg-blue-100 text-blue-700">
            {previewBadgeText}
          </Badge>
        )}
        {actionButton && (
          <div className="flex items-center gap-2">
            {isCopied && (
              <span className="text-sm text-green-600 font-medium animate-in fade-in duration-200">
                복사됨!
              </span>
            )}
            <IconButton
              icon={isCopied ? Check : actionButton.icon}
              onClick={actionButton.onClick}
              size="sm"
              className={`h-8 px-3 transition-colors duration-200 ${
                isCopied
                  ? 'text-green-600 bg-green-50 hover:bg-green-100'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
              aria-label={actionButton.ariaLabel}
            />
          </div>
        )}
      </div>
      <div className="relative">
        <Textarea
          value={value}
          onChange={(e) => onChange?.(e.target.value)}
          placeholder={placeholder}
          className={`${editorClassName} ${
            readonly
              ? 'bg-gradient-to-br from-purple-50 to-blue-50 border-purple-200/50'
              : 'bg-white border-gray-200 focus:border-blue-400 focus:ring-blue-400/20'
          } rounded-xl shadow-sm whitespace-pre-wrap`}
          disabled={isPreviewMode}
          readOnly={readonly}
          style={{ whiteSpace: 'pre-wrap' }}
        />
        {isPreviewMode && (
          <div className="absolute inset-0 bg-blue-50/30 rounded-xl pointer-events-none" />
        )}
      </div>
    </div>
  );
}