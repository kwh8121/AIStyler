import { Button } from "../ui/button";
import { TextEditor } from "../molecules/TextEditor";
import { TextSelector } from "../molecules/TextSelector";
import { Edit3, Sparkles, Send, FileText, Loader2 } from "lucide-react";

interface SeoTitleSectionProps {
  inputText: string;
  isExtracting: boolean;
  isApplying: boolean;
  onInputChange: (value: string) => void;
  outputOptions: string[];
  displaySelectedIndex: number;
  onSelectOutput: (index: number, option: string) => void;
  showSelectionWarning: boolean;
  onApplyResult: () => void;
  onExtractAgain: () => void;
}

export function SeoTitleSection({
  inputText,
  isExtracting,
  isApplying,
  onInputChange,
  outputOptions,
  displaySelectedIndex,
  onSelectOutput,
  showSelectionWarning,
  onApplyResult,
  onExtractAgain,
}: SeoTitleSectionProps) {
  return (
    <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-lg border border-white/20">
      {/* Content Tab */}
      <div className="p-6 border-b border-gray-100">
        {/* 첫 번째 행: SEO Title 배지 */}
        <div className="flex items-center gap-6">
          <div className="inline-flex items-center px-4 py-2 bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-full border border-blue-200/30">
            <FileText className="w-4 h-4 mr-2 text-blue-600" />
            <span className="text-sm font-medium text-blue-700">
              SEO Title
            </span>
          </div>
        </div>
      </div>

      <div className="p-6">
        {/* Text Input/Output Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          <TextEditor
            value={inputText}
            onChange={onInputChange}
            placeholder="텍스트를 입력하세요..."
            label="Text Input"
            icon={Edit3}
            isPreviewMode={false}
            editorClassName="min-h-[350px] resize-none bg-white border-gray-200 focus:border-blue-400 focus:ring-blue-400/20 rounded-xl shadow-sm"
          />

          <TextSelector
            displayOutputOptions={outputOptions}
            displaySelectedIndex={displaySelectedIndex}
            onSelectOutput={onSelectOutput}
            showSelectionWarning={showSelectionWarning}
          />
        </div>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-between items-center">
          <img
            src="/Black_logo.png"
            alt="Multi AI Desk"
            className="h-9 object-contain"
          />
          <div className="flex flex-col sm:flex-row gap-4">
            <Button
              variant="outline"
              onClick={onApplyResult}
              disabled={isApplying}
              className="bg-white border-gray-300 hover:bg-gray-50 shadow-sm transition-colors duration-200 disabled:opacity-70"
            >
              {isApplying ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Send className="w-4 h-4 mr-2" />
              )}
              결과 적용
            </Button>

            <Button
              onClick={onExtractAgain}
              className="shadow-lg bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white"
            >
              {isExtracting ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Sparkles className="w-4 h-4 mr-2" />
              )}
              다시 추출
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
