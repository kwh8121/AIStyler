import { Sparkles } from "lucide-react";

interface TextSelectorProps {
  displayOutputOptions: string[];
  displaySelectedIndex: number;
  onSelectOutput: (index: number, option: string) => void;
  showSelectionWarning: boolean;
  className?: string;
}

export function TextSelector({
  displayOutputOptions,
  displaySelectedIndex,
  onSelectOutput,
  showSelectionWarning,
  className = "",
}: TextSelectorProps) {
  return (
    <div className={`space-y-4 ${className}`}>
      <div className="flex items-center h-[29px]">
        <div className="flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-purple-600" />
          <label className="font-semibold text-gray-900">Text Output</label>
        </div>
      </div>
      <div className="min-h-[350px] bg-gradient-to-br from-purple-50 to-blue-50 border border-purple-200/50 rounded-xl shadow-sm p-4">
        <div className="space-y-3">
          {displayOutputOptions.map((option, index) => (
            <div
              key={index}
              className={`p-4 rounded-lg border-2 transition-all cursor-pointer ${
                displaySelectedIndex === index
                  ? "border-purple-400 bg-purple-50"
                  : "border-gray-200 bg-white hover:border-purple-200 hover:bg-purple-25"
              }`}
              onClick={() => onSelectOutput(index, option)}
            >
              <div className="flex items-start gap-3">
                <input
                  type="radio"
                  name="output-option"
                  checked={displaySelectedIndex === index}
                  className="mt-1 w-4 h-4 appearance-none border-2 border-gray-300 rounded-full bg-white checked:bg-purple-600 checked:border-purple-600 focus:ring-2 focus:ring-purple-500 focus:ring-offset-0"
                />

                <div className="flex-1">
                  <p className="text-gray-800 leading-relaxed">{option}</p>
                </div>
              </div>
            </div>
          ))}
          {showSelectionWarning && (
            <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-700">결과를 선택해주세요.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
