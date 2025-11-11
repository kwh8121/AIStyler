import * as React from "react";
import { Button } from "../ui/button";
import { Textarea } from "../ui/textarea";
import { TextEditor } from "../molecules/TextEditor";
import {
  Edit3,
  Sparkles,
  Copy,
  Send,
  Play,
  FileText,
  Eye,
  Loader2,
} from "lucide-react";

interface TextEditorSectionProps {
  category: string;
  articleId?: string;
  inputText: string;
  outputText: string;
  isPreviewMode: boolean;
  currentTab: "styler" | "translator";
  additionalInstructions: string;
  isStylerLoading?: boolean;
  isTranslating?: boolean;
  isApplying?: boolean;
  isCopied?: boolean;
  onInputChange: (value: string) => void;
  onOutputChange?: (value: string) => void;
  onAdditionalInstructionsChange: (value: string) => void;
  onTabChange: (tab: "styler" | "translator") => void;
  onCancelPreview: () => void;
  onCopyResult: () => void;
  onApplyResult: () => void;
  onStyleText: () => void;
  onTranslate: () => void;
}

export function TextEditorSection({
  category,
  articleId,
  inputText,
  outputText,
  isPreviewMode,
  currentTab,
  additionalInstructions,
  isStylerLoading,
  isTranslating,
  isApplying,
  isCopied,
  onInputChange,
  onAdditionalInstructionsChange,
  onTabChange,
  onCancelPreview,
  onCopyResult,
  onApplyResult,
  onStyleText,
  onTranslate,
}: TextEditorSectionProps) {
  const [showAdditionalPromptPlaceholder, setShowAdditionalPromptPlaceholder] =
    React.useState(true);

  return (
    <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-lg border border-white/20">
      {/* Content Tab */}
      <div className="p-6 border-b border-gray-100">
        {/* 첫 번째 행: 카테고리 배지 + 탭 */}
        <div className="flex items-center gap-6">
          <div className="inline-flex items-center px-4 py-2 bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-full border border-blue-200/30">
            <FileText className="w-4 h-4 mr-2 text-blue-600" />
            <span className="text-sm font-medium text-blue-700">
              {category}
            </span>
          </div>
          {!isPreviewMode && (
            <div className="flex">
              <button
                onClick={() => onTabChange("styler")}
                className={`px-4 py-2 text-sm font-medium transition-colors ${
                  currentTab === "styler"
                    ? "text-blue-700 border-b-2 border-blue-600"
                    : "text-gray-500 hover:text-gray-700"
                }`}
              >
                Styler
              </button>
              <button
                onClick={() => onTabChange("translator")}
                className={`px-4 py-2 text-sm font-medium transition-colors ${
                  currentTab === "translator"
                    ? "text-blue-700 border-b-2 border-blue-600"
                    : "text-gray-500 hover:text-gray-700"
                }`}
              >
                Translator
              </button>
            </div>
          )}
        </div>

        {/* 두 번째 행: Article ID */}
        {articleId && (
          <div className="mt-3">
            <div
              className="bg-gradient-to-r from-amber-100 to-orange-100 border-2 border-orange-300 px-6 py-4 rounded-xl shadow-sm"
              style={{
                background: "linear-gradient(to right, #fef3c7, #ffedd5)",
                border: "2px solid #fdba74",
                borderRadius: "0.75rem",
                padding: "1rem 1.5rem",
                boxShadow: "0 1px 2px 0 rgb(0 0 0 / 0.05)",
              }}
            >
              <div
                className="flex items-center gap-3"
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "0.75rem",
                }}
              >
                <div
                  className="w-2 h-2 bg-orange-500 rounded-full shadow-sm"
                  style={{
                    width: "0.5rem",
                    height: "0.5rem",
                    backgroundColor: "#f97316",
                    borderRadius: "9999px",
                    boxShadow: "0 1px 2px 0 rgb(0 0 0 / 0.05)",
                  }}
                ></div>
                <span
                  className="font-medium text-orange-600"
                  style={{
                    fontWeight: "500",
                    color: "#ea580c",
                  }}
                >
                  기사 ID: {articleId}
                </span>
              </div>
            </div>
          </div>
        )}
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
            isPreviewMode={isPreviewMode}
            editorClassName="min-h-[350px] resize-none bg-white border-gray-200 focus:border-blue-400 focus:ring-blue-400/20 rounded-xl shadow-sm"
          />

          <TextEditor
            value={outputText}
            label="Text Output"
            icon={Sparkles}
            readonly
            actionButton={{
              icon: Copy,
              onClick: onCopyResult,
              ariaLabel: "Copy result",
            }}
            isCopied={isCopied}
          />
        </div>

        {/* Preview Mode Notice */}
        {isPreviewMode && (
          <div className="flex items-center justify-between p-4 mb-6 bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200/50 rounded-xl">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                <Eye className="w-4 h-4 text-blue-600" />
              </div>
              <div>
                <p className="font-medium text-blue-900">
                  이전 히스토리 미리보기 상태
                </p>
                <p className="text-sm text-blue-700">
                  선택한 이력의 내용을 확인하고 있습니다
                </p>
              </div>
            </div>
            <Button
              variant="ghost"
              onClick={onCancelPreview}
              className="text-blue-600 hover:text-blue-800 hover:bg-blue-100"
            >
              보기취소
            </Button>
          </div>
        )}

        {/* Instructions */}
        {!isPreviewMode && currentTab === "styler" && (
          <div className="p-6 mb-6 bg-gray-50/50 border border-gray-200/50 rounded-xl">
            <Textarea
              value={additionalInstructions}
              onChange={(e) => onAdditionalInstructionsChange(e.target.value)}
              placeholder={
                showAdditionalPromptPlaceholder
                  ? "이번 검수에만 포함할 추가 지침이 있는 경우 여기에 입력하세요."
                  : ""
              }
              className="resize-none bg-transparent border-none focus:ring-0 focus:border-none text-center text-gray-600 placeholder:text-gray-400 py-1 px-0 leading-tight"
              style={{
                height: "30px",
                minHeight: "30px",
              }}
              onFocus={() => setShowAdditionalPromptPlaceholder(false)}
              onBlur={() => {
                if (additionalInstructions === "") {
                  setShowAdditionalPromptPlaceholder(true);
                }
              }}
              onInput={(e) => {
                const target = e.target as HTMLTextAreaElement;
                target.style.height = "30px";
                target.style.height = Math.max(30, target.scrollHeight) + "px";
              }}
            />
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-between items-center">
          <a
            href="https://aidesk.koreatimes.co.kr"
            target="_blank"
            rel="noopener noreferrer"
            className="h-9 cursor-pointer"
          >
            <img
              src="/Black_logo.png"
              alt="Multi AI Desk"
              // className="h-9 object-contain opacity-50"
              className="h-9 object-contain"
            />
          </a>
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
            {!isPreviewMode &&
              (currentTab === "styler" ? (
                <Button
                  onClick={onStyleText}
                  className="shadow-lg bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white"
                >
                  <Sparkles className="w-4 h-4 mr-2" />
                  AI Styler 실행
                </Button>
              ) : (
                <Button
                  onClick={onTranslate}
                  disabled={isTranslating}
                  className="shadow-lg bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white disabled:opacity-70"
                >
                  {isTranslating ? (
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  ) : (
                    <Play className="w-4 h-4 mr-2" />
                  )}
                  번역
                </Button>
              ))}
          </div>
        </div>
      </div>
    </div>
  );
}
