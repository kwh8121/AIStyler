import React from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "../ui/dialog";
import { Button } from "../ui/button";
import { X, FileText } from "lucide-react";
import { StyleGuideCard, StyleGuideItem } from "../molecules/StyleGuideCard";
import type { AppliedStyleGuide } from "../../api/types";
import { mapBackendCategoryToDisplay } from "./StyleGuidePopup";

interface AppliedStyleGuidesModalProps {
  isOpen: boolean;
  onClose: () => void;
  appliedStyles: AppliedStyleGuide[];
  beforeText?: string;
  afterText?: string;
  historyDate?: string;
}

export function AppliedStyleGuidesModal({
  isOpen,
  onClose,
  appliedStyles,
  beforeText,
  afterText,
  historyDate,
}: AppliedStyleGuidesModalProps) {
  // AppliedStyleGuide를 StyleGuideItem으로 변환
  // 각 스타일 가이드의 개별 before_text/after_text를 사용하거나, 없으면 전체 텍스트 사용
  const styleGuideItems: StyleGuideItem[] = appliedStyles.map((style, idx) => ({
    category: mapBackendCategoryToDisplay(style.category),
    guideNumber: style.number || idx + 1,
    title: style.name,
    description: style.docs || "",
    examples: {
      // 개별 문장 교정 정보가 있으면 사용, 없으면 전체 텍스트 사용
      incorrect: style.before_text
        ? [style.before_text]
        : beforeText
        ? [beforeText]
        : undefined,
      correct: style.after_text
        ? [style.after_text]
        : afterText
        ? [afterText]
        : undefined,
    },
  }));

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent
        className="!block max-w-none w-[99vw] h-[90vh] !p-0 bg-white rounded-lg [&>button]:hidden"
        style={{
          width: "99vw",
          height: "90vh",
          maxWidth: "none",
          display: "flex",
          flexDirection: "column",
          padding: 0,
        }}
      >
        <div className="flex flex-col h-full overflow-hidden">
          {/* 헤더 - 고정 */}
          <div className="flex-shrink-0 p-6 pb-4 border-b bg-white">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <DialogTitle className="flex items-center gap-2">
                  <FileText className="w-5 h-5 text-blue-600" />
                  적용된 스타일 가이드
                  {historyDate && (
                    <span className="text-sm text-gray-500 font-normal">
                      ({historyDate})
                    </span>
                  )}
                </DialogTitle>
                <span className="text-sm text-muted-foreground">
                  총 {styleGuideItems.length}개 가이드 적용
                </span>
              </div>
              <Button
                variant="ghost"
                size="icon"
                onClick={onClose}
                className="h-8 w-8"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {/* 스크롤 영역 */}
          <div className="flex-1 min-h-0 overflow-y-auto p-6">
            {styleGuideItems.length === 0 ? (
              <div className="text-center py-8">
                <FileText className="w-12 h-12 text-gray-300 mx-auto mb-4" />
                <p className="text-gray-500">
                  적용된 스타일 가이드가 없습니다.
                </p>
              </div>
            ) : (
              <div className="space-y-6">
                {styleGuideItems.map((item) => (
                  <StyleGuideCard key={item.guideNumber} item={item} />
                ))}
              </div>
            )}

            {/* 전체 텍스트 미리보기 - 개별 문장 교정이 없는 경우에만 표시 */}
            {beforeText && !appliedStyles.some((s) => s.before_text) && (
              <div className="mt-6 border-t pt-4">
                <h4 className="font-medium text-gray-900 mb-2">
                  전체 원본 텍스트
                </h4>
                <div className="bg-gray-50 rounded-lg p-3 text-sm text-gray-700">
                  {beforeText}
                </div>
              </div>
            )}
            {afterText && !appliedStyles.some((s) => s.after_text) && (
              <div className="mt-2">
                <h4 className="font-medium text-gray-900 mb-2">
                  전체 교정 텍스트
                </h4>
                <div className="bg-blue-50 rounded-lg p-3 text-sm text-gray-700">
                  {afterText}
                </div>
              </div>
            )}
          </div>

          {/* 푸터 - 고정 */}
          <div className="flex-shrink-0 p-6 pt-4 border-t bg-white">
            <div className="flex justify-end">
              <Button onClick={onClose} variant="outline">
                닫기
              </Button>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
