import React, { useEffect, useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "../ui/dialog";
import { Button } from "../ui/button";
import { X, FileText, Calendar, Tag, Loader2 } from "lucide-react";
import { api } from "../../api/client";
import type { AppliedStyleGuide, StyleGuideOut } from "../../api/types";
import { mapBackendCategoryToDisplay } from "./StyleGuidePopup";

interface StyleGuideDetailModalProps {
  isOpen: boolean;
  onClose: () => void;
  appliedStyleGuide: AppliedStyleGuide | null;
  beforeText?: string; // 적용 전 텍스트 (히스토리에서)
}

export function StyleGuideDetailModal({
  isOpen,
  onClose,
  appliedStyleGuide,
  beforeText,
}: StyleGuideDetailModalProps) {
  const [styleGuideDetail, setStyleGuideDetail] =
    useState<StyleGuideOut | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadStyleGuide() {
      if (!appliedStyleGuide || !isOpen) {
        setStyleGuideDetail(null);
        return;
      }

      try {
        setIsLoading(true);
        setError(null);
        const detail = await api.getStyleGuide(appliedStyleGuide.style_id);
        setStyleGuideDetail(detail);
      } catch (err) {
        console.error("Failed to load style guide:", err);
        setError("스타일 가이드를 불러올 수 없습니다.");
      } finally {
        setIsLoading(false);
      }
    }

    loadStyleGuide();
  }, [appliedStyleGuide, isOpen]);

  const formatDate = (dateStr: string) => {
    try {
      return new Date(dateStr).toLocaleString("ko-KR", {
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return dateStr;
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[80vh] overflow-hidden flex flex-col">
        <DialogHeader className="pb-4 border-b">
          <DialogTitle className="flex items-center gap-2">
            <FileText className="w-5 h-5 text-blue-600" />
            스타일 가이드 상세 정보
          </DialogTitle>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto space-y-6">
          {isLoading && (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="w-6 h-6 animate-spin text-blue-600" />
              <span className="ml-2 text-gray-600">
                스타일 가이드를 불러오는 중...
              </span>
            </div>
          )}

          {error && (
            <div className="text-center py-8">
              <div className="text-red-600 font-medium mb-4">{error}</div>
              <Button onClick={onClose} variant="outline">
                닫기
              </Button>
            </div>
          )}

          {!isLoading && !error && appliedStyleGuide && (
            <>
              {/* 적용 정보 */}
              <div className="bg-blue-50 rounded-lg p-4">
                <h3 className="font-semibold text-blue-900 mb-3">적용 정보</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                  <div className="flex items-center gap-2">
                    <FileText className="w-4 h-4 text-blue-600" />
                    <span className="font-medium">이름:</span>
                    <span>{appliedStyleGuide.name}</span>
                    {appliedStyleGuide.number && (
                      <span className="text-blue-600">
                        #{appliedStyleGuide.number}
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <Tag className="w-4 h-4 text-blue-600" />
                    <span className="font-medium">카테고리:</span>
                    <span>
                      {mapBackendCategoryToDisplay(appliedStyleGuide.category)}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Calendar className="w-4 h-4 text-blue-600" />
                    <span className="font-medium">적용 시간:</span>
                    <span>{formatDate(appliedStyleGuide.applied_at)}</span>
                  </div>
                  {appliedStyleGuide.note && (
                    <div className="flex items-start gap-2 md:col-span-2">
                      <span className="font-medium">메모:</span>
                      <span className="text-gray-700">
                        {appliedStyleGuide.note}
                      </span>
                    </div>
                  )}
                </div>
              </div>

              {/* 스타일 가이드 상세 정보 */}
              {styleGuideDetail && (
                <div className="space-y-4">
                  <h3 className="font-semibold text-gray-900">
                    스타일 가이드 상세
                  </h3>

                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm mb-4">
                      <div>
                        <span className="font-medium text-gray-700">ID:</span>
                        <span className="ml-2">{styleGuideDetail.id}</span>
                      </div>
                      <div>
                        <span className="font-medium text-gray-700">버전:</span>
                        <span className="ml-2">
                          v{styleGuideDetail.version}
                        </span>
                      </div>
                      <div>
                        <span className="font-medium text-gray-700">
                          생성일:
                        </span>
                        <span className="ml-2">
                          {formatDate(styleGuideDetail.created_at)}
                        </span>
                      </div>
                      <div>
                        <span className="font-medium text-gray-700">
                          수정일:
                        </span>
                        <span className="ml-2">
                          {formatDate(styleGuideDetail.updated_at)}
                        </span>
                      </div>
                    </div>

                    <div>
                      <h4 className="font-medium text-gray-700 mb-2">
                        가이드 내용:
                      </h4>
                      <div className="bg-white rounded border p-3 text-sm leading-relaxed whitespace-pre-wrap">
                        {styleGuideDetail.docs}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* 적용 대상 텍스트 */}
              {beforeText && (
                <div className="space-y-4">
                  <h3 className="font-semibold text-gray-900">적용된 텍스트</h3>
                  <div className="bg-amber-50 rounded-lg p-4">
                    <h4 className="font-medium text-amber-800 mb-2">
                      원본 텍스트:
                    </h4>
                    <div className="bg-white rounded border p-3 text-sm leading-relaxed">
                      {beforeText}
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {!error && (
          <div className="flex justify-end pt-4 border-t">
            <Button onClick={onClose} variant="outline">
              닫기
            </Button>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
