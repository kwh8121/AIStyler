import React, { useState } from "react";
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "../ui/sheet";
import { Button } from "../ui/button";
import { Clock, RotateCcw } from "lucide-react";
import { HistoryCard, HistoryItem } from "../molecules/HistoryCard";
import { HeaderSection } from "../molecules/HeaderSection";
import { AppliedStyleGuidesModal } from "./AppliedStyleGuidesModal";

interface HistorySidebarProps {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  historyItems: HistoryItem[];
  selectedHistoryId: string | null;
  onHistorySelect: (item: HistoryItem) => void;
  onRestore: () => void;
  onShowStyleGuide?: () => void;
}

export function HistorySidebar({
  isOpen,
  onOpenChange,
  historyItems,
  selectedHistoryId,
  onHistorySelect,
  onRestore,
  onShowStyleGuide,
}: HistorySidebarProps) {
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [selectedHistoryItem, setSelectedHistoryItem] = useState<HistoryItem | null>(null);
  const [showAppliedStyleGuides, setShowAppliedStyleGuides] = useState(false);

  const handleMoreOptionsClick = async (item: HistoryItem) => {
    try {
      setLoadingId(item.id);
      // 해당 히스토리 아이템의 적용된 스타일 가이드 목록을 보여줌
      setSelectedHistoryItem(item);
      setShowAppliedStyleGuides(true);
    } finally {
      setLoadingId(null);
    }
  };
  return (
    <Sheet open={isOpen} onOpenChange={onOpenChange}>
      <SheetContent className="w-[350px] sm:w-[450px] bg-white/95 backdrop-blur-sm border-l border-gray-200/50 flex flex-col p-0">
        <SheetHeader className="flex flex-row items-center justify-between p-6 pb-4 border-b border-gray-100 flex-shrink-0">
          <HeaderSection
            icon={Clock}
            title="History"
            iconClassName="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg flex items-center justify-center"
            titleClassName="text-xl"
            subtitle=""
          />
          {selectedHistoryId && (
            <Button
              variant="outline"
              size="sm"
              onClick={onRestore}
              className="bg-green-50 border-green-200 text-green-700 hover:bg-green-100"
            >
              <RotateCcw className="w-4 h-4 mr-1" />
              복원
            </Button>
          )}
        </SheetHeader>

        <div className="flex-1 overflow-y-auto px-6 py-3">
          {historyItems.length > 0 ? (
            <div className="space-y-3">
              {historyItems.map((item) => (
                <div key={item.id}>
                  <HistoryCard
                    item={item}
                    isSelected={selectedHistoryId === item.id}
                    onSelect={onHistorySelect}
                    onMoreOptions={
                      item.type === "AI"
                        ? () => handleMoreOptionsClick(item)
                        : undefined
                    }
                    isLoadingMore={loadingId === item.id}
                  />
                </div>
              ))}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-full py-12 text-center">
              <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mb-4">
                <Clock className="w-8 h-8 text-gray-400" />
              </div>
              <p className="text-gray-500 font-medium">아직 이력이 없습니다</p>
              <p className="text-sm text-gray-400">
                AI Styler를 실행하면 이력이 저장됩니다
              </p>
            </div>
          )}
        </div>

        {/* 적용된 스타일 가이드 목록 모달 */}
        <AppliedStyleGuidesModal
          isOpen={showAppliedStyleGuides}
          onClose={() => {
            setShowAppliedStyleGuides(false);
            setSelectedHistoryItem(null);
          }}
          appliedStyles={selectedHistoryItem?.appliedStyles || []}
          beforeText={selectedHistoryItem?.inputText}
          afterText={selectedHistoryItem?.outputText}
          historyDate={selectedHistoryItem?.date}
        />
      </SheetContent>
    </Sheet>
  );
}
