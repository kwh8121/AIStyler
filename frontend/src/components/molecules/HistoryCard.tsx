import React from "react";
import { StatusBadge, StatusType } from "../atoms/StatusBadge";
import { IconButton } from "../atoms/IconButton";
import { Tooltip, TooltipContent, TooltipTrigger } from "../ui/tooltip";
import { MoreHorizontal, Loader2 } from "lucide-react";
import type { AppliedStyleGuide } from "../../api/types";

export interface HistoryItem {
  id: string;
  date: string;
  inputText: string;
  outputText: string;
  type: StatusType;
  appliedStyles?: AppliedStyleGuide[];
}

interface HistoryCardProps {
  item: HistoryItem;
  isSelected?: boolean;
  onSelect?: (item: HistoryItem) => void;
  onMoreOptions?: () => void;
  isLoadingMore?: boolean;
}

export function HistoryCard({
  item,
  isSelected = false,
  onSelect,
  onMoreOptions,
  isLoadingMore = false,
}: HistoryCardProps) {
  return (
    <div
      className={`p-4 rounded-xl cursor-pointer transition-all duration-200 ${
        isSelected
          ? "bg-gradient-to-r from-blue-50 to-indigo-50 border-2 border-blue-200 shadow-md"
          : "bg-white/60 border border-gray-200/50 hover:bg-white/80 hover:shadow-sm"
      }`}
      onClick={() => onSelect?.(item)}
    >
      <div className="flex items-center justify-between">
        <div className="flex-1 space-y-2">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-gray-800">
              {item.date}
            </span>
            <StatusBadge type={item.type} />
          </div>
        </div>
        {item.type === "AI" &&
          onMoreOptions &&
          (isLoadingMore ? (
            <Loader2 className="h-8 w-8 text-gray-400 animate-spin" />
          ) : (
            <Tooltip>
              <TooltipTrigger asChild>
                <IconButton
                  icon={MoreHorizontal}
                  size="icon"
                  className="h-8 w-8 text-gray-400 hover:text-gray-600"
                  onClick={(event) => {
                    event.stopPropagation();
                    onMoreOptions();
                  }}
                  aria-label="스타일가이드 보기"
                  title="스타일가이드"
                />
              </TooltipTrigger>
              <TooltipContent side="bottom" align="center" className="text-xs">
                스타일가이드
              </TooltipContent>
            </Tooltip>
          ))}
      </div>
    </div>
  );
}
