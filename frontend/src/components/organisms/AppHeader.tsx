import { Button } from "../ui/button";
import { Sparkles, Clock, Loader2 } from "lucide-react";
import React from "react";
import { HeaderSection } from "../molecules/HeaderSection";

interface AppHeaderProps {
  onHistoryClick: () => void | Promise<void>;
  isHistoryLoading?: boolean;
}

export function AppHeader({
  onHistoryClick,
  isHistoryLoading = false,
}: AppHeaderProps) {
  return (
    <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-lg border border-white/20 mb-6">
      <div className="flex items-center justify-between p-6">
        <HeaderSection
          icon={Sparkles}
          title="AI Styler"
          subtitle="텍스트 검수 및 스타일링 도구"
        />
        <Button
          variant="outline"
          onClick={onHistoryClick}
          disabled={isHistoryLoading}
          className="bg-white/50 border-gray-200/50 hover:bg-gray-50/80 backdrop-blur-sm shadow-sm transition-colors duration-200"
        >
          {isHistoryLoading ? (
            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
          ) : (
            <Clock className="w-4 h-4 mr-2" />
          )}
          {"History"}
        </Button>
      </div>
    </div>
  );
}
