import React, { useRef } from "react";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import { Database, Download, Upload, Trash2, HardDrive } from "lucide-react";

interface LocalModeIndicatorProps {
  historyCount: number;
  storageUsedMB: string;
  onExportHistory: () => void;
  onImportHistory: (file: File) => void;
  onClearHistory: () => void;
  onSwitchToServer?: () => void;
}

export function LocalModeIndicator({
  historyCount,
  storageUsedMB,
  onExportHistory,
  onImportHistory,
  onClearHistory,
  onSwitchToServer,
}: LocalModeIndicatorProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImportClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onImportHistory(file);
      // Reset input
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  return (
    <div className="bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-lg p-3 mb-4">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <HardDrive className="w-4 h-4 text-green-600" />
            <span className="font-medium text-green-800">로컬 모드</span>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="secondary" className="bg-green-100 text-green-700">
              {historyCount}/100 저장됨
            </Badge>
            <Badge variant="outline" className="border-green-200 text-green-700">
              {storageUsedMB} MB 사용
            </Badge>
          </div>
        </div>

        <div className="flex gap-2">
          <Button
            size="sm"
            variant="ghost"
            onClick={onExportHistory}
            className="text-green-700 hover:text-green-900 hover:bg-green-100"
          >
            <Download className="w-4 h-4 mr-1" />
            내보내기
          </Button>

          <Button
            size="sm"
            variant="ghost"
            onClick={handleImportClick}
            className="text-green-700 hover:text-green-900 hover:bg-green-100"
          >
            <Upload className="w-4 h-4 mr-1" />
            가져오기
          </Button>

          <input
            ref={fileInputRef}
            type="file"
            accept=".json"
            onChange={handleFileChange}
            className="hidden"
          />

          {historyCount > 0 && (
            <Button
              size="sm"
              variant="ghost"
              onClick={onClearHistory}
              className="text-red-600 hover:text-red-700 hover:bg-red-50"
            >
              <Trash2 className="w-4 h-4 mr-1" />
              전체 삭제
            </Button>
          )}

          {onSwitchToServer && (
            <Button
              size="sm"
              variant="ghost"
              onClick={onSwitchToServer}
              className="text-blue-600 hover:text-blue-700 hover:bg-blue-50"
            >
              <Database className="w-4 h-4 mr-1" />
              서버 모드
            </Button>
          )}
        </div>
      </div>

      <div className="mt-2 text-xs text-green-600">
        모든 데이터는 브라우저에 저장되며, 다른 기기와 동기화되지 않습니다. 정기적으로 백업하시기 바랍니다.
      </div>
    </div>
  );
}