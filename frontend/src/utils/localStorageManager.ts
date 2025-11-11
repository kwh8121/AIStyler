import type { HistoryItem } from "../components/molecules/HistoryCard";
import type { ArticleCategory, AppliedStyleGuide } from "../api/types";

export interface LocalHistoryItem extends HistoryItem {
  localId: string;
  category: ArticleCategory;
  operationType: "CORRECTION" | "TRANSLATION" | "RESTORATION";
  version: number;
  createdAt: string;
  metadata?: {
    inputLength: number;
    outputLength: number;
    processingTime?: number;
  };
}

interface LocalStorageData {
  version: "1.0";
  histories: LocalHistoryItem[];
  lastUpdated: string;
  totalCount: number;
}

export class LocalStorageManager {
  private readonly STORAGE_KEY = "kt-styler-local-history";
  private readonly MAX_ITEMS = 100;
  private readonly MAX_STORAGE_SIZE = 2 * 1024 * 1024; // 2MB

  private getData(): LocalStorageData {
    try {
      const stored = localStorage.getItem(this.STORAGE_KEY);
      if (!stored) {
        return this.getEmptyData();
      }
      const data = JSON.parse(stored) as LocalStorageData;
      // 버전 체크 및 마이그레이션
      if (data.version !== "1.0") {
        return this.getEmptyData();
      }
      return data;
    } catch (error) {
      console.error("Failed to load local history:", error);
      return this.getEmptyData();
    }
  }

  private setData(data: LocalStorageData): void {
    try {
      data.lastUpdated = new Date().toISOString();
      data.totalCount = data.histories.length;
      localStorage.setItem(this.STORAGE_KEY, JSON.stringify(data));
    } catch (error) {
      console.error("Failed to save local history:", error);
      // 스토리지 용량 초과 시 오래된 항목 제거 후 재시도
      if (error instanceof DOMException && error.name === "QuotaExceededError") {
        this.optimizeStorage(data);
        try {
          localStorage.setItem(this.STORAGE_KEY, JSON.stringify(data));
        } catch (retryError) {
          console.error("Failed to save after optimization:", retryError);
        }
      }
    }
  }

  private getEmptyData(): LocalStorageData {
    return {
      version: "1.0",
      histories: [],
      lastUpdated: new Date().toISOString(),
      totalCount: 0,
    };
  }

  // 히스토리 저장
  saveHistory(item: Omit<LocalHistoryItem, "localId" | "version" | "createdAt">): void {
    const data = this.getData();

    const newItem: LocalHistoryItem = {
      ...item,
      localId: `local-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      version: 1,
      createdAt: new Date().toISOString(),
      metadata: {
        inputLength: item.inputText.length,
        outputLength: item.outputText.length,
      },
    };

    // 새 항목을 맨 앞에 추가
    data.histories.unshift(newItem);

    // 100개 초과 시 오래된 항목 제거
    if (data.histories.length > this.MAX_ITEMS) {
      data.histories = data.histories.slice(0, this.MAX_ITEMS);
    }

    // 크기 체크 및 필요시 압축
    this.optimizeStorage(data);
    this.setData(data);
  }

  // 히스토리 목록 가져오기
  getHistories(): LocalHistoryItem[] {
    const data = this.getData();
    return data.histories;
  }

  // 특정 히스토리 가져오기
  getHistory(localId: string): LocalHistoryItem | undefined {
    const data = this.getData();
    return data.histories.find((h) => h.localId === localId || h.id === localId);
  }

  // 히스토리 삭제
  deleteHistory(localId: string): void {
    const data = this.getData();
    data.histories = data.histories.filter((h) => h.localId !== localId && h.id !== localId);
    this.setData(data);
  }

  // 모든 히스토리 삭제
  clearAllHistories(): void {
    this.setData(this.getEmptyData());
  }

  // 스토리지 최적화
  private optimizeStorage(data: LocalStorageData): void {
    const size = new Blob([JSON.stringify(data)]).size;
    if (size > this.MAX_STORAGE_SIZE) {
      // 20개 이후의 항목은 텍스트를 압축
      data.histories.forEach((item, idx) => {
        if (idx > 20) {
          // 긴 텍스트를 요약으로 대체
          if (item.inputText.length > 500) {
            item.inputText = item.inputText.substring(0, 497) + "...";
          }
          if (item.outputText.length > 500) {
            item.outputText = item.outputText.substring(0, 497) + "...";
          }
          // appliedStyles의 docs도 압축
          if (item.appliedStyles) {
            item.appliedStyles.forEach((style) => {
              if (style.docs && style.docs.length > 100) {
                style.docs = style.docs.substring(0, 97) + "...";
              }
            });
          }
        }
      });
    }
  }

  // 내보내기
  exportToJSON(): string {
    const data = this.getData();
    return JSON.stringify(data, null, 2);
  }

  // 가져오기
  importFromJSON(jsonString: string): boolean {
    try {
      const imported = JSON.parse(jsonString) as LocalStorageData;

      // 유효성 검사
      if (imported.version !== "1.0" || !Array.isArray(imported.histories)) {
        throw new Error("Invalid data format");
      }

      // 기존 데이터와 병합
      const current = this.getData();
      const merged = {
        ...imported,
        histories: [...imported.histories, ...current.histories],
      };

      // 중복 제거 (localId 기준)
      const uniqueMap = new Map<string, LocalHistoryItem>();
      merged.histories.forEach((item) => {
        if (!uniqueMap.has(item.localId)) {
          uniqueMap.set(item.localId, item);
        }
      });

      merged.histories = Array.from(uniqueMap.values())
        .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())
        .slice(0, this.MAX_ITEMS);

      this.setData(merged);
      return true;
    } catch (error) {
      console.error("Failed to import data:", error);
      return false;
    }
  }

  // 통계 정보
  getStats(): {
    totalItems: number;
    storageUsed: number;
    storageUsedMB: string;
    oldestEntry?: string;
    newestEntry?: string;
    averageTextLength: number;
  } {
    const data = this.getData();
    const storageUsed = new Blob([JSON.stringify(data)]).size;

    return {
      totalItems: data.histories.length,
      storageUsed,
      storageUsedMB: (storageUsed / (1024 * 1024)).toFixed(2),
      oldestEntry: data.histories[data.histories.length - 1]?.createdAt,
      newestEntry: data.histories[0]?.createdAt,
      averageTextLength:
        data.histories.reduce((sum, h) => sum + (h.metadata?.inputLength || 0), 0) /
        Math.max(data.histories.length, 1),
    };
  }

  // 자동 백업 생성
  createBackup(): Blob {
    const data = this.exportToJSON();
    return new Blob([data], { type: "application/json" });
  }

  // 백업 다운로드 트리거
  downloadBackup(filename?: string): void {
    const blob = this.createBackup();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const date = new Date().toISOString().split("T")[0];
    a.href = url;
    a.download = filename || `kt-styler-backup-${date}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }
}

// 싱글톤 인스턴스
export const localStorageManager = new LocalStorageManager();