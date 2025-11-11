import { useEffect, useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "../ui/dialog";
import { Badge } from "../ui/badge";
import { ExternalLink } from "lucide-react";
import { StyleGuideCard, StyleGuideItem } from "../molecules/StyleGuideCard";
import { api } from "../../api/client";

interface StyleGuidePopupProps {
  onClose?: () => void;
}

// 백엔드 카테고리를 프론트엔드 표시용 카테고리로 변환
export function mapBackendCategoryToDisplay(backendCategory: string): string {
  const normalized = backendCategory.toUpperCase();

  const categoryMap: Record<string, string> = {
    "TITLE": "Headline",
    "BODY": "Content",
    "CAPTION": "Caption",
    "SEO_TITLE": "SEO Title",
    "SEO": "SEO Title",
    "ARTICLES_TRANSLATOR": "Content",
  };

  return categoryMap[normalized] || backendCategory;
}

export function StyleGuidePopup({ onClose }: StyleGuidePopupProps) {
  const [isOpen, setIsOpen] = useState(true);
  const [items, setItems] = useState<StyleGuideItem[]>([]);

  const currentDateTime = new Date().toLocaleString("ko-KR", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  const handleOpenChange = (open: boolean) => {
    setIsOpen(open);
    if (!open && onClose) {
      onClose();
    }
  };

  useEffect(() => {
    (async () => {
      try {
        const list = await api.listStyleGuides({ limit: 50 });
        const mapped: StyleGuideItem[] = list.map((sg, idx) => ({
          category: mapBackendCategoryToDisplay(sg.category),
          guideNumber: idx + 1,
          title: sg.name,
          description: sg.docs,
          examples: {},
        }));
        setItems(mapped);
      } catch (_) {
        // keep default
      }
    })();
  }, []);

  return (
    <Dialog open={isOpen} onOpenChange={handleOpenChange}>
      <DialogContent
        className="!block max-w-none w-[99vw] h-[95vh] !p-0 bg-white"
        style={{
          width: "99vw",
          height: "95vh",
          maxWidth: "none",
          display: "flex",
          flexDirection: "column",
          padding: 0
        }}
      >
        <DialogHeader className="flex-shrink-0 p-6 pb-4 border-b">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <DialogTitle className="flex items-center gap-2">
                적용된 스타일가이드
                <ExternalLink className="h-4 w-4 text-muted-foreground" />
              </DialogTitle>
              <Badge variant="secondary" className="px-3 py-1">
                총 {items.length}개 가이드 적용
              </Badge>
            </div>
            <span className="text-sm text-muted-foreground flex-shrink-0">
              {currentDateTime}
            </span>
          </div>
        </DialogHeader>

        <div className="flex-1 min-h-0 overflow-y-auto p-6">
          <div className="space-y-6">
            {items.map((item) => (
              <StyleGuideCard key={item.guideNumber} item={item} />
            ))}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
