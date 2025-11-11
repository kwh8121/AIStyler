import type { StyleGuideOut } from "../../api/types";

export async function listStyleGuidesMock({
  category,
  limit = 10,
}: { category?: "TITLE" | "BODY" | "CAPTION"; limit?: number } = {}): Promise<
  StyleGuideOut[]
> {
  const now = new Date().toISOString();
  const items: StyleGuideOut[] = [
    {
      id: 1,
      name: "제목 작성 가이드",
      category: "TITLE",
      docs: "제목은 명확하고 간결하게 작성해야 합니다.",
      version: 1,
      created_at: now,
      updated_at: now,
      deleted_at: null,
    },
    {
      id: 2,
      name: "본문 작성 가이드",
      category: "BODY",
      docs: "본문은 사실에 기반하고 중립적인 어조를 유지합니다.",
      version: 1,
      created_at: now,
      updated_at: now,
      deleted_at: null,
    },
    {
      id: 3,
      name: "캡션 가이드",
      category: "CAPTION",
      docs: "사진 설명은 1~2문장으로 핵심 정보를 제공합니다.",
      version: 1,
      created_at: now,
      updated_at: now,
      deleted_at: null,
    },
  ];
  return items
    .filter((i) => !category || i.category === category)
    .slice(0, limit);
}
