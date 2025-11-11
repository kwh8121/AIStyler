import type { NewsHistoryResponseItem, OperationType } from "../../api/types";

export async function listNewsHistoryMock(
  articleId: string,
  operationType?: OperationType
): Promise<NewsHistoryResponseItem[]> {
  const items: NewsHistoryResponseItem[] = [
    {
      history_id: 1,
      news_key: articleId,
      category: "TITLE",
      version: 1,
      before_text: "MOCK1 원본 제목",
      after_text: "MOCK1 처리된 제목 (mock)",
      operation_type: "TRANSLATION",
      source_lang: "KO",
      target_lang: "EN-US",
      created_at: new Date().toISOString(),
    },
    {
      history_id: 2,
      news_key: articleId,
      category: "BODY",
      version: 1,
      before_text: "MOCK2 원본 본문",
      after_text: "MOCK2 교정된 본문 (mock)",
      operation_type: "TRANSLATION_CORRECTION",
      source_lang: "KO",
      target_lang: "EN-US",
      created_at: new Date().toISOString(),
    },
  ];
  return operationType
    ? items.filter((i) => i.operation_type === operationType)
    : items;
}
