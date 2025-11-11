import type { CMSSaveRequest, CMSSaveResponse } from "../../api/types";

/**
 * Mock API for CMS save
 * Simulates saving article content to CMS
 */
export async function saveToCMSMock(
  body: CMSSaveRequest
): Promise<CMSSaveResponse> {
  // Simulate network delay
  await new Promise((resolve) => setTimeout(resolve, 500));

  console.log("📝 [Mock] Saving to CMS:", body);

  // Simulate successful save
  return {
    article_id: body.article_id,
    news_key: body.article_id, // Use article_id as news_key
    category: body.category,
    success: true,
  };
}
