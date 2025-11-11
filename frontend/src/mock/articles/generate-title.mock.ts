import type {
  TitleGenerationRequest,
  TitleGenerationResponse,
} from "../../api/types";

export async function generateTitleMock(
  body: TitleGenerationRequest
): Promise<TitleGenerationResponse> {
  const base = body.input_text.trim();
  const edited = `${base} — edited (mock)`;
  return {
    edited_title: edited,
    seo_titles: [`${edited} v1`, `${edited} v2`, `${edited} v3`],
    raw_response: `Edited Title: ${edited}\nSEO Title 1: ${edited} v1\nSEO Title 2: ${edited} v2\nSEO Title 3: ${edited} v3`,
  };
}
