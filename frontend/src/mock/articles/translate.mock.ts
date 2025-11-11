import type { TranslationRequest, TranslationResponse } from "../../api/types";

export async function translateMock(
  body: TranslationRequest
): Promise<TranslationResponse> {
  // Simple mock that reverses language fields and adds a suffix
  return {
    translated_text: `[MOCK TRANSLATED to ${body.target_lang}] ${body.text}`,
    source_lang: body.source_lang || "KO",
    target_lang: body.target_lang,
    history_id: 1,
    version: 1,
  };
}
