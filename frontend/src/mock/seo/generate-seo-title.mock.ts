import type {
  SeoTitleGenerationRequest,
  SeoTitleGenerationResponse,
} from "../../api/types";

export async function generateSeoTitleMock(
  body: SeoTitleGenerationRequest
): Promise<SeoTitleGenerationResponse> {
  return {
    seo_titles: [
      "President Yoon apology debate sparks opposition 'persecution' accusations in South Korea",
      "South Korea opposition accuses Yoon of 'persecution' after apology debate handshake",
      "Yoon apology debate handshake fuels opposition 'persecution' accusations in South Korea",
    ],
  };
}
