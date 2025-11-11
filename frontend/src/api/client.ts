import type {
  ArticleCorrectionRequest,
  TranslationRequest,
  TranslationResponse,
  TitleGenerationRequest,
  TitleGenerationResponse,
  NewsHistoryResponseItem,
  SeoTitleGenerationRequest,
  SeoTitleGenerationResponse,
  StyleGuideOut,
  SSEMessage,
  CMSGetResponse,
  CMSSaveRequest,
  CMSSaveResponse,
} from "./types";

const USE_MOCK = (import.meta as any).env?.VITE_USE_MOCK_API === "true";
//const API_BASE_URL: string =
//  (import.meta as any).env?.VITE_API_BASE_URL || "http://localhost:8080";

const API_BASE_URL: string = (import.meta as any).env?.VITE_API_BASE_URL ?? "";

async function http<T>(path: string, options: RequestInit = {}): Promise<T> {
  // URL 파라미터에서 userId 가져오기 (CMS 통합용)
  const userId = new URLSearchParams(window.location.search).get("userId");

  const res = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(userId ? { "x-user-id": userId } : {}),
      ...(options.headers || {}),
    },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  async translate(body: TranslationRequest): Promise<TranslationResponse> {
    if (USE_MOCK) {
      const mod = await import("../mock/articles/translate.mock.ts");
      return mod.translateMock(body);
    }
    return http<TranslationResponse>("/articles/translate", {
      method: "POST",
      body: JSON.stringify(body),
    });
  },

  async generateTitle(
    body: TitleGenerationRequest
  ): Promise<TitleGenerationResponse> {
    if (USE_MOCK) {
      const mod = await import("../mock/articles/generate-title.mock.ts");
      return mod.generateTitleMock(body);
    }
    // Backend의 /articles/seo 엔드포인트 사용
    return http<TitleGenerationResponse>("/articles/seo", {
      method: "POST",
      body: JSON.stringify(body),
    });
  },

  async listStyleGuides(params?: {
    category?: "TITLE" | "BODY" | "CAPTION";
    limit?: number;
    skip?: number;
  }): Promise<StyleGuideOut[]> {
    if (USE_MOCK) {
      const mod = await import("../mock/styleguides/list.mock.ts");
      return mod.listStyleGuidesMock(params);
    }
    const query = new URLSearchParams();
    if (params?.category) query.set("category", params.category);
    if (typeof params?.limit === "number")
      query.set("limit", String(params.limit));
    if (typeof params?.skip === "number")
      query.set("skip", String(params.skip));
    const qs = query.toString();
    return http<StyleGuideOut[]>(`/styleguides/${qs ? `?${qs}` : ""}`);
  },

  async getStyleGuide(styleId: number): Promise<StyleGuideOut> {
    if (USE_MOCK) {
      const mod = await import("../mock/styleguides/list.mock.ts");
      const allGuides = await mod.listStyleGuidesMock();
      const guide = allGuides.find((g) => g.id === styleId);
      if (!guide) throw new Error(`Style guide with id ${styleId} not found`);
      return guide;
    }
    return http<StyleGuideOut>(`/styleguides/${styleId}`);
  },

  async listNewsHistory(
    articleId: string,
    operationType?: "CORRECTION" | "TRANSLATION" | "TRANSLATION_CORRECTION",
    category?: string
  ): Promise<NewsHistoryResponseItem[]> {
    if (USE_MOCK) {
      const mod = await import("../mock/articles/history.mock.ts");
      return mod.listNewsHistoryMock(articleId, operationType);
    }
    const query = new URLSearchParams();
    if (operationType) query.set("operation_type", operationType);
    if (category) query.set("category", category);
    const qs = query.toString();
    return http<NewsHistoryResponseItem[]>(
      `/articles/${encodeURIComponent(articleId)}/history${qs ? `?${qs}` : ""}`
    );
  },

  streamCorrection(
    body: ArticleCorrectionRequest,
    onMessage: (message: SSEMessage) => void,
    onError?: (e: any) => void
  ): () => void {
    if (USE_MOCK) {
      // Simulate SSE via mock
      let cancelled = false;
      const modPromise = import("../mock/articles/correct-stream.mock.ts");
      modPromise
        .then((mod) =>
          mod.correctStreamMock(body, (chunk: SSEMessage) => {
            if (!cancelled) onMessage(chunk);
          })
        )
        .catch((e) => onError?.(e));
      return () => {
        cancelled = true;
      };
    }
    // SSE with POST needs Fetch + ReadableStream approach
    // EventSource only supports GET, so we use fetch with streaming
    let aborted = false;
    (async () => {
      try {
        // URL 파라미터에서 userId 가져오기 (CMS 통합용)
        const userId = new URLSearchParams(window.location.search).get(
          "userId"
        );

        const res = await fetch(`${API_BASE_URL}/articles/correct/stream`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(userId ? { "x-user-id": userId } : {}),
          },
          body: JSON.stringify(body),
        });

        // HTTP 상태 코드 체크 (502, 500 등 에러 처리)
        if (!res.ok) {
          const errorText = await res.text().catch(() => "Network error");
          throw new Error(`HTTP ${res.status}: ${errorText}`);
        }

        if (!res.body) throw new Error("No response body");
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        while (!aborted) {
          const { value, done } = await reader.read();
          if (done) break;
          const text = decoder.decode(value, { stream: true });
          text.split("\n\n").forEach((line) => {
            if (line.startsWith("data: ")) {
              onMessage(line.slice(6));
            }
          });
        }
      } catch (e) {
        if (!aborted) {
          onError?.(e);
        }
      }
    })();
    return () => {
      aborted = true;
    };
  },

  async generateSeoTitle(
    body: SeoTitleGenerationRequest
  ): Promise<SeoTitleGenerationResponse> {
    if (USE_MOCK) {
      const mod = await import("../mock/seo/generate-seo-title.mock.ts");
      return mod.generateSeoTitleMock(body);
    }

    // Backend API에 맞게 요청 변환
    const titleRequest: TitleGenerationRequest = {
      news_key: body.news_key,
      input_text: body.input_text,
      data_type: "headline", // SEO 타이틀 생성용 필수 필드
      model: "o4-mini", // 기본 모델
      selected_type: null,
      guideline_text: null,
    };

    const response = await http<TitleGenerationResponse>("/articles/seo", {
      method: "POST",
      body: JSON.stringify(titleRequest),
    });

    // 응답을 SeoTitleGenerationResponse 형식으로 변환
    return {
      seo_titles: response.seo_titles,
    };
  },

  async getArticle(
    articleId: string,
    category?: string
  ): Promise<CMSGetResponse> {
    const query = new URLSearchParams();
    if (category) query.set("category", category);
    const qs = query.toString();
    return http<CMSGetResponse>(
      `/articles/${encodeURIComponent(articleId)}${qs ? `?${qs}` : ""}`
    );
  },

  async restoreHistory(
    newsKey: string,
    category: string,
    historyId: number
  ): Promise<{ history_id: number; version: number; before_text: string; after_text: string }> {
    if (USE_MOCK) {
      // Mock implementation
      return {
        history_id: Date.now(),
        version: 1,
        before_text: "Restored input text",
        after_text: "Restored output text",
      };
    }
    return http<{ history_id: number; version: number; before_text: string; after_text: string }>(
      "/articles/history/restore",
      {
        method: "POST",
        body: JSON.stringify({
          news_key: newsKey,
          category: category,
          history_id: historyId,
        }),
      }
    );
  },

  async saveToCMS(body: CMSSaveRequest): Promise<CMSSaveResponse> {
    if (USE_MOCK) {
      const mod = await import("../mock/cms/save.mock.ts");
      return mod.saveToCMSMock(body);
    }
    return http<CMSSaveResponse>("/articles/save", {
      method: "POST",
      body: JSON.stringify(body),
    });
  },
};
