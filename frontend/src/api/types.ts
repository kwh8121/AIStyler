export type ArticleCategory =
  | "TITLE"
  | "BODY"
  | "CAPTION"
  | "Headline"
  | "Content"
  | "Caption"
  | "SEO Title";
export type ApiCategory =
  | "TITLE"
  | "BODY"
  | "CAPTION"
  | "SEO";
export type OperationType =
  | "CORRECTION"
  | "TRANSLATION"
  | "TRANSLATION_CORRECTION";

export interface ArticleCorrectionRequest {
  news_key: string;
  category: ApiCategory;
  text: string;
  prompt?: string | null;
}

export interface ArticleCorrectionResponseItem {
  history_id: number;
  news_key: string;
  category: ArticleCategory;
  version: number;
  before_text: string;
  after_text: string;
  operation_type?: OperationType;
  created_at?: string;
}

export interface AppliedStyleGuide {
  style_id: number;
  number?: number;
  name: string;
  category: string;
  docs?: string;
  applied_at: string;
  note?: string;

  // 문장별 교정 정보
  sentence_index?: number;
  before_text?: string;
  after_text?: string;
  violations?: any[];
}

export interface NewsHistoryResponseItem {
  history_id: number;
  news_key: string;
  category: ArticleCategory;
  version: number;
  original_text: string;
  before_text: string;
  after_text: string;
  operation_type?: OperationType;
  source_lang?: string;
  target_lang?: string;
  created_at?: string;
  applied_styles: AppliedStyleGuide[];
}

export interface TranslationRequest {
  news_key: string;
  category: ApiCategory;
  text: string;
  source_lang?: string | null;
  target_lang: string; // e.g. EN-US
}

export interface TranslationResponse {
  translated_text: string;
  source_lang: string;
  target_lang: string;
  history_id?: number;
  version?: number;
}

export interface TitleGenerationRequest {
  news_key: string;
  input_text: string;
  selected_type?: string | null;
  data_type: string; // 필수: "headline" 등
  model?: string; // 기본값: "o4-mini"
  guideline_text?: string | null;
}

export interface TitleGenerationResponse {
  edited_title: string;
  seo_titles: string[];
  raw_response: string;
}

export interface StyleGuideOut {
  id: number;
  name: string;
  category: ArticleCategory;
  docs: string;
  version: number;
  created_at: string;
  updated_at: string;
  deleted_at?: string | null;
}

export interface LoginResponse {
  access_token: string;
  refresh_token: string;
  token_type: "bearer";
}

export type SSEMessage = string;

export interface SeoTitleGenerationRequest {
  news_key: string;
  input_text: string;
}

export interface SeoTitleGenerationResponse {
  seo_titles: string[];
}

export interface CMSGetResponse {
  article_id: string;
  news_key: string;
  category: string;
  content: string;
}

export interface CMSSaveRequest {
  article_id: string;
  category: string;
  content: string;
  author_id?: string;
}

export interface CMSSaveResponse {
  article_id: string;
  news_key: string;
  category: string;
  success: boolean;
}
