import React, { useEffect, useMemo, useState } from "react";
import { HistoryItem } from "./components/molecules/HistoryCard";
import { AppHeader } from "./components/organisms/AppHeader";
import { HistorySidebar } from "./components/organisms/HistorySidebar";
import { TextEditorSection } from "./components/organisms/TextEditorSection";
import { SeoTitleSection } from "./components/organisms/SeoTitleSection";
import { LoadingModal } from "./components/organisms/LoadingModal";
import { StyleGuidePopup } from "./components/organisms/StyleGuidePopup";
import { LocalModeIndicator } from "./components/organisms/LocalModeIndicator";
import { api } from "./api/client";
import {
  localStorageManager,
  type LocalHistoryItem,
} from "./utils/localStorageManager";
import type { ArticleCategory, ApiCategory } from "./api/types";
import type { StatusType } from "./components/atoms/StatusBadge";
import { toast, Toaster } from "sonner";

// 유효한 CMS 카테고리 목록 (s 있는 버전과 없는 버전 모두 지원)
const VALID_CMS_CATEGORIES = [
  "headline",
  "headlines",
  "article",
  "articles",
  "caption",
  "captions",
  "article_translator",
  "articles_translator",
  "seo",
];

// URL 파라미터 검증 결과 타입
interface ValidationResult {
  isValid: boolean;
  error?: string;
  errorType?:
    | "missing_category"
    | "invalid_category"
    | "missing_article_id"
    | "invalid_article_id";
}

// 로컬 모드 감지 함수
function detectLocalMode(
  cmsCategory: string | null,
  articleId: string | null
): boolean {
  const params = new URLSearchParams(window.location.search);

  // mode=local 파라미터가 있으면 로컬 모드
  if (params.get("mode") === "local") {
    return true;
  }

  // article_id가 없으면 로컬 모드
  if (!articleId || articleId.trim() === "") {
    return true;
  }

  return false;
}

// URL 파라미터 검증 함수
function validateURLParams(
  cmsCategory: string | null,
  articleId: string | null,
  isLocalMode: boolean
): ValidationResult {
  // 로컬 모드에서는 카테고리만 검증
  if (isLocalMode) {
    // 로컬 모드에서는 카테고리가 없어도 기본값 사용
    if (!cmsCategory || cmsCategory.trim() === "") {
      return { isValid: true }; // 기본값 사용
    }

    if (!VALID_CMS_CATEGORIES.includes(cmsCategory.toLowerCase())) {
      return {
        isValid: false,
        error: `유효하지 않은 카테고리입니다: "${cmsCategory}". 사용 가능한 카테고리: ${VALID_CMS_CATEGORIES.join(
          ", "
        )}`,
        errorType: "invalid_category",
      };
    }

    return { isValid: true };
  }

  // 서버 모드에서는 기존 검증 로직
  if (!cmsCategory || cmsCategory.trim() === "") {
    return {
      isValid: false,
      error:
        "category 파라미터가 필요합니다. (예: ?category=headlines&article_id=123)",
      errorType: "missing_category",
    };
  }

  if (!VALID_CMS_CATEGORIES.includes(cmsCategory.toLowerCase())) {
    return {
      isValid: false,
      error: `유효하지 않은 카테고리입니다: "${cmsCategory}". 사용 가능한 카테고리: ${VALID_CMS_CATEGORIES.join(
        ", "
      )}`,
      errorType: "invalid_category",
    };
  }

  // Article ID 검증 (존재 여부만 확인, 형식 제한 없음)
  if (!articleId || articleId.trim() === "") {
    return {
      isValid: false,
      error:
        "article_id 파라미터가 필요합니다. (예: ?category=headlines&article_id=123)",
      errorType: "missing_article_id",
    };
  }

  return { isValid: true };
}

// CMS 카테고리를 내부 상태로 매핑하는 함수
function mapCMSParams(cmsCategory: string) {
  const normalized = cmsCategory.toLowerCase();

  switch (normalized) {
    case "headline":
    case "headlines":
      return {
        category: "Headline" as ArticleCategory,
        apiCategory: "TITLE" as const, // API 요청용 카테고리
        tool: "styler" as const,
        currentTab: "styler" as const,
        backendCategory: "headlines", // 서버에는 항상 복수형 전달
      };
    case "article":
    case "articles":
      return {
        category: "Content" as ArticleCategory,
        apiCategory: "BODY" as const, // API 요청용 카테고리
        tool: "styler" as const,
        currentTab: "styler" as const,
        backendCategory: "articles", // 서버에는 항상 복수형 전달
      };
    case "caption":
    case "captions":
      return {
        category: "Caption" as ArticleCategory,
        apiCategory: "CAPTION" as const, // API 요청용 카테고리
        tool: "styler" as const,
        currentTab: "styler" as const,
        backendCategory: "captions", // 서버에는 항상 복수형 전달
      };
    case "article_translator":
    case "articles_translator":
      return {
        category: "Content" as ArticleCategory,
        apiCategory: "BODY" as const, // 번역도 본문 카테고리로 저장
        tool: "translator" as const,
        currentTab: "translator" as const,
        backendCategory: "articles", // 히스토리는 본문 카테고리와 동일하게 처리
      };
    case "seo":
      return {
        category: "SEO Title" as ArticleCategory,
        apiCategory: "SEO" as const, // API 요청용 카테고리
        tool: "seo" as const,
        currentTab: "styler" as const,
        backendCategory: "seo", // 서버에는 원본 카테고리 전달
      };
    default:
      // 이 케이스는 validateURLParams에서 이미 걸러짐
      return {
        category: "Content" as ArticleCategory,
        apiCategory: "BODY" as const, // API 요청용 카테고리
        tool: "styler" as const,
        currentTab: "styler" as const,
        backendCategory: "articles", // 기본값도 articles로 설정
      };
  }
}

export default function App() {
  // URL 파라미터 파싱
  const urlParams = new URLSearchParams(window.location.search);
  const cmsCategory = urlParams.get("category");
  const translatorFlag =
    (urlParams.get("translator") || "").toLowerCase() === "1" ||
    (urlParams.get("mode") || "").toLowerCase() === "translator";
  const articleIdParam =
    urlParams.get("article_id") || urlParams.get("articleId");
  const initialText = urlParams.get("text") || "";

  // 로컬 모드 감지
  const isLocalMode = detectLocalMode(cmsCategory, articleIdParam);

  // 개발모드 디버깅
  const isDevelopment = (import.meta as any).env?.MODE === "development";

  if (isDevelopment) {
    console.group("🔧 CMS Integration Debug");
    console.log("URL Parameters:", {
      category: cmsCategory,
      article_id: articleIdParam,
      text: initialText
        ? `"${initialText.substring(0, 50)}${
            initialText.length > 50 ? "..." : ""
          }"`
        : null,
      full_url: window.location.href,
    });
  }

  // URL 파라미터 검증 (로컬 모드 포함)
  const validationResult = validateURLParams(
    cmsCategory,
    articleIdParam,
    isLocalMode
  );

  if (isDevelopment) {
    console.log("Validation Result:", validationResult);
    if (validationResult.isValid) {
      console.log("✅ URL parameters are valid");
    } else {
      console.error("❌ URL validation failed:", validationResult.error);
    }
  }

  // 검증 실패 시 에러 상태로 처리
  const [validationError, setValidationError] = useState<string | null>(() =>
    validationResult.isValid
      ? null
      : validationResult.error || "URL 파라미터가 올바르지 않습니다."
  );

  // articles_translator 카테고리 → articles + translator 탭으로 리다이렉션 (URL 정규화)
  if (
    validationResult.isValid &&
    cmsCategory &&
    ["article_translator", "articles_translator", "translator"].includes(
      cmsCategory.toLowerCase()
    )
  ) {
    const usp = new URLSearchParams(window.location.search);
    usp.set("category", "articles");
    usp.set("translator", "1");
    // 하드 리다이렉트하여 초기 매핑/로딩 단계부터 일관되게 처리
    window.location.replace(`${window.location.pathname}?${usp.toString()}`);
  }

  // CMS 카테고리 매핑 (검증된 경우만)
  let mappedParams = validationResult.isValid
    ? mapCMSParams(cmsCategory!)
    : {
        category: "Contents" as ArticleCategory,
        apiCategory: "BODY" as const,
        tool: "styler" as const,
        currentTab: "styler" as const,
        backendCategory: "articles",
      };

  // translator 플래그가 있으면 번역 탭 강제 활성화 (category=articles일 때)
  if (translatorFlag) {
    mappedParams = {
      ...mappedParams,
      apiCategory: "BODY",
      tool: "translator",
      currentTab: "translator",
      backendCategory: "articles",
      category: "Content" as ArticleCategory,
    };
  }

  if (isDevelopment) {
    console.log("📝 Mapped Parameters:", mappedParams);
    console.groupEnd();
  }

  const category: ArticleCategory = mappedParams.category;
  const [userId, setUserId] = useState<string | null>(() =>
    urlParams.get("userId")
  );
  const [articleId, setArticleId] = useState<string | null>(articleIdParam);
  const [articleTitle, setArticleTitle] = useState<string | null>(() =>
    urlParams.get("articleTitle")
  );
  const [inputText, setInputText] = useState<string>(initialText);
  // CMS가 최초 전달한 원문 스냅샷 (사용자가 편집하더라도 원문 기준 반환 위해 보관)
  const [cmsOriginalText, setCmsOriginalText] = useState<string>(
    initialText || ""
  );
  const [outputText, setOutputText] = useState<string>("");
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [selectedHistoryId, setSelectedHistoryId] = useState<string | null>(
    null
  );
  const [isPreviewMode, setIsPreviewMode] = useState(false);
  const [tool, setTool] = useState<"styler" | "translator" | "seo">(
    mappedParams.tool
  );
  const [currentTab, setCurrentTab] = useState<"styler" | "translator">(
    mappedParams.currentTab
  );

  // 기사 로딩 상태
  const [isLoadingArticle, setIsLoadingArticle] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [additionalInstructions, setAdditionalInstructions] = useState("");
  const [isTranslating, setIsTranslating] = useState(false);
  const [isStylerLoading, setIsStylerLoading] = useState(false);
  const [stylerStatus, setStylerStatus] = useState<
    | "loading"
    | "translating"
    | "translation_complete"
    | "applying_style"
    | "complete"
    | undefined
  >(undefined);
  const [stylerPercent, setStylerPercent] = useState<number | undefined>(
    undefined
  );
  const [streamCancelFn, setStreamCancelFn] = useState<(() => void) | null>(
    null
  ); // 스트림 취소 함수
  const [isApplying, setIsApplying] = useState(false);
  const [showStyleGuide, setShowStyleGuide] = useState(false);
  const [isHistoryLoading, setIsHistoryLoading] = useState(false);

  const [historyItems, setHistoryItems] = useState<HistoryItem[]>([]);

  // 로컬 모드 관련 상태
  const [localHistoryStats, setLocalHistoryStats] = useState({
    totalItems: 0,
    storageUsedMB: "0.00",
  });

  const [outputOptions, setOutputOptions] = useState<string[]>([]);
  const [displaySelectedIndex, setDisplaySelectedIndex] = useState<number>(0);

  const [showSelectionWarning, setShowSelectionWarning] =
    useState<boolean>(false);
  const [isExtracting, setIsExtracting] = useState<boolean>(false);
  const [isCopied, setIsCopied] = useState<boolean>(false);

  // 로컬 모드 히스토리 로드
  const loadLocalHistories = () => {
    if (isLocalMode) {
      const localHistories = localStorageManager.getHistories();
      const mapped: HistoryItem[] = localHistories.map((h) => ({
        id: h.localId || h.id,
        date: new Date(h.createdAt).toLocaleString("ko-KR"),
        inputText: h.originalText,
        outputText: h.outputText,
        type: (h.operationType === "TRANSLATION" ? "번역" : "AI") as StatusType,
        appliedStyles: h.appliedStyles,
      }));
      setHistoryItems(mapped);

      // 통계 업데이트
      const stats = localStorageManager.getStats();
      setLocalHistoryStats({
        totalItems: stats.totalItems,
        storageUsedMB: stats.storageUsedMB,
      });
    }
  };

  // 로컬 모드 히스토리 내보내기
  const handleExportHistory = () => {
    localStorageManager.downloadBackup();
  };

  // 로컬 모드 히스토리 가져오기
  const handleImportHistory = async (file: File) => {
    const text = await file.text();
    const success = localStorageManager.importFromJSON(text);
    if (success) {
      loadLocalHistories();
      alert("히스토리를 성공적으로 가져왔습니다.");
    } else {
      alert("히스토리 가져오기에 실패했습니다.");
    }
  };

  // 로컬 모드 히스토리 전체 삭제
  const handleClearHistory = () => {
    if (
      confirm(
        "모든 로컬 히스토리를 삭제하시겠습니까? 이 작업은 취소할 수 없습니다."
      )
    ) {
      localStorageManager.clearAllHistories();
      loadLocalHistories();
    }
  };

  function mapOperationTypeToStatus(op?: string): StatusType {
    switch (op) {
      case "TRANSLATION":
        return "번역";
      case "RESTORATION":
        return "복원";
      case "CORRECTION":
      case "TRANSLATION_CORRECTION":
      default:
        return "AI";
    }
  }

  function formatDate(iso?: string): string {
    if (!iso) return "";
    const d = new Date(iso);
    const pad = (n: number) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(
      d.getDate()
    )} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
  }

  // 기사 자동 로딩
  useEffect(() => {
    async function loadArticle() {
      if (!articleId) {
        if (isDevelopment) {
          console.log("📄 Skipping article loading: No article_id", {
            articleId,
          });
        }
        return;
      }

      if (isDevelopment) {
        console.group("🔄 Loading Article");
        console.log("Article ID:", articleId);
        console.log("Expected Category:", mappedParams.backendCategory);
      }

      try {
        setIsLoadingArticle(true);
        setLoadError(null);
        // 서버의 최신 기사 텍스트를 우선 사용 (URL initialText는 폴백)
        let response: Awaited<ReturnType<typeof api.getArticle>> | null = null;
        try {
          response = await api.getArticle(
            articleId,
            mappedParams.backendCategory
          );
        } catch (e) {
          if (isDevelopment) {
            console.warn(
              "⚠️ getArticle failed, falling back to initialText",
              e
            );
          }
        }

        if (isDevelopment) {
          console.log("📥 Article Response:", response);
          console.log("📋 Category:", response.category);
        }

        if (response && response.content) {
          setInputText(response.content);
          setCmsOriginalText(response.content);
        } else if (initialText) {
          setInputText(initialText);
          setCmsOriginalText(initialText);
        }

        // SEO 모드인 경우 상태 변수를 설정하여 한 번만 실행되도록 함
        if (tool === "seo" && response.content) {
          // useEffect에서 처리되도록 함 (중복 호출 방지)
        }

        if (isDevelopment) {
          console.log("✅ Article loaded successfully");
          console.log(
            "Content preview:",
            `"${response.content.substring(0, 100)}${
              response.content.length > 100 ? "..." : ""
            }"`
          );
          console.groupEnd();
        }
      } catch (error) {
        console.error("Failed to load article:", error);
        setLoadError("기사를 불러올 수 없습니다.");

        if (isDevelopment) {
          console.error("❌ Article loading failed:", error);
          console.groupEnd();
        }
      } finally {
        setIsLoadingArticle(false);
      }
    }

    loadArticle();
  }, [
    articleId,
    initialText,
    mappedParams.backendCategory,
    cmsCategory,
    isDevelopment,
  ]);

  useEffect(() => {
    if (!isHistoryOpen) return;

    // 로컬 모드에서는 LocalStorage에서 히스토리 로드
    if (isLocalMode) {
      setIsHistoryLoading(true);
      loadLocalHistories();
      setIsHistoryLoading(false);
      return;
    }

    // 서버 모드에서는 API 호출
    (async () => {
      try {
        setIsHistoryLoading(true);
        // 현재 탭 카테고리에 맞춰 히스토리 필터링
        const list = await api.listNewsHistory(
          articleId || "demo-news-1",
          undefined, // operationType - 모든 타입 (번역, 교정, 복원)
          mappedParams.backendCategory // category 필터 적용
        );
        const mapped: HistoryItem[] = list.map((h) => ({
          id: String(h.history_id),
          date: formatDate(h.created_at),
          inputText: h.original_text,
          outputText: h.after_text,
          type: mapOperationTypeToStatus(h.operation_type),
          appliedStyles: h.applied_styles || [],
        }));
        setHistoryItems(mapped);
      } catch (_) {
        setHistoryItems([]);
      } finally {
        setIsHistoryLoading(false);
      }
    })();
  }, [isHistoryOpen, articleId, isLocalMode, mappedParams.backendCategory]);

  const selectedHistory = historyItems.find(
    (item) => item.id === selectedHistoryId
  );

  // 원본 텍스트 백업 (preview 취소 시 복원용)
  const [originalInputText, setOriginalInputText] = useState<string>("");
  const [originalOutputText, setOriginalOutputText] = useState<string>("");

  const handleHistorySelect = (item: HistoryItem) => {
    if (isDevelopment) {
      console.group("📋 History Preview");
      console.log("Selected history:", item);
      console.log("Current text backup:", {
        input: inputText,
        output: outputText,
      });
      console.log("Current state:", { tool, currentTab, category });
    }

    // 현재 텍스트를 백업
    setOriginalInputText(inputText);
    setOriginalOutputText(outputText);

    // 선택된 히스토리 내용을 화면에 표시
    setInputText(item.inputText);
    setOutputText(item.outputText);

    // Preview 모드 활성화
    setSelectedHistoryId(item.id);
    setIsPreviewMode(true);

    if (isDevelopment) {
      console.log("✅ Preview mode activated with history content");
      console.log("Final state:", { tool, currentTab, category });
      console.groupEnd();
    }
  };

  const handleRestore = async () => {
    if (selectedHistory) {
      if (isDevelopment) {
        console.log("💾 Restoring history to main editor");
      }

      // 로컬 모드: localStorage에 복원 기록 저장
      if (isLocalMode) {
        const newsKey = `local-${Date.now()}`;
        localStorageManager.saveHistory({
          id: newsKey,
          date: new Date().toLocaleString("ko-KR"),
          inputText: selectedHistory.inputText,
          outputText: selectedHistory.outputText,
          type: "복원" as StatusType,
          category: category,
          operationType: "RESTORATION",
        });
        loadLocalHistories();
      } else {
        // 서버 모드: API 호출하여 복원 히스토리 저장
        try {
          await api.restoreHistory(
            articleId || "demo-news-1",
            mappedParams.backendCategory,
            parseInt(selectedHistory.id)
          );
          if (isDevelopment) {
            console.log("✅ History restored on server");
          }
        } catch (error) {
          console.error("Failed to restore history:", error);
          toast.error(
            "히스토리를 불러올 수 없어요. 잠시 후 다시 시도해 주세요."
          );
        }
      }

      // Preview 모드를 해제하고 히스토리를 닫음 (텍스트는 이미 설정됨)
      setIsPreviewMode(false);
      setSelectedHistoryId(null);
      setIsHistoryOpen(false);

      // 백업 초기화
      setOriginalInputText("");
      setOriginalOutputText("");
    }
  };

  const handleCancelPreview = () => {
    if (isDevelopment) {
      console.log("❌ Canceling preview, restoring original text");
    }

    // 원본 텍스트 복원
    setInputText(originalInputText);
    setOutputText(originalOutputText);

    // Preview 모드 해제
    setIsPreviewMode(false);
    setSelectedHistoryId(null);

    // 백업 초기화
    setOriginalInputText("");
    setOriginalOutputText("");
  };

  const handleStyleText = () => {
    // Input 검증: 비어있거나 공백만 있는 경우
    if (!inputText || inputText.trim() === "") {
      toast.error("요청을 처리할 수 없어요. Text Input을 확인해 주세요.");
      return;
    }

    setIsStylerLoading(true);
    setStylerStatus("loading");
    setStylerPercent(undefined); // 진행률 초기화
    // API 요청에는 mappedParams.apiCategory를 사용
    const reqCategory = mappedParams.apiCategory;
    const newsKey = isLocalMode
      ? `local-${Date.now()}`
      : articleId || "demo-news-1";

    let appliedStylesBuffer: any[] = [];
    let finalOutputText = "";
    let hasReceivedFirstText = false; // 첫 텍스트 도착 감지
    let textBuffer = ""; // 텍스트 버퍼링
    let analysisData: any = null; // 분석 데이터 저장
    let hasError = false; // 에러 발생 여부
    let errorMessage = ""; // 에러 메시지
    let hasReceivedCompleteStatus = false; // 정상 완료 상태 수신 여부

    const controller = api.streamCorrection(
      {
        news_key: newsKey,
        category: reqCategory,
        text: inputText,
        prompt: additionalInstructions || undefined,
      },
      (message) => {
        try {
          const obj = JSON.parse(message);

          // 실제 텍스트 처리
          const delta = (obj?.choices?.[0]?.delta?.content ??
            obj?.data?.choices?.[0]?.delta?.content) as string | undefined;
          if (delta !== undefined) {
            // 첫 텍스트 도착 시 진행률 100%로 점프 후 모달 해제
            if (!hasReceivedFirstText) {
              hasReceivedFirstText = true;
              setStylerPercent(100); // 100%로 빠르게 애니메이션
              setTimeout(() => {
                setIsStylerLoading(false); // 약간의 딜레이 후 모달 해제
              }, 300); // 100% 애니메이션이 보이도록 300ms 대기
            }

            // 줄바꿈 처리: 서버가 "\\n"을 문자열로 보낼 수 있음
            const processedDelta = delta.replace(/\\n/g, "\n");

            // 텍스트 버퍼링 및 자연스러운 표시
            textBuffer += processedDelta;

            // 공백이나 구두점을 만났을 때 한 번에 표시
            if (
              processedDelta === " " ||
              processedDelta === "." ||
              processedDelta === "!" ||
              processedDelta === "?" ||
              processedDelta === "," ||
              processedDelta === "\n" ||
              textBuffer.length > 10
            ) {
              // 또는 버퍼가 10자 이상일 때
              finalOutputText = finalOutputText + textBuffer;
              setOutputText(finalOutputText);
              textBuffer = "";
            }
          }

          // applied_styles 정보 수집
          if (obj?.applied_styles) {
            appliedStylesBuffer = obj.applied_styles;
          }

          // 상태 메시지 처리
          const status = obj?.status as string | undefined;
          if (status) {
            // 에러 상태 처리
            if (status === "error") {
              hasError = true;
              errorMessage = obj?.message || "처리 중 오류가 발생했습니다.";
              setIsStylerLoading(false);
              setStylerStatus(undefined);
              setStylerPercent(undefined);
              toast.error(
                "일시적인 오류가 발생했어요. 잠시 후 다시 시도해 주세요."
              );
              return;
            }

            if (status === "analysis_complete") {
              // 분석 완료 시 데이터 저장
              analysisData = obj?.analysis;
              if (analysisData?.style_guide_violations) {
                appliedStylesBuffer = analysisData.style_guide_violations;
              }
              setStylerStatus("applying_style"); // 다음 단계로 진행
            } else if (
              status === "translating" ||
              status === "translation_complete" ||
              status === "applying_style"
            ) {
              setStylerStatus(status as any);
            } else if (status === "complete") {
              hasReceivedCompleteStatus = true; // 정상 완료 상태 수신
              setStylerStatus("complete");
            }
          }
        } catch (_) {
          // status messages, ignore unless complete
          if (message.includes('"status": "complete"')) {
            hasReceivedCompleteStatus = true; // 정상 완료 상태 수신
            setIsStylerLoading(false);
            setStylerStatus("complete");
          }
        }
        if (message === "[DONE]") {
          // 남은 버퍼 내용 출력
          if (textBuffer) {
            finalOutputText = finalOutputText + textBuffer;
            textBuffer = "";
          }

          setIsStylerLoading(false);
          setStylerStatus("complete");

          // 줄바꿈 정규화만 수행 (과도한 자동 문단 삽입 제거)
          const formattedText = finalOutputText
            .replace(/\r\n/g, "\n")
            .replace(/\n{3,}/g, "\n\n");

          setOutputText(formattedText);
          finalOutputText = formattedText;

          // 추가 프롬프트 초기화
          setAdditionalInstructions("");

          // 로컬 모드에서 히스토리 저장
          // 조건: finalOutputText가 있고, 실제 텍스트를 받았으며(hasReceivedFirstText), 에러 메시지가 아닌 경우
          const isValidOutput =
            finalOutputText &&
            hasReceivedFirstText &&
            !hasError &&
            !finalOutputText.toLowerCase().includes("error") &&
            !finalOutputText.toLowerCase().includes("failed") &&
            finalOutputText.length > 10; // 최소 길이 확인

          // 입력과 출력이 동일한지 확인 (공백 제거 후 비교)
          const isUnchanged = inputText.trim() === finalOutputText.trim();

          // 서버/모델 에러 체크 및 토스트 표시
          if (hasError) {
            // 이미 에러 토스트가 표시되었으므로 추가 토스트 없음
          } else if (!isValidOutput) {
            toast.error(
              "일시적인 오류가 발생했어요. 잠시 후 다시 시도해 주세요."
            );
          } else if (isUnchanged) {
            // 입력과 출력이 동일한 경우 - 두 가지 케이스 구분
            if (hasReceivedCompleteStatus) {
              // 정상 완료되었지만 교정할 내용이 없는 경우
              toast.info("교정할 내용이 없습니다.");
            } else {
              // complete 상태를 받지 못한 경우 - 처리 중 문제 발생
              toast.warning("처리 중 문제가 발생했어요. 다시 시도해 주세요.");
            }
          } else {
            // 정상 성공
            toast.success("교정이 완료되었어요.");
          }

          if (isLocalMode && isValidOutput) {
            localStorageManager.saveHistory({
              id: newsKey,
              date: new Date().toLocaleString("ko-KR"),
              inputText,
              outputText: finalOutputText,
              type: "AI" as StatusType,
              category: category, // ArticleCategory 타입 (Headline, Content 등)
              operationType: "CORRECTION",
              appliedStyles: appliedStylesBuffer,
            });
            loadLocalHistories();
          }
        }
      },
      (error) => {
        // 네트워크/연결 에러 처리
        console.error("Stream correction error:", error);
        setIsStylerLoading(false);
        setStylerStatus(undefined);
        setStylerPercent(undefined);
        toast.error("요청을 처리할 수 없어요. 연결 상태를 확인해 주세요.");
      }
    );

    // 스트림 취소 함수 저장
    setStreamCancelFn(() => controller);
  };

  // 교정 스트림 취소 핸들러
  const handleCancelCorrection = () => {
    if (streamCancelFn) {
      streamCancelFn(); // 스트림 중단
      setStreamCancelFn(null);
    }
    setIsStylerLoading(false);
    setStylerStatus(undefined);
    setStylerPercent(undefined);
  };

  const handleTranslate = async () => {
    // 입력 검증
    if (!inputText || inputText.trim() === "") {
      toast.error("요청을 처리할 수 없어요. Text Input을 확인해 주세요.");
      return;
    }

    setIsTranslating(true);
    try {
      const newsKey = isLocalMode
        ? `local-${Date.now()}`
        : articleId || "demo-news-1";

      const res = await api.translate({
        news_key: newsKey,
        category: mappedParams.apiCategory,
        text: inputText,
        target_lang: "EN-US",
      });
      setOutputText(res.translated_text);

      // 로컬 모드에서 히스토리 저장
      if (isLocalMode) {
        localStorageManager.saveHistory({
          id: newsKey,
          date: new Date().toLocaleString("ko-KR"),
          inputText,
          outputText: res.translated_text,
          type: "번역" as StatusType,
          category: category,
          operationType: "TRANSLATION",
        });
        loadLocalHistories();
      }

      // 성공 토스트
      toast.success("번역이 완료되었어요.");
    } catch (error) {
      console.error("Translation error:", error);
      toast.error("요청을 처리할 수 없어요. 연결 상태를 확인해 주세요.");
    } finally {
      setIsTranslating(false);
    }
  };

  const handleCopyResult = async () => {
    try {
      // outputText가 비어있으면 복사하지 않음
      if (!outputText || outputText.trim() === "") {
        console.warn("복사할 내용이 없습니다.");
        return;
      }

      await navigator.clipboard.writeText(outputText);
      setIsCopied(true);
      console.log("결과 복사됨:", outputText.substring(0, 50) + "...");

      // 2초 후 복사 상태 초기화
      setTimeout(() => {
        setIsCopied(false);
      }, 2000);
    } catch (error) {
      console.error("복사 실패:", error);
      // Fallback: textarea를 이용한 복사 방식
      try {
        const textarea = document.createElement("textarea");
        textarea.value = outputText;
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand("copy");
        document.body.removeChild(textarea);

        setIsCopied(true);
        console.log("Fallback 방식으로 복사 성공");

        setTimeout(() => {
          setIsCopied(false);
        }, 2000);
      } catch (fallbackError) {
        console.error("Fallback 복사도 실패:", fallbackError);
        alert("복사에 실패했습니다. 브라우저 설정을 확인해주세요.");
      }
    }
  };

  const handleApplyResult = async () => {
    // Output 검증: 비어있거나 공백만 있는 경우
    if (!outputText || outputText.trim() === "") {
      toast.error("적용할 결과가 없어요. 먼저 교정을 실행해 주세요.");
      return;
    }

    // 로컬 모드에서는 단순히 저장만 수행
    if (isLocalMode) {
      setIsApplying(true);
      // 현재 상태를 히스토리에 저장 (수동 저장)
      const newsKey = `local-${Date.now()}`;
      localStorageManager.saveHistory({
        id: newsKey,
        date: new Date().toLocaleString("ko-KR"),
        inputText,
        outputText,
        type: "User" as StatusType,
        category: category,
        operationType: "CORRECTION",
      });
      loadLocalHistories();

      setTimeout(() => {
        setIsApplying(false);
        toast.success("로컬 히스토리에 저장되었어요.");
      }, 500);
      return;
    }

    // 연결 상태 확인: Parent window나 Opener가 없는 경우
    const hasParent = window.parent && window.parent !== window;
    const hasOpener = !!window.opener;

    if (!hasParent && !hasOpener) {
      toast.error("요청을 처리할 수 없어요. 연결 상태를 확인해 주세요.");
      return;
    }

    // 서버 모드: 결과 적용 로직 (postMessage to CMS)
    setIsApplying(true);
    console.log("결과 적용됨:", outputText);

    try {
      // 1) 메시지 타입 결정 (CMS 요구 포맷)
      const typeForCMS = (() => {
        if (tool === "seo") return "STYLED_SEO_RESULT" as const;
        // backendCategory는 heads/articles/captions 중 하나
        const bcat = mappedParams.backendCategory;
        if (bcat === "headlines") return "STYLED_HEADLINES_RESULT" as const;
        if (bcat === "captions") return "STYLED_CAPTIONS_RESULT" as const;
        if (bcat === "articles") {
          // 번역 탭이면 번역 결과 타입, 아니면 기사 본문 결과 타입
          return currentTab === "translator"
            ? ("STYLED_ARTICLES_TRANSLATOR_RESULT" as const)
            : ("STYLED_ARTICLES_RESULT" as const);
        }
        return "STYLED_ARTICLES_RESULT" as const;
      })();

      // 2) original/styled 배열 구성 (문단 기준: \n\n)
      // 요구사항: original은 현재 입력칸의 값(inputText)을 기준으로 전송
      const originalBase =
        inputText && inputText.length > 0 ? inputText : cmsOriginalText;
      const originalParas = (originalBase || "").split(/\r?\n\r?\n/);
      const styledFull = outputText || "";
      const styledParas = styledFull.split(/\r?\n\r?\n/);

      const payload = (() => {
        const arr: { original: string; styled: string }[] = [];
        const max = Math.max(originalParas.length, styledParas.length);
        for (let i = 0; i < max; i++) {
          const orig = originalParas[i] ?? "";
          let styled = styledParas[i] ?? "";
          // 만약 styled가 더 많은 경우, 남은 것들을 마지막에 합쳐 전달
          if (
            i === originalParas.length - 1 &&
            styledParas.length > originalParas.length
          ) {
            const rest = styledParas.slice(i).join("\n\n");
            styled = rest;
          }
          arr.push({ original: orig, styled });
        }
        // original이 더 많은 경우 styled가 비어 있을 수 있음 → 그대로 전송
        return arr;
      })();

      const resultData = {
        type: typeForCMS,
        payload,
      } as const;

      // Parent window가 존재하는 경우 (iframe 내에서 실행)
      if (window.parent && window.parent !== window) {
        window.parent.postMessage(resultData, "*"); // 실제 환경에서는 CMS 도메인 지정 권장
        console.log("결과를 Parent window로 전송:", resultData);
      }

      // Opener가 존재하는 경우 (팝업으로 열린 경우)
      if (window.opener) {
        window.opener.postMessage(resultData, "*"); // 실제 환경에서는 CMS 도메인 지정 권장
        console.log("결과를 Opener window로 전송:", resultData);
      }

      // 성공 토스트 표시
      toast.success("적용 요청이 완료되었어요.");

      // 로딩 완료 후 창 닫기
      setTimeout(() => {
        setIsApplying(false);
        console.log("결과 적용 완료");

        // 팝업인 경우 창 닫기
        if (window.opener) {
          window.close();
        }
      }, 100);
    } catch (error) {
      console.error("CMS 저장 실패:", error);
      setIsApplying(false);
      toast.error("적용 중 오류가 발생했어요. 잠시 후 다시 시도해 주세요.");
    }
  };

  // SEO 모드일 때, article 로딩 후 자동으로 SEO 타이틀 생성 (최초 1회만)
  const [hasSeoExtracted, setHasSeoExtracted] = useState(false);

  // SEO 자동 API 호출 임시 주석 처리
  // useEffect(() => {
  //   // SEO 모드에서 inputText가 설정되고 아직 추출하지 않은 경우에만 실행
  //   if (tool === "seo" && inputText && !isLoadingArticle && !hasSeoExtracted) {
  //     handleExtractAgain();
  //     setHasSeoExtracted(true);
  //   }

  //   // tool이 변경되면 추출 상태 리셋
  //   if (tool !== "seo") {
  //     setHasSeoExtracted(false);
  //   }
  // }, [tool, inputText, isLoadingArticle, hasSeoExtracted]);

  const handleSelectOutput = (index: number, option: string) => {
    setDisplaySelectedIndex(index);
    setOutputText(option);
  };

  const handleExtractAgain = async () => {
    // inputText가 없으면 실행하지 않음
    if (!inputText || inputText.trim() === "") {
      console.error("SEO API call requires input text");
      return;
    }

    try {
      setIsExtracting(true);
      const newsKey = isLocalMode
        ? `local-${Date.now()}`
        : articleId || "demo-news-1";
      const res = await api.generateSeoTitle({
        news_key: newsKey,
        input_text: inputText,
      });
      setOutputOptions(res.seo_titles);
    } catch (error) {
      console.error("Failed to generate SEO titles:", error);
      setOutputOptions([]);
    } finally {
      setIsExtracting(false);
    }
  };

  // URL 파라미터 검증 에러
  if (validationError) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 p-6">
        <div className="max-w-6xl mx-auto">
          <div className="flex items-center justify-center min-h-screen">
            <div className="text-center max-w-lg">
              <div className="mb-6">
                <div className="text-6xl text-red-500 mb-4">⚠️</div>
                <h1 className="text-2xl font-bold text-gray-900 mb-2">
                  잘못된 URL 요청
                </h1>
              </div>

              <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
                <div className="text-lg font-medium text-red-700 mb-3">
                  {validationError}
                </div>

                {validationResult.errorType === "missing_category" && (
                  <div className="text-sm text-gray-700 space-y-2">
                    <p className="font-medium">올바른 URL 형식:</p>
                    <code className="block bg-gray-100 p-2 rounded text-xs">
                      {window.location.origin}?category=headlines&article_id=123
                    </code>
                  </div>
                )}

                {validationResult.errorType === "invalid_category" && (
                  <div className="text-sm text-gray-700 space-y-2">
                    <p className="font-medium">사용 가능한 카테고리:</p>
                    <ul className="text-xs space-y-1">
                      <li>
                        • <code>headline</code> / <code>headlines</code> - 제목
                        스타일링
                      </li>
                      <li>
                        • <code>article</code> / <code>articles</code> - 본문
                        스타일링
                      </li>
                      <li>
                        • <code>caption</code> / <code>captions</code> - 캡션
                        스타일링
                      </li>
                      <li>
                        • <code>articles_translator</code> - 본문 번역
                      </li>
                      <li>
                        • <code>seo</code> - SEO 제목 생성
                      </li>
                    </ul>
                  </div>
                )}

                {validationResult.errorType === "missing_article_id" && (
                  <div className="text-sm text-gray-700">
                    <p>CMS에서 올바른 article_id와 함께 요청해주세요.</p>
                  </div>
                )}
              </div>

              <div className="space-x-3">
                <button
                  onClick={() => window.history.back()}
                  className="px-6 py-2 bg-gray-500 text-white rounded-md hover:bg-gray-600 transition-colors"
                >
                  이전으로
                </button>
                <button
                  onClick={() => window.location.reload()}
                  className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
                >
                  다시 시도
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // 기사 로딩 중
  if (isLoadingArticle) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 p-6">
        <div className="max-w-6xl mx-auto">
          <LoadingModal isOpen={true} variant="simple" />
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              <div className="text-lg font-medium text-gray-700 mb-2">
                기사를 불러오는 중...
              </div>
              <div className="text-sm text-gray-500">잠시만 기다려주세요.</div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // 기사 로드 에러
  if (loadError) {
    const isCategoryMismatch =
      loadError.includes("카테고리") && loadError.includes("일치하지 않습니다");

    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 p-6">
        <div className="max-w-6xl mx-auto">
          <div className="flex items-center justify-center h-64">
            <div className="text-center max-w-md">
              <div className="text-lg font-medium text-red-600 mb-4">
                {loadError}
              </div>
              {isCategoryMismatch ? (
                <div className="space-y-3">
                  <div className="text-sm text-gray-600 mb-4">
                    올바른 URL로 다시 접속하거나 CMS에서 올바른 카테고리로
                    요청해주세요.
                  </div>
                  <div className="space-x-2">
                    <button
                      onClick={() => window.history.back()}
                      className="px-4 py-2 bg-gray-500 text-white rounded-md hover:bg-gray-600 transition-colors"
                    >
                      이전으로
                    </button>
                    <button
                      onClick={() => window.location.reload()}
                      className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
                    >
                      다시 시도
                    </button>
                  </div>
                </div>
              ) : (
                <button
                  onClick={() => window.location.reload()}
                  className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
                >
                  다시 시도
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 p-6">
      <div className="max-w-6xl mx-auto">
        {isLocalMode && (
          <LocalModeIndicator
            historyCount={localHistoryStats.totalItems}
            storageUsedMB={localHistoryStats.storageUsedMB}
            onExportHistory={handleExportHistory}
            onImportHistory={handleImportHistory}
            onClearHistory={handleClearHistory}
          />
        )}

        <AppHeader
          onHistoryClick={async () => {
            if (isDevelopment) {
              console.log("📂 Opening history sidebar", {
                currentTab,
                tool,
                category,
              });
            }
            // Open drawer and trigger loading
            setIsHistoryOpen(true);
          }}
          isHistoryLoading={isHistoryLoading}
        />

        {tool !== "seo" ? (
          <TextEditorSection
            category={category}
            articleId={articleId}
            inputText={inputText}
            outputText={outputText}
            isPreviewMode={isPreviewMode}
            currentTab={currentTab}
            additionalInstructions={additionalInstructions}
            isStylerLoading={isStylerLoading}
            isTranslating={isTranslating}
            isApplying={isApplying}
            isCopied={isCopied}
            onInputChange={(value) => {
              if (isDevelopment) {
                console.log("✏️ Input change:", {
                  value: value.substring(0, 50),
                  isPreviewMode,
                });
              }
              !isPreviewMode && setInputText(value);
            }}
            onAdditionalInstructionsChange={setAdditionalInstructions}
            onTabChange={(tab) => {
              if (isDevelopment) {
                console.log("🔄 Tab change:", { from: currentTab, to: tab });
              }
              setCurrentTab(tab);
              // 탭 전환 시 output 초기화 (UX 개선)
              setOutputText("");
            }}
            onCancelPreview={handleCancelPreview}
            onCopyResult={handleCopyResult}
            onApplyResult={handleApplyResult}
            onStyleText={handleStyleText}
            onTranslate={handleTranslate}
          />
        ) : (
          <SeoTitleSection
            inputText={inputText}
            isExtracting={isExtracting}
            isApplying={isApplying}
            onInputChange={(value) => !isPreviewMode && setInputText(value)}
            outputOptions={outputOptions}
            displaySelectedIndex={displaySelectedIndex}
            onSelectOutput={handleSelectOutput}
            showSelectionWarning={showSelectionWarning}
            onApplyResult={handleApplyResult}
            onExtractAgain={handleExtractAgain}
          />
        )}

        {/* Footer */}
        <footer className="flex items-center justify-center gap-2 mt-2 py-3">
          <img
            src="/logo.png"
            alt="한국언론진흥재단"
            style={{ maxHeight: "56px", height: "auto", width: "auto" }}
            className="object-contain"
          />
        </footer>
      </div>

      <HistorySidebar
        isOpen={isHistoryOpen}
        onOpenChange={(open) => {
          if (isDevelopment) {
            console.log(`📂 History sidebar ${open ? "opened" : "closed"}`, {
              currentTab,
              tool,
              category,
              isPreviewMode,
            });
          }
          setIsHistoryOpen(open);

          // 사이드바를 닫아도 preview 모드는 유지 (사용자가 명시적으로 취소할 때까지)
          // Preview 모드는 "보기 취소" 버튼을 통해서만 해제됨
        }}
        historyItems={historyItems}
        selectedHistoryId={selectedHistoryId}
        onHistorySelect={handleHistorySelect}
        onRestore={handleRestore}
        onShowStyleGuide={() => setShowStyleGuide(true)}
      />

      {/* Loading Modal */}
      <LoadingModal
        isOpen={isStylerLoading}
        variant="progress"
        status={stylerStatus}
        percent={stylerPercent}
        onCancel={handleCancelCorrection}
      />

      {/* Style Guide Popup */}
      {showStyleGuide && (
        <StyleGuidePopup onClose={() => setShowStyleGuide(false)} />
      )}

      {/* Toast Notifications */}
      <Toaster position="top-right" duration={3000} />
    </div>
  );
}
