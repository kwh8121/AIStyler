import type { ArticleCorrectionRequest, SSEMessage } from "../../api/types";

export function correctStreamMock(
  body: ArticleCorrectionRequest,
  push: (message: SSEMessage) => void
) {
  const steps: SSEMessage[] = [
    JSON.stringify({ status: "translating", message: "번역중..." }),
    JSON.stringify({ status: "translation_complete", message: "번역 완료" }),
    JSON.stringify({
      status: "applying_style",
      message: "스타일 가이드 적용중...",
    }),
  ];
  let idx = 0;
  const interval = setInterval(() => {
    if (idx < steps.length) {
      push(steps[idx++]);
    } else if (idx < steps.length + 5) {
      // emit 5 chunks
      const chunk = {
        choices: [{ delta: { content: " corrected-chunk(mock)" } }],
      };
      push(JSON.stringify(chunk));
      idx++;
    } else if (idx === steps.length + 5) {
      push(JSON.stringify({ status: "complete", message: "교정 완료" }));
      idx++;
    } else {
      push("[DONE]");
      clearInterval(interval);
    }
  }, 400);
}
