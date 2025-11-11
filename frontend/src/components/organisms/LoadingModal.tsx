import React, { useEffect, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogDescription,
} from "../ui/dialog";
import { FileText, Languages, Paintbrush, Sparkles } from "lucide-react";
import { LoadingSpinner } from "../atoms/LoadingSpinner";
import { ProgressCircle } from "../atoms/ProgressCircle";
import { LoadingStep, LoadingStepData } from "../molecules/LoadingStep";

interface LoadingModalProps {
  isOpen: boolean;
  variant?: "simple" | "progress";
  status?:
    | "loading"
    | "translating"
    | "translation_complete"
    | "analysis_complete"
    | "applying_style"
    | "complete";
  percent?: number;
  message?: string;
  onCancel?: () => void; // 취소 버튼 콜백
}

const loadingStepsData: LoadingStepData[] = [
  {
    name: "Loading",
    icon: FileText,
    description: "텍스트 분석 중",
    progress: 33,
    color: "from-blue-500 to-blue-600",
  },
  {
    name: "Translating",
    icon: Languages,
    description: "언어 처리 중",
    progress: 66,
    color: "from-purple-500 to-purple-600",
  },
  {
    name: "Styling",
    icon: Paintbrush,
    description: "스타일 적용 중",
    progress: 100,
    color: "from-green-500 to-green-600",
  },
];

export function LoadingModal({
  isOpen,
  variant = "simple",
  status,
  percent,
  message,
  onCancel,
}: LoadingModalProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [progress, setProgress] = useState(0);
  const [startTime, setStartTime] = useState<number>(0);
  const timerIntervalRef = React.useRef<ReturnType<typeof setInterval> | null>(
    null
  );

  // 모달이 열릴 때 시작 시간 기록 및 타이머 시작
  useEffect(() => {
    if (isOpen && startTime === 0) {
      setStartTime(Date.now());
    } else if (!isOpen) {
      // 모달 닫힐 때 cleanup
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
        timerIntervalRef.current = null;
      }
      setCurrentStep(0);
      setProgress(0);
      setStartTime(0);
    }
  }, [isOpen, startTime]);

  // 타이머 기반 진행률: 2분(120초) 동안 0% → 95%까지 자연스럽게 증가
  useEffect(() => {
    if (!isOpen || startTime === 0 || variant !== "progress") return;

    // 이미 타이머가 실행 중이면 중복 실행 방지
    if (timerIntervalRef.current) return;

    const MAX_DURATION = 120000; // 2분
    const MAX_AUTO_PROGRESS = 95; // 자동으로 95%까지만

    timerIntervalRef.current = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const timerProgress = Math.min(
        (elapsed / MAX_DURATION) * MAX_AUTO_PROGRESS,
        MAX_AUTO_PROGRESS
      );
      const rounded = Math.round(timerProgress);

      setProgress((prev: number) => Math.max(prev, rounded));
    }, 100);

    return () => {
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
        timerIntervalRef.current = null;
      }
    };
  }, [isOpen, startTime, variant]);

  // SSE 이벤트 기반 진행률 점프 및 단계 변경
  useEffect(() => {
    if (!isOpen || !status) return;

    const statusToStepIndex: Record<string, number> = {
      loading: 0,
      translating: 1,
      translation_complete: 1,
      analysis_complete: 2,
      applying_style: 2,
      complete: 2,
    };
    const idx = statusToStepIndex[status] ?? 0;
    setCurrentStep(idx);

    // SSE 상태별 최소 진행률 보장
    const statusToMinProgress: Record<string, number> = {
      loading: 10,
      translating: 10, // 번역 시작 → 10% 유지
      translation_complete: 30, // 번역 완료 → 30%로 점프 (요구사항 반영)
      analysis_complete: 30, // 분석 완료 → 50%
      applying_style: 30, // 스타일 적용 → 50%
      complete: 100, // 완료 → 100%
    };

    const minProgress = statusToMinProgress[status] ?? 0;

    if (typeof percent === "number") {
      setProgress((prev: number) => {
        const newProgress = Math.max(prev, percent);
        // 점프가 발생한 경우 startTime 재조정
        if (newProgress > prev && startTime > 0) {
          const MAX_DURATION = 150000;
          const MAX_AUTO_PROGRESS = 95;
          // 현재 진행률에 해당하는 경과 시간 계산
          const equivalentElapsed =
            (newProgress / MAX_AUTO_PROGRESS) * MAX_DURATION;
          // startTime을 재조정하여 타이머가 현재 진행률부터 계속 증가하도록 함
          setStartTime(Date.now() - equivalentElapsed);
        }
        return newProgress;
      });
    } else {
      setProgress((prev: number) => {
        const newProgress = Math.max(prev, minProgress);
        // 점프가 발생한 경우 startTime 재조정
        if (newProgress > prev && startTime > 0) {
          const MAX_DURATION = 120000;
          const MAX_AUTO_PROGRESS = 95;
          // 현재 진행률에 해당하는 경과 시간 계산
          const equivalentElapsed =
            (newProgress / MAX_AUTO_PROGRESS) * MAX_DURATION;
          // startTime을 재조정하여 타이머가 현재 진행률부터 계속 증가하도록 함
          setStartTime(Date.now() - equivalentElapsed);
        }
        return newProgress;
      });
    }
  }, [status, percent, isOpen, startTime]);

  if (!isOpen) return null;

  const currentStepData = loadingStepsData[currentStep];

  if (variant === "simple") {
    return (
      <Dialog open={isOpen}>
        <DialogContent className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-auto border-0 bg-white rounded-2xl shadow-2xl px-12 py-10 text-center [&>button]:hidden">
          <DialogTitle className="text-2xl font-bold text-gray-900 mb-8">
            AI Styler Processing
          </DialogTitle>
          <DialogDescription className="sr-only">
            AI 텍스트 스타일링 처리 중입니다. Loading, Translating, Styling
            단계로 진행됩니다.
          </DialogDescription>

          <motion.div className="flex justify-center mb-8">
            <LoadingSpinner size="lg" />
          </motion.div>

          <motion.div className="text-lg text-gray-600">
            {loadingStepsData.map((step, index) => (
              <motion.span key={step.name} className="inline-flex items-center">
                <motion.span
                  className={`transition-colors duration-300 ${
                    index <= currentStep
                      ? "text-blue-600 font-medium"
                      : "text-gray-400"
                  }`}
                >
                  {step.name}
                </motion.span>
                {index < loadingStepsData.length - 1 && (
                  <motion.span className="mx-2 text-gray-300">
                    {" "}
                    {">"}{" "}
                  </motion.span>
                )}
              </motion.span>
            ))}
          </motion.div>
        </DialogContent>
      </Dialog>
    );
  }

  return (
    <Dialog
      open={isOpen}
      onOpenChange={(open: boolean) => {
        // Dialog가 닫힐 때 (X 버튼 클릭 or ESC) onCancel 호출
        if (!open && onCancel) {
          onCancel();
        }
      }}
    >
      <DialogContent className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-[520px] border-0 bg-white rounded-3xl shadow-2xl p-0 overflow-hidden">
        {/* 접근성을 위한 숨겨진 제목과 설명 */}
        <DialogTitle className="sr-only">AI Styler 처리 중</DialogTitle>
        <DialogDescription className="sr-only">
          AI 텍스트 스타일링 처리가 진행 중입니다. 번역, 분석, 스타일 적용
          단계로 진행됩니다.
        </DialogDescription>

        {/* 상단 프로그레스 바 애니메이션 */}
        <motion.div
          className={`h-2 bg-gradient-to-r ${currentStepData.color}`}
          initial={{ scaleX: 0 }}
          animate={{ scaleX: progress / 100 }}
          style={{ transformOrigin: "left" }}
          transition={{ duration: 0.3, ease: "easeOut" }}
        />

        <motion.div className="px-12 py-12 relative">
          {/* 배경 파티클 효과 */}
          <div className="absolute inset-0 overflow-hidden">
            {[...Array(5)].map((_, i) => (
              <motion.div
                key={i}
                className="absolute w-2 h-2 bg-gradient-to-r from-blue-400 to-purple-400 rounded-full opacity-30"
                initial={{
                  x: Math.random() * 520,
                  y: Math.random() * 400,
                }}
                animate={{
                  x: Math.random() * 520,
                  y: Math.random() * 400,
                }}
                transition={{
                  duration: 10 + Math.random() * 10,
                  repeat: Infinity,
                  repeatType: "reverse",
                  ease: "linear",
                }}
              />
            ))}
          </div>

          <motion.div
            className="text-center mb-8 relative z-10"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <motion.div className="inline-flex items-center gap-2 mb-2">
              <motion.div className="text-3xl font-bold text-gray-800">
                AI Styler
              </motion.div>
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
              >
                <Sparkles className="w-6 h-6 text-yellow-500" />
              </motion.div>
            </motion.div>
            <AnimatePresence mode="wait">
              <motion.div
                key={status}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.3 }}
              >
                <motion.div className="text-gray-600">
                  {status === "translating" && "번역 중입니다..."}
                  {status === "translation_complete" && "번역 완료!"}
                  {status === "analysis_complete" && "스타일 가이드 분석 완료!"}
                  {status === "applying_style" && "스타일 적용 중..."}
                  {(!status || status === "loading") &&
                    "고품질 텍스트 처리 중입니다"}
                </motion.div>
              </motion.div>
            </AnimatePresence>
          </motion.div>

          <motion.div
            className="flex justify-center mb-8 relative z-10"
            animate={{ scale: [1, 1.05, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            <div className="relative">
              <ProgressCircle
                progress={progress}
                icon={currentStepData.icon}
                size="md"
                fromColor={
                  currentStepData.color.includes("blue")
                    ? "#3b82f6"
                    : currentStepData.color.includes("purple")
                    ? "#8b5cf6"
                    : "#10b981"
                }
                toColor={
                  currentStepData.color.includes("blue")
                    ? "#2563eb"
                    : currentStepData.color.includes("purple")
                    ? "#7c3aed"
                    : "#059669"
                }
              />
              {/* 반짝임 효과 */}
              <motion.div
                className="absolute inset-0 rounded-full"
                style={{
                  background: `radial-gradient(circle, ${
                    currentStepData.color.includes("blue")
                      ? "rgba(59,130,246,0.3)"
                      : currentStepData.color.includes("purple")
                      ? "rgba(139,92,246,0.3)"
                      : "rgba(16,185,129,0.3)"
                  } 0%, transparent 70%)`,
                }}
                animate={{ scale: [1, 1.3, 1], opacity: [0.5, 0, 0.5] }}
                transition={{ duration: 2, repeat: Infinity }}
              />
            </div>
          </motion.div>

          <motion.div
            className="relative z-10"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
          >
            <LoadingStep
              step={currentStepData}
              index={currentStep}
              currentStep={currentStep}
              isTimeline={false}
            />
          </motion.div>

          <div className="space-y-4 mt-8 relative z-10">
            {loadingStepsData.map((step, index) => (
              <motion.div
                key={step.name}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <LoadingStep
                  step={step}
                  index={index}
                  currentStep={currentStep}
                  isTimeline={true}
                />
              </motion.div>
            ))}
          </div>

          <motion.div
            className="mt-8 text-center text-sm text-gray-500 relative z-10"
            animate={{ opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            <motion.div className="flex items-center justify-center gap-2">
              <motion.span
                animate={{ opacity: [0, 1, 0] }}
                transition={{ duration: 1.5, repeat: Infinity, delay: 0 }}
                className="w-1 h-1 bg-gray-400 rounded-full"
              />
              <motion.span
                animate={{ opacity: [0, 1, 0] }}
                transition={{ duration: 1.5, repeat: Infinity, delay: 0.2 }}
                className="w-1 h-1 bg-gray-400 rounded-full"
              />
              <motion.span
                animate={{ opacity: [0, 1, 0] }}
                transition={{ duration: 1.5, repeat: Infinity, delay: 0.4 }}
                className="w-1 h-1 bg-gray-400 rounded-full"
              />
              <motion.span>
                {message ||
                  "잠시만 기다려주세요. 최고 품질의 결과를 준비하고 있습니다."}
              </motion.span>
            </motion.div>
          </motion.div>
        </motion.div>
      </DialogContent>
    </Dialog>
  );
}
