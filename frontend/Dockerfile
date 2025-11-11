# Stage 1: Build
FROM node:20-bookworm-slim AS builder

WORKDIR /app

# 네이티브 번들(rollup) 빌드에 필요한 도구 설치 후 캐시 정리
RUN apt-get update && apt-get install -y python3 make g++ git \
    && rm -rf /var/lib/apt/lists/*

# 패키지 매니페스트만 먼저 복사해 설치 캐시 최적화
COPY package.json ./
COPY package-lock.json* ./

# npm 10.9 이상은 musl/arm64 선택적 의존성 버그가 해결되어 있음
RUN npm install

# 선택적 네이티브 의존성까지 확실히 설치
RUN npm ci --legacy-peer-deps --include=optional

# 애플리케이션 소스 복사
COPY . .

# 빌드 시 사용될 환경 변수 (필요 시 docker build --build-arg 로 재정의)
ARG VITE_USE_MOCK_API=false
ARG VITE_API_BASE_URL=
ENV VITE_USE_MOCK_API=${VITE_USE_MOCK_API}
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

# 프로덕션 빌드
RUN npm run build

# Stage 2: Serve with Nginx
FROM nginx:alpine

# 빌드된 파일 복사 (Vite는 build 디렉토리 사용)
COPY --from=builder /app/build /usr/share/nginx/html

# Nginx 설정 파일 복사 (Docker 전용 설정 사용)
COPY nginx.docker.conf /etc/nginx/conf.d/default.conf

# 포트 노출
EXPOSE 80

# Nginx 실행
CMD ["nginx", "-g", "daemon off;"]
