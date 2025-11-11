#!/bin/bash

# Korea Times AI Styler Frontend 배포 스크립트
# 사용법: ./deploy.sh [옵션]
# 옵션:
#   --docker    : Docker를 사용한 배포
#   --direct    : 직접 파일 복사 배포
#   --build-only: 빌드만 수행

set -e  # 오류 발생 시 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 로그 함수
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# 배포 방식 확인
DEPLOY_METHOD=${1:-"--build-only"}

# 환경 변수 확인
if [ -f .env.production ]; then
    log_info "프로덕션 환경 변수 파일 발견"
    export $(cat .env.production | grep -v '^#' | xargs)
else
    log_error ".env.production 파일이 없습니다"
    exit 1
fi

# 1. 의존성 설치
log_info "의존성 설치 중..."
npm ci

# 2. 빌드 실행
log_info "프로덕션 빌드 시작..."
npm run build

if [ $? -eq 0 ]; then
    log_info "빌드 성공!"
    log_info "빌드 디렉토리 크기: $(du -sh build | cut -f1)"
else
    log_error "빌드 실패"
    exit 1
fi

# 빌드만 수행하는 경우 여기서 종료
if [ "$DEPLOY_METHOD" = "--build-only" ]; then
    log_info "빌드 완료. 배포는 수행하지 않습니다."
    exit 0
fi

# Docker 배포
if [ "$DEPLOY_METHOD" = "--docker" ]; then
    log_info "Docker 이미지 빌드 중..."
    docker build -t koreatimes-styler-frontend:latest .

    log_info "기존 컨테이너 중지 및 제거..."
    docker-compose down || true

    log_info "새 컨테이너 시작..."
    docker-compose up -d

    if [ $? -eq 0 ]; then
        log_info "Docker 배포 성공!"
        log_info "컨테이너 상태:"
        docker-compose ps
    else
        log_error "Docker 배포 실패"
        exit 1
    fi
fi

# 직접 배포 (파일 복사)
if [ "$DEPLOY_METHOD" = "--direct" ]; then
    # 서버 정보 설정 (실제 서버 정보로 변경 필요)
    SERVER_USER="your-username"
    SERVER_HOST="your-server-ip"
    SERVER_PATH="/var/www/html/styler"

    log_warning "직접 배포를 시작합니다."
    log_info "대상 서버: ${SERVER_USER}@${SERVER_HOST}"

    # 서버에 디렉토리 생성
    log_info "서버에 디렉토리 생성..."
    ssh ${SERVER_USER}@${SERVER_HOST} "mkdir -p ${SERVER_PATH}"

    # 빌드 파일 복사
    log_info "빌드 파일을 서버로 복사 중..."
    rsync -avz --delete ./build/ ${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}/

    # Nginx 설정 복사
    log_info "Nginx 설정 복사..."
    scp nginx.conf ${SERVER_USER}@${SERVER_HOST}:/tmp/styler-nginx.conf
    ssh ${SERVER_USER}@${SERVER_HOST} "sudo mv /tmp/styler-nginx.conf /etc/nginx/sites-available/styler && sudo ln -sf /etc/nginx/sites-available/styler /etc/nginx/sites-enabled/"

    # Nginx 재시작
    log_info "Nginx 재시작..."
    ssh ${SERVER_USER}@${SERVER_HOST} "sudo nginx -t && sudo systemctl reload nginx"

    if [ $? -eq 0 ]; then
        log_info "직접 배포 성공!"
    else
        log_error "직접 배포 실패"
        exit 1
    fi
fi

log_info "배포 완료!"
log_info "애플리케이션 상태를 확인해주세요."