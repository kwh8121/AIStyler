# KT-Styler AWS EC2 배포 가이드

> **버전**: 1.0
> **최종 수정**: 2025년 12월
> **대상 환경**: AWS EC2 t3.small + RDS PostgreSQL

---

## 목차

1. [개요](#1-개요)
2. [사전 준비](#2-사전-준비)
3. [서버 초기 설정](#3-서버-초기-설정)
4. [Docker 설치](#4-docker-설치)
5. [프로젝트 배포](#5-프로젝트-배포)
6. [부록](#6-부록)

---

## 1. 개요

### 1.1 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                         Internet                                 │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AWS EC2 (t3.small)                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    Docker Compose                        │    │
│  │  ┌─────────────────┐      ┌─────────────────────────┐   │    │
│  │  │    Frontend     │      │       Backend           │   │    │
│  │  │  (Nginx + React)│──────│  (FastAPI + Uvicorn)    │   │    │
│  │  │    Port: 80     │      │      Port: 8080         │   │    │
│  │  └─────────────────┘      └───────────┬─────────────┘   │    │
│  └───────────────────────────────────────┼─────────────────┘    │
└──────────────────────────────────────────┼──────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     AWS RDS (PostgreSQL 15)                      │
│                         Port: 5432                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 사전 준비

### 2.1 EC2 인스턴스 생성

#### Step 1: EC2 대시보드에서 인스턴스 시작

1. AWS Console → EC2 → **인스턴스 시작**
2. 설정값:
   - **이름**: `kt-styler-prod` (예시)
   - **AMI**: Ubuntu Server 22.04 LTS (권장) 또는 Amazon Linux 2023
   - **인스턴스 유형**: t3.small
   - **키 페어**: 새로 생성 또는 기존 키 선택 (`.pem` 파일 안전하게 보관)

#### Step 2: 스토리지 설정

- **볼륨 크기**: 최소 20GB (권장 30GB)
- **볼륨 유형**: gp3 (범용 SSD)

#### Step 3: 네트워크 설정

- **VPC**: 기본 VPC 또는 커스텀 VPC
- **서브넷**: 퍼블릭 서브넷 선택
- **퍼블릭 IP 자동 할당**: 활성화

### 2.2 보안 그룹 설정

인바운드 규칙을 다음과 같이 설정합니다:

| 유형  | 프로토콜 | 포트 범위 | 소스      | 용도          |
| ----- | -------- | --------- | --------- | ------------- |
| SSH   | TCP      | 22        | 내 IP     | SSH 접속      |
| HTTP  | TCP      | 80        | 0.0.0.0/0 | 웹 서비스     |
| HTTPS | TCP      | 443       | 0.0.0.0/0 | SSL 웹 서비스 |

> 💡 **팁**: 개발 중에는 8080 포트도 열어두면 백엔드 직접 테스트가 가능합니다.

---

## 3. 서버 초기 설정

### 3.1 SSH 접속

```bash
# 키 파일 권한 설정 (최초 1회)
chmod 400 your-key.pem

# SSH 접속
ssh -i your-key.pem ubuntu@[EC2-PUBLIC-IP]

# Amazon Linux의 경우
ssh -i your-key.pem ec2-user@[EC2-PUBLIC-IP]
```

### 3.2 시스템 업데이트

#### Ubuntu 22.04:

```bash
sudo apt update && sudo apt upgrade -y
```

#### Amazon Linux 2023:

```bash
sudo dnf update -y
```

### 3.2 타임존 설정

```bash
# 현재 시간 확인
date

# 타임존을 한국 시간으로 변경
sudo timedatectl set-timezone Asia/Seoul

# 변경 확인
date
timedatectl
```

### 3.4 필수 패키지 설치

```bash
# Ubuntu
sudo apt install -y curl wget git vim htop

# Amazon Linux
sudo dnf install -y curl wget git vim htop
```

---

## 4. Docker 설치

### 4.1 Docker Engine 설치

#### Ubuntu 22.04:

```bash
# 이전 버전 제거 (있는 경우)
sudo apt remove docker docker-engine docker.io containerd runc 2>/dev/null

# 필수 패키지 설치
sudo apt update
sudo apt install -y ca-certificates curl gnupg lsb-release

# Docker GPG 키 추가
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Docker 저장소 추가
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Docker 설치
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

#### Amazon Linux 2023:

```bash
# Docker 설치
sudo dnf install -y docker

# Docker Compose 플러그인 설치
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 -o /usr/local/lib/docker/cli-plugins/docker-compose
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
```

### 4.2 Docker 서비스 시작 및 자동 시작 설정

```bash
# Docker 서비스 시작
sudo systemctl start docker

# 부팅 시 자동 시작 설정
sudo systemctl enable docker

# 상태 확인
sudo systemctl status docker
```

### 4.3 Docker 권한 설정 (sudo 없이 실행)

```bash
# docker 그룹에 현재 사용자 추가
sudo usermod -aG docker $USER

# 그룹 변경 적용 (재로그인 또는 아래 명령 실행)
newgrp docker

# 테스트
docker run hello-world
```

### 4.4 설치 확인

```bash
docker --version
# Docker version 24.x.x, build xxxxxxx

docker compose version
# Docker Compose version v2.x.x
```

---

## 5. 프로젝트 배포

### 5.1 프로젝트 클론

```bash
# 홈 디렉토리로 이동
cd ~

# Git 클론 (HTTPS)
git clone https://github.com/your-username/kt-styler.git

# 또는 SSH (SSH 키 설정 필요)
git clone git@github.com:your-username/kt-styler.git

# 프로젝트 디렉토리로 이동
cd kt-styler
```

### 5.2 환경변수 설정

```bash
# .env.example을 복사하여 .env 생성
cp .env.example .env

# .env 파일 편집
vim .env
```

### 5.3 Docker Compose로 서비스 시작

```bash
# 프로젝트 디렉토리 확인
cd ~/kt-styler

# 개발 환경으로 빌드 및 실행 DB 포함
docker compose up

# 프로덕션 환경으로 빌드 및 실행
docker compose -f docker-compose.prod.yml up -d --build
```

### 5.4 빌드 진행 상황 모니터링

```bash
# 실시간 로그 확인
docker compose -f docker-compose.prod.yml logs -f

# 특정 서비스만 확인
docker compose -f docker-compose.prod.yml logs -f backend
docker compose -f docker-compose.prod.yml logs -f frontend
```

### 5.5 서비스 상태 확인

```bash
# 컨테이너 상태 확인
docker compose -f docker-compose.prod.yml ps

# 예상 출력:
# NAME                IMAGE                    STATUS
# kt-styler-backend   kt-styler-backend        Up (healthy)
# kt-styler-frontend  kt-styler-frontend       Up
```

### 5.6 헬스체크

```bash
# Backend 헬스체크
curl http://localhost:8080/health
# 응답: {"status":"ok"}

# Frontend 확인
curl -I http://localhost
# 응답: HTTP/1.1 200 OK
```

### 5.7 브라우저에서 확인

- Frontend: `http://[EC2-PUBLIC-IP]`
- API 문서: `http://[EC2-PUBLIC-IP]:8080/docs`

---

## 6. 부록

### 6.1 전체 명령어 요약 (복사-붙여넣기용)

#### 서버 초기 설정 (Ubuntu):

```bash
# 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# 타임존 설정
sudo timedatectl set-timezone Asia/Seoul

# 필수 패키지
sudo apt install -y curl wget git vim htop postgresql-client
```

#### Docker 설치 (Ubuntu):

```bash
sudo apt install -y ca-certificates curl gnupg lsb-release
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
newgrp docker
```

#### 프로젝트 배포:

```bash
cd ~
git clone https://github.com/your-username/kt-styler.git
cd kt-styler
cp .env.example .env
vim .env  # 환경변수 설정
docker compose -f docker-compose.prod.yml up -d --build
```
