#!/bin/sh
set -e

echo "⏳  DB 연결 대기 중..."
until pg_isready -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER"; do
  sleep 1
done
echo "✅  DB 준비 완료"

echo "🔄  Requirements 설치 중..."
uv sync


echo "🚀  Alembic 마이그레이션 실행"
cd /app
uv run alembic upgrade head

if [ -n "$STYLE_GUIDES_FILE" ] && [ -f "$STYLE_GUIDES_FILE" ]; then
  echo "🔎  Checking if style_guides table is empty before seeding..."
  COUNT=-1
  if command -v psql >/dev/null 2>&1; then
    COUNT=$(PGPASSWORD="$POSTGRES_PASSWORD" \
      psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
      -Atc "SELECT COUNT(*) FROM style_guides;" 2>/dev/null || echo -1)
  fi

  if [ "$COUNT" = "0" ]; then
    echo "🌱  style_guides is empty. Seeding from $STYLE_GUIDES_FILE (skip duplicates)"
    uv run python manage.py import-styleguides --file "$STYLE_GUIDES_FILE" --mode skip || true
  else
    echo "⏭️  Skipping seed. style_guides has $COUNT rows (or count check unavailable)."
  fi
fi

echo "🌏  Uvicorn 기동"
exec uv run uvicorn app.main:app --host 0.0.0.0 --port 8080
