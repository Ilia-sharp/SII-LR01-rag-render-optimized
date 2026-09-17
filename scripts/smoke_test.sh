#!/usr/bin/env bash
# Быстрая проверка запущенного сервиса: bash scripts/smoke_test.sh
set -euo pipefail
HOST="${1:-http://localhost:8000}"

echo "== GET /health =="
curl -s "$HOST/health"; echo

echo "== POST /ask (вопрос по корпусу) =="
curl -s -X POST "$HOST/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"За сколько дней нужно подать заявление на отпуск?"}'; echo

echo "== POST /ask (вопрос про командировку) =="
curl -s -X POST "$HOST/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"Какой лимит на проживание в гостинице в Москве?"}'; echo

echo "== POST /ask (темы нет в документах) =="
curl -s -X POST "$HOST/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"Как приготовить борщ из марсианской свёклы?"}'; echo
