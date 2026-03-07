#!/bin/bash
# WoW 스토리 연대기 포스트 push

cd /home/dupre/.openclaw/workspace/anaham.github.io

git add _posts/2026-02-28-wow-story-chronicle.md

git commit -m "Add: 월드 오브 워크래프트 스토리 연대기 포스트"

git push

echo ""
echo "Push 완료!"
