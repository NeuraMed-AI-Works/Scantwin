#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
[ -d .git ] || git init
git branch -M main
git add .
git commit -m "Initial commit: ScanTwin v3 by NEURA MED AI" || true
git remote remove origin 2>/dev/null || true
git remote add origin https://github.com/NeuraMed-AI-Works/scantwin.git
git push -u origin main
