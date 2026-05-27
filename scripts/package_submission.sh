#!/usr/bin/env bash
set -euo pipefail
mkdir -p submission
cp -r hw submission/hw
cp report.md submission/report.md 2>/dev/null || true
if [ -d artifacts ]; then
  find artifacts -maxdepth 2 -type f \( -name "*.pt" -o -name "*.safetensors" -o -name "*.json" \) -print0 | \
    xargs -0 -I{} cp --parents {} submission/ || true
fi
zip -r submission.zip submission
printf "Created submission.zip\n"
