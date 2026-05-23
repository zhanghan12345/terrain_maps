#!/bin/bash
cd "$(dirname "$0")"
git add -A
git commit -m "update $(date '+%m-%d %H:%M')"
git push
echo "推送完成!"
