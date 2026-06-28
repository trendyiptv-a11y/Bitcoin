#!/data/data/com.termux/files/usr/bin/bash
set -e

cd ~/Bitcoin || exit 1

LAST_PUSH_FILE="btc-swing-strategy/last_public_status_push_epoch.txt"
NOW=$(date +%s)
MIN_INTERVAL=$((60 * 60))

if [ -f "$LAST_PUSH_FILE" ]; then
  LAST_PUSH=$(cat "$LAST_PUSH_FILE")
else
  LAST_PUSH=0
fi

AGE=$((NOW - LAST_PUSH))

if [ "$AGE" -lt "$MIN_INTERVAL" ]; then
  REMAINING=$((MIN_INTERVAL - AGE))
  echo "Public status push skipped. Next allowed in ${REMAINING}s."
  exit 0
fi

echo "Publishing public CohesivX status hourly..."

python publish_public_status.py

git add cohesivx-public-status.json

if git diff --cached --quiet; then
  echo "No public status changes to push."
  echo "$NOW" > "$LAST_PUSH_FILE"
  exit 0
fi

git commit -m "Update CohesivX public status"

echo "Syncing with GitHub..."
git pull --rebase origin main

echo "Pushing public status..."
git push

echo "$NOW" > "$LAST_PUSH_FILE"

echo "Public CohesivX status pushed successfully."
