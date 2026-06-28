#!/data/data/com.termux/files/usr/bin/bash
set -e

cd ~/Bitcoin || exit 1

echo "Publishing public CohesivX status..."

python publish_public_status.py

git add cohesivx-public-status.json

if git diff --cached --quiet; then
  echo "No public status changes to push."
  exit 0
fi

git commit -m "Update CohesivX public status"

echo "Syncing with GitHub..."
git pull --rebase origin main

echo "Pushing public status..."
git push

echo "Public CohesivX status pushed successfully."
