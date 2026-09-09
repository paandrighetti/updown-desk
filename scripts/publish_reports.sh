#!/usr/bin/env sh
# Commit and push new reports. Run from the repository root, daily, after the reporter.
# Example crontab line (reporter runs at 06:00 UTC):
#   30 6 * * * cd /opt/updown-desk && sh scripts/publish_reports.sh >> publish.log 2>&1
set -eu
git add reports
if git diff --cached --quiet; then
  echo "no new report"
  exit 0
fi
git commit -m "report: $(date -u +%F)" --quiet
git push --quiet
echo "pushed report $(date -u +%F)"
