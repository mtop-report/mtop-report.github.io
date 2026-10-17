#!/usr/bin/env bash
# Fails if git tracks anything that could contain patient-level data.
set -euo pipefail
bad=$(git ls-files | grep -E '^(data/raw/|data/processed/|artifacts/private/|reports/)' | grep -vE '(^data/raw/README\.md$|^reports/README\.md$)' || true)
bad+=$(git ls-files | grep -E '\.(xlsx|xls|csv|parquet|sav|dta)$' | grep -vE '^(data/synthetic/registry_synthetic\.xlsx|artifacts/public/demo_synthetic\.csv)$' || true)
if [ -n "$bad" ]; then
  echo "Refusing: these tracked files may contain patient data:"
  echo "$bad"
  exit 1
fi
echo "OK: no patient-level files are tracked."
