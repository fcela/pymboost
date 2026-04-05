#!/usr/bin/env bash
set -euo pipefail

rm -rf public book/_build

jupyter-book build book

if [ -d "book/_build/html" ]; then
  cp -R book/_build/html public
else
  echo "Expected book/_build/html to exist after Jupyter Book build" >&2
  exit 1
fi
