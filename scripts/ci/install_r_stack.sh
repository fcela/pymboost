#!/usr/bin/env bash
set -euo pipefail

apt-get update
apt-get install -y --no-install-recommends \
  build-essential \
  gfortran \
  r-base \
  r-base-dev

Rscript -e 'install.packages("mboost", repos = "https://cloud.r-project.org")'
