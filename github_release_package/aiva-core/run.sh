#!/usr/bin/env bash
set -euo pipefail
python -m pip install -r requirements.txt
python -m core.boot_all memory/PAC_DeltaMemory.vessel .
