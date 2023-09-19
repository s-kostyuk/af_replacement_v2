#!/bin/bash

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=".:$PYTHONPATH"

SRGK_VENV_PYTHON="./venv/bin/python3"
HOSTNAME="$(hostname)"
SEEDS=(100 128 1999 7823 42)
AAF_FAMILY="$1"
shift

if [[ -z "$AAF_FAMILY" ]]; then
  echo "AAF family required as argument 1 (ahaf or fuzzy_ffn)"
  return 1
fi

if [[ "$AAF_FAMILY" == "ahaf" ]]; then
  AAF_LIKE="all_lus"
else
  AAF_LIKE="all_bfs"
fi

if [[ -f "$SRGK_VENV_PYTHON" ]]; then
  SRGK_PYEXEC="$SRGK_VENV_PYTHON"
else
  SRGK_PYEXEC="python3"
fi

for el in "${SEEDS[@]}"; do
  echo "-------------------------------------------"
  echo " Running experiments with seed value: $el"
  echo "-------------------------------------------"

  # Separate folders by seeds
  SRGK_RUNS_PATH="runs_${HOSTNAME}/runs_seed${el}"
  mkdir -pv "$SRGK_RUNS_PATH"
  export SRGK_RUNS_PATH
  #rm -v runs || exit 1
  #ln -vs "runs_${HOSTNAME}/runs_seed${el}" runs

  # Run the experiments in the current runs folder
  "$SRGK_PYEXEC" experiments/train_individual.py base \
    --acts "$AAF_LIKE" --seed "$el" \
    --start_ep 0 --end_ep 100 \
    "$@"
  "$SRGK_PYEXEC" experiments/train_individual.py "$AAF_FAMILY" \
    --acts "$AAF_LIKE" --seed "$el" \
    --start_ep 100 --end_ep 150 --patch_base --tune_aaf \
    "$@"
  "$SRGK_PYEXEC" experiments/train_individual.py "$AAF_FAMILY" \
    --acts "$AAF_LIKE" --seed "$el" \
    --start_ep 0 --end_ep 100 \
    "$@"

  # Visualize the results
  "$SRGK_PYEXEC" post_experiment/show_aaf_form.py
  "$SRGK_PYEXEC" post_experiment/show_progress_charts.py
  "$SRGK_PYEXEC" post_experiment/show_progress_summary.py
done
