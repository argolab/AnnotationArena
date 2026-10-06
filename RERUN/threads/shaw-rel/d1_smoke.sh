set -euo pipefail
RERUN=/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN
SH_ROOT=/weka/scratch/jhu/jeisner1/xwang397/Marformer-shaw/imputer/ranking
source "${RERUN}/config/cluster.sh"
export DOMAIN3_REPO_ROOT_OVERRIDE="${SH_ROOT}" PYTHONPATH="${SH_ROOT}:${PYTHONPATH:-}"
rerun_activate_env
DATA="${RERUN}/runs/dataseed-s101/DATA/STAN/DOMAIN3-SIGMOID"
R="${RERUN}/runs/shaw-smoke"; mkdir -p "${R}/logs"
for arm in bits shaw_k shaw_kv bits_v; do
  case "$arm" in
    bits)    K=bits;    V=none ;;
    shaw_k)  K=learned; V=none ;;
    shaw_kv) K=learned; V=learned ;;
    bits_v)  K=bits;    V=learned ;;
  esac
  echo "=== ${arm} (rel_key=${K} rel_value=${V}) ==="
  env DOMAIN3_DATA_ROOT="${DATA}" DOMAIN3_MAX_ITEM=300 DOMAIN3_SEED=42 \
      DOMAIN3_EPOCHS=4 DOMAIN3_VAL_PROTOCOL=block \
      DOMAIN3_REL_KEY="${K}" DOMAIN3_REL_VALUE="${V}" \
      DOMAIN3_DATASET_LABEL=DOMAIN3-SIGMOID DOMAIN3_SPLIT_PREFIX=DOMAIN3-SIGMOID \
      DOMAIN3_EXP_PREFIX=domain3_sigmoid PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      bash "${SH_ROOT}/scripts/DOMAIN3/run_marformer_group.sh" \
        --family-label Marformer \
        --output-root "${R}/RESULTS/MARFORMER/DOMAIN3-SIGMOID/ITEM/${arm}/Marformer/seed42" \
        --axis item --protocol transductive --sizes 300 \
      > "${R}/logs/${arm}.log" 2>&1 \
    && echo "  trained ok" || { echo "  FAILED -- tail:"; tail -25 "${R}/logs/${arm}.log"; exit 1; }
  cfg=$(find "${R}/RESULTS/MARFORMER/DOMAIN3-SIGMOID/ITEM/${arm}" -name train_config.json | head -1)
  python - "$cfg" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))["model"]
print(f"  train_config: rel_key={m['rel_key']} rel_value={m['rel_value']} rel_init={m['rel_init']}")
PY
done
echo "ALL SMOKE ARMS OK"
