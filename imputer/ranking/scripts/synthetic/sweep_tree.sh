#!/usr/bin/env bash
set -euo pipefail

# Checkpoint 1 sweep battery for EntityMarformer synthetic tree tasks.
#
# Each sweep group targets one hypothesis (see plan file for details).

PY=${PY:-python}
BASE_OUT=${BASE_OUT:-OUTPUT/SYNTHETIC/tree}
PLOT_ROOT=${PLOT_ROOT:-${BASE_OUT}/plots}
COMMON="--task tree --epochs 25 --lr 1e-3 --num-train-graphs 200 --num-test-graphs 50 --seed 42 --embedding-dim 16 --attention-heads 4"

echo "BASE_OUT=${BASE_OUT}"
echo "PLOT_ROOT=${PLOT_ROOT}"
mkdir -p "${PLOT_ROOT}"

###############################################################################
# Sweep 1: Depth vs Layers grid (core)
###############################################################################
for depth in 1 2 3 4 5 6; do
  for layers in 1 2 3 4 5 6; do
    out="${BASE_OUT}/depth_vs_layers/d${depth}_L${layers}"
    ${PY} -m imputer.entity_mf.synthetic.train_synthetic \
      ${COMMON} \
      --tree-depth "${depth}" --tree-width 3 --num-layers "${layers}" \
      --aggregate count \
      --output-dir "${out}"
    ${PY} -m imputer.entity_mf.synthetic.plot_curves \
      --run-dir "${out}" \
      --plot-path "${PLOT_ROOT}/depth_vs_layers_d${depth}_L${layers}.png" \
      --title "depth=${depth}, layers=${layers}"
  done
done

###############################################################################
# Sweep 2: Width scaling (should be flat)
###############################################################################
for width in 2 3 5 8 10; do
  out="${BASE_OUT}/width/w${width}"
  ${PY} -m imputer.entity_mf.synthetic.train_synthetic \
    ${COMMON} \
    --tree-depth 3 --tree-width "${width}" --num-layers 4 \
    --aggregate count \
    --output-dir "${out}"
  ${PY} -m imputer.entity_mf.synthetic.plot_curves \
    --run-dir "${out}" \
    --plot-path "${PLOT_ROOT}/width_w${width}.png" \
    --title "width=${width}"
done

###############################################################################
# Sweep 3: Edge direction ablation
###############################################################################
for dir in both c2p p2c; do
  for depth in 2 3 4 5; do
    out="${BASE_OUT}/edge_dir/${dir}_d${depth}"
    ${PY} -m imputer.entity_mf.synthetic.train_synthetic \
      ${COMMON} \
      --tree-depth "${depth}" --tree-width 3 --num-layers "${depth}" \
      --edge-direction "${dir}" --aggregate count \
      --output-dir "${out}"
    ${PY} -m imputer.entity_mf.synthetic.plot_curves \
      --run-dir "${out}" \
      --plot-path "${PLOT_ROOT}/edge_dir_${dir}_d${depth}.png" \
      --title "edge_dir=${dir}, depth=${depth}"
  done
done

###############################################################################
# Sweep 4: Counting variants (empty-count vs scalar count vs vector sum)
###############################################################################
for variant in empty-count count sum; do
  out="${BASE_OUT}/variant/${variant}"
  extra=()
  if [[ "${variant}" == "empty-count" ]]; then
    extra+=(--aggregate count --empty-param)
  elif [[ "${variant}" == "count" ]]; then
    extra+=(--aggregate count)
  elif [[ "${variant}" == "sum" ]]; then
    extra+=(--aggregate sum --param-dim 4)
  fi
  ${PY} -m imputer.entity_mf.synthetic.train_synthetic \
    ${COMMON} \
    --tree-depth 3 --tree-width 3 --num-layers 4 \
    "${extra[@]}" \
    --output-dir "${out}"
  ${PY} -m imputer.entity_mf.synthetic.plot_curves \
    --run-dir "${out}" \
    --plot-path "${PLOT_ROOT}/variant_${variant}.png" \
    --title "variant=${variant}"
done

###############################################################################
# Sweep 5: Leaf-only vs all-node (pass-through aggregation vs contribute+relay)
###############################################################################
for mode in all leaf_only; do
  out="${BASE_OUT}/leaf/${mode}"
  extra=()
  if [[ "${mode}" == "leaf_only" ]]; then
    extra+=(--leaf-only)
  fi
  ${PY} -m imputer.entity_mf.synthetic.train_synthetic \
    ${COMMON} \
    --tree-depth 3 --tree-width 3 --num-layers 4 \
    --aggregate sum --param-dim 4 \
    "${extra[@]}" \
    --output-dir "${out}"
  ${PY} -m imputer.entity_mf.synthetic.plot_curves \
    --run-dir "${out}" \
    --plot-path "${PLOT_ROOT}/leaf_${mode}.png" \
    --title "leaf_mode=${mode}"
done

###############################################################################
# Sweep 6: Forest (disconnected components)
###############################################################################
for T in 1 2 4 8; do
  out="${BASE_OUT}/forest/T${T}"
  ${PY} -m imputer.entity_mf.synthetic.train_synthetic \
    ${COMMON} \
    --tree-depth 3 --tree-width 3 --num-trees "${T}" --num-layers 4 \
    --aggregate count \
    --output-dir "${out}"
  ${PY} -m imputer.entity_mf.synthetic.plot_curves \
    --run-dir "${out}" \
    --plot-path "${PLOT_ROOT}/forest_T${T}.png" \
    --title "num_trees=${T}"
done

###############################################################################
# Sweep 7: Param-dim scaling for vector sum
###############################################################################
for D in 1 2 4 8 16 32; do
  out="${BASE_OUT}/param_dim/D${D}"
  ${PY} -m imputer.entity_mf.synthetic.train_synthetic \
    ${COMMON} \
    --tree-depth 3 --tree-width 3 --num-layers 4 \
    --aggregate sum --param-dim "${D}" \
    --output-dir "${out}"
  ${PY} -m imputer.entity_mf.synthetic.plot_curves \
    --run-dir "${out}" \
    --plot-path "${PLOT_ROOT}/param_dim_D${D}.png" \
    --title "param_dim=${D}"
done

echo "Done."

