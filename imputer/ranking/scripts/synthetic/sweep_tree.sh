#!/usr/bin/env bash
set -euo pipefail

# Synthetic tree task sweeps for EntityMarformer (depth, width, edges, variants, etc.).
# Architecture is fixed via COMMON (matches train_synthetic.py defaults). Edit COMMON or
# duplicate this script to sweep other Marformer settings.

PY=${PY:-python}
BASE_OUT=${BASE_OUT:-OUTPUT/SYNTHETIC/tree_new}
PLOT_ROOT=${PLOT_ROOT:-${BASE_OUT}/plots}
# COMMON: Marformer defaults + data generation for every sweep:
#   --shuffle-nodes --randomize-tree: varied topologies per seed and no fixed index->target mapping
#   Architecture: shared relational bias (use-per-head-rel off by default), rel-value, add-one attn,
#   feature-only norm, scaled shared-rel, normal type init
COMMON="--task tree --epochs 25 --lr 1e-3 --num-train-graphs 200 --num-test-graphs 50 --seed 42 --embedding-dim 16 --attention-heads 4 \
  --use-rel-value --use-addone-attn --use-feature-only-norm --scale-shared-rel --type-embedding-init normal \
  --shuffle-nodes --randomize-tree"

echo "BASE_OUT=${BASE_OUT}"
echo "PLOT_ROOT=${PLOT_ROOT}"
mkdir -p "${PLOT_ROOT}"

# ###############################################################################
# # Sweep 1: Depth vs Layers grid (core)
# ###############################################################################
# for depth in 1 3 6 9; do
#   for layers in 1 2 3 5; do
#     out="${BASE_OUT}/depth_vs_layers/d${depth}_L${layers}"
#     ${PY} -m imputer.entity_mf.synthetic.train_synthetic \
#       ${COMMON} \
#       --tree-depth "${depth}" --tree-width 3 --num-layers "${layers}" \
#       --aggregate count \
#       --output-dir "${out}"
#     ${PY} -m imputer.entity_mf.synthetic.plot_curves \
#       --run-dir "${out}" \
#       --plot-path "${PLOT_ROOT}/depth_vs_layers_d${depth}_L${layers}.png" \
#       --title "depth=${depth}, layers=${layers}"
#   done
# done

# ###############################################################################
# # Sweep 2: Width scaling (should be flat)
# ###############################################################################
# for width in 2 3 5 8 10; do
#   out="${BASE_OUT}/width/w${width}"
#   ${PY} -m imputer.entity_mf.synthetic.train_synthetic \
#     ${COMMON} \
#     --tree-depth 3 --tree-width "${width}" --num-layers 4 \
#     --aggregate count \
#     --output-dir "${out}"
#   ${PY} -m imputer.entity_mf.synthetic.plot_curves \
#     --run-dir "${out}" \
#     --plot-path "${PLOT_ROOT}/width_w${width}.png" \
#     --title "width=${width}"
# done

# ###############################################################################
# # Sweep 3: Edge direction ablation
# ###############################################################################
# for dir in both c2p p2c; do
#   for depth in 2 3 4 5; do
#     out="${BASE_OUT}/edge_dir/${dir}_d${depth}"
#     ${PY} -m imputer.entity_mf.synthetic.train_synthetic \
#       ${COMMON} \
#       --tree-depth "${depth}" --tree-width 3 --num-layers "${depth}" \
#       --edge-direction "${dir}" --aggregate count \
#       --output-dir "${out}"
#     ${PY} -m imputer.entity_mf.synthetic.plot_curves \
#       --run-dir "${out}" \
#       --plot-path "${PLOT_ROOT}/edge_dir_${dir}_d${depth}.png" \
#       --title "edge_dir=${dir}, depth=${depth}"
#   done
# done

# ###############################################################################
# # Sweep 4: Counting variants (empty-count vs scalar count vs vector sum)
# ###############################################################################
# for variant in empty-count count sum; do
#   out="${BASE_OUT}/variant/${variant}"
#   extra=()
#   if [[ "${variant}" == "empty-count" ]]; then
#     extra+=(--aggregate count --empty-param)
#   elif [[ "${variant}" == "count" ]]; then
#     extra+=(--aggregate count)
#   elif [[ "${variant}" == "sum" ]]; then
#     extra+=(--aggregate sum --param-dim 4)
#   fi
#   ${PY} -m imputer.entity_mf.synthetic.train_synthetic \
#     ${COMMON} \
#     --tree-depth 3 --tree-width 3 --num-layers 4 \
#     "${extra[@]}" \
#     --output-dir "${out}"
#   ${PY} -m imputer.entity_mf.synthetic.plot_curves \
#     --run-dir "${out}" \
#     --plot-path "${PLOT_ROOT}/variant_${variant}.png" \
#     --title "variant=${variant}"
# done

# ###############################################################################
# # Sweep 5: Leaf-only vs all-node (pass-through aggregation vs contribute+relay)
# ###############################################################################
# for mode in all leaf_only; do
#   out="${BASE_OUT}/leaf/${mode}"
#   extra=()
#   if [[ "${mode}" == "leaf_only" ]]; then
#     extra+=(--leaf-only)
#   fi
#   ${PY} -m imputer.entity_mf.synthetic.train_synthetic \
#     ${COMMON} \
#     --tree-depth 3 --tree-width 3 --num-layers 4 \
#     --aggregate sum --param-dim 4 \
#     "${extra[@]}" \
#     --output-dir "${out}"
#   ${PY} -m imputer.entity_mf.synthetic.plot_curves \
#     --run-dir "${out}" \
#     --plot-path "${PLOT_ROOT}/leaf_${mode}.png" \
#     --title "leaf_mode=${mode}"
# done

# ###############################################################################
# # Sweep 6: Forest (disconnected components)
# ###############################################################################
# for T in 1 2 4 8; do
#   out="${BASE_OUT}/forest/T${T}"
#   ${PY} -m imputer.entity_mf.synthetic.train_synthetic \
#     ${COMMON} \
#     --tree-depth 3 --tree-width 3 --num-trees "${T}" --num-layers 4 \
#     --aggregate count \
#     --output-dir "${out}"
#   ${PY} -m imputer.entity_mf.synthetic.plot_curves \
#     --run-dir "${out}" \
#     --plot-path "${PLOT_ROOT}/forest_T${T}.png" \
#     --title "num_trees=${T}"
# done

# ###############################################################################
# # Sweep 7: Param-dim scaling for vector sum
# #
# # For sum tasks, SyntheticRegressionType.param_dim = slices.input_dim + slices.output_dim.
# # In datagen_tree, both are set to cfg.param_dim=D, so global_param_dim = 2*D.
# # EntityMarformer requires embedding_dim > global_param_dim; with D=32 => global_param_dim=64.
# ###############################################################################
# for D in 1 2 4 8 16 32; do
#   out="${BASE_OUT}/param_dim/D${D}"
#   ${PY} -m imputer.entity_mf.synthetic.train_synthetic \
#     ${COMMON} \
#     --embedding-dim 68 \
#     --tree-depth 3 --tree-width 3 --num-layers 4 \
#     --aggregate sum --param-dim "${D}" \
#     --output-dir "${out}"
#   ${PY} -m imputer.entity_mf.synthetic.plot_curves \
#     --run-dir "${out}" \
#     --plot-path "${PLOT_ROOT}/param_dim_D${D}.png" \
#     --title "param_dim=${D}"
# done

###############################################################################
# Sweep 8: Bold width scaling (fixed tree_depth=3, num_layers=3)
###############################################################################
for width in 2 5 8 13 21 34 55; do
  out="${BASE_OUT}/width8/d3_L3/w${width}"
  ${PY} -m imputer.entity_mf.synthetic.train_synthetic \
    ${COMMON} --epochs 50 \
    --tree-depth 3 --tree-width "${width}" --num-layers 3 \
    --aggregate count \
    --output-dir "${out}"
  ${PY} -m imputer.entity_mf.synthetic.plot_curves \
    --run-dir "${out}" \
    --plot-path "${PLOT_ROOT}/sweep8_width_d3_L3_w${width}.png" \
    --title "sweep8: width=${width}, depth=3, layers=3"
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
${PY} "${SCRIPT_DIR}/plot_sweep_tree_summary.py" \
  --base-out "${BASE_OUT}" \
  --plot-root "${PLOT_ROOT}/summary" \
  --log-y \
  --annotate-heatmap

echo "Done."

