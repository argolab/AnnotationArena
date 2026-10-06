set -e
source /weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/config/cluster.sh >/dev/null 2>&1
rerun_activate_env
R=/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/runs
bind() { cd /weka/scratch/jhu/jeisner1/xwang397/Marformer/imputer/ranking
         for n in DATA RESULTS PLOTS; do ln -sfn "$R/$1/$n" "$n"; done
         cd /weka/scratch/jhu/jeisner1/xwang397/Marformer; }
echo "########## final-dual at chunk 300 (replicate-01's chunk) ##########"
bind final-dual
python RERUN/analysis/smece_decompose.py --device cuda --eval-max-item 300 \
  -o "$R/final-dual/reports/smece_decompose_mi300.json"
echo "########## replicate-01 at chunk 500 (final-dual's chunk) ##########"
bind replicate-01
python RERUN/analysis/smece_decompose.py --device cuda --run replicate-01 --eval-max-item 500 \
  -o "$R/replicate-01/reports/smece_decompose_mi500.json"
