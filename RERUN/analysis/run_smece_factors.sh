set -e
source /weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/config/cluster.sh >/dev/null 2>&1
rerun_activate_env
cd /weka/scratch/jhu/jeisner1/xwang397/Marformer/imputer/ranking
for n in DATA RESULTS PLOTS; do ln -sfn "/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/runs/final-dual/$n" "$n"; done
cd /weka/scratch/jhu/jeisner1/xwang397/Marformer
python RERUN/analysis/smece_factors.py --device cuda --population corner
