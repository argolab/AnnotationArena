"""Print the final missing-rating CE from a training_history.json, or CRASH."""
import json, sys

try:
    h = json.load(open(sys.argv[1]))
    for key in ("combined_eval", "val_eval", "train_eval"):
        m = h[-1].get(key, {}).get("metrics", {}).get("missing", {}).get("rating", {})
        if "xent" in m:
            print(f"{m['xent']:.4f}")
            break
    else:
        print("no-metric")
except Exception:
    print("CRASH")
