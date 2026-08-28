"""Print the test-missing metrics from one or more TEST_RESULTS json files."""
import json, sys

for path in sys.argv[1:]:
    m = json.load(open(path))["missing"]
    tag = path.split("/out/")[1].split("/")[0] if "/out/" in path else path
    print(f"{tag:>20}  log_loss={m['log_loss']:.4f}  acc={m['accuracy']:.4f}  n={m['n']}")
