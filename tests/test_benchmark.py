import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from scripts.run_benchmark import evaluate
def test_exact_recall(): assert evaluate('digits','exact',0,0.5)['recall@10'] == 1.0
def test_approximate_metrics_finite():
    r=evaluate('digits','lsh',0,0.5); assert 0 <= r['recall@10'] <= 1 and r['query_ms_p95'] > 0
