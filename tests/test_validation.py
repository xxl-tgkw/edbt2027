import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_validation import workload_ids, exact_truth, NativeFaiss


def test_workload_roles_and_nested_queries():
    base, updates, queries = workload_ids(3)
    assert len(set(np.r_[base, updates.ravel(), queries.ravel()])) == 7300
    assert queries.shape == (5,160)
    assert queries[:,:10].shape == (5,10)
    assert np.array_equal(queries, workload_ids(3)[2])


def test_ground_truth_changes_after_insert():
    base=np.array([[1.,0.],[2.,0.]],dtype='float32')
    update=np.array([[0.,0.]],dtype='float32')
    idx=NativeFaiss(base,'faiss_flat',0)
    query=np.array([0.,0.],dtype='float32')
    assert idx.search(query,k=1)[0] == 0
    idx.add(update)
    assert idx.search(query,k=1)[0] == 2
    assert exact_truth(np.vstack([base,update]),query[None])[0][0] == 2
