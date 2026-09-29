import sys
from pathlib import Path
import faiss
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_semantic import make_workload, NativeIndex


def test_arrival_order_pairing():
    rng=np.random.default_rng(2)
    pool=dict(vectors=rng.normal(size=(60,8)).astype('float32'),
              queries=rng.normal(size=(40,8)).astype('float32'),
              source_ids=np.arange(60),old_pool=np.arange(30),new_pool=np.arange(30,60),
              new_query_pool=np.arange(40))
    shifted=make_workload(pool,3,'shifted',base=20,batch=5,rounds=2,nq=4)
    shuffled=make_workload(pool,3,'shuffled',base=20,batch=5,rounds=2,nq=4)
    assert sorted(shifted[3]['pool_ids'])==sorted(shuffled[3]['pool_ids'])
    assert np.array_equal(shifted[1],shuffled[1])
    for split in range(2):
        a=np.array(shifted[3]['source_train_ids'])[shifted[2][-1][split]]
        b=np.array(shuffled[3]['source_train_ids'])[shuffled[2][-1][split]]
        assert np.array_equal(a,b)


def test_seeded_hnsw_is_deterministic_and_inserts():
    faiss.omp_set_num_threads(1)
    x=np.random.default_rng(2).normal(size=(300,12)).astype('float32')
    first=NativeIndex(x[:250],'hnsw',4)
    second=NativeIndex(x[:250],'hnsw',4)
    first.index.add(x[250:]); second.index.add(x[250:])
    first.set_budget(64); second.set_budget(64)
    assert np.array_equal(faiss.serialize_index(first.index),faiss.serialize_index(second.index))
    got,ms,count=first.search_timed(x[-1])
    assert got[0]==299 and ms>=0 and count>0


def test_native_flat_exact_top_ten():
    x=np.random.default_rng(5).normal(size=(80,7)).astype('float32')
    idx=NativeIndex(x,'flat',5)
    q=np.ones(7,dtype='float32')
    got,_,count=idx.search_timed(q)
    truth=np.argsort(np.sum((x.astype('float64')-q)**2,axis=1))[:10]
    assert set(got)==set(truth) and count==len(x)
