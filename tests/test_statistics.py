import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from stat_tests import exact_signed_rank


def test_five_same_direction_pairs_cannot_reach_point_zero_five():
    assert exact_signed_rank([1,2,3,4,5]) == .0625
    assert exact_signed_rank([-1,-2,-3,-4,-5]) == .0625


def test_zero_and_tied_differences():
    assert exact_signed_rank([0,0,0,0,0]) == 1.
    assert exact_signed_rank([1,1,-1,-1,0]) == 1.
