import pyclassify
import sys 
print(sys.path)
from pyclassify import utils
import pytest
def test_distance() :
    x = [1,2,3]
    y = [4,5,6] 
    z = [3,1,4]
    a = utils.distance(x,y)
    b = utils.distance(y,z)
    c = utils.distance(x,z)
    assert a == utils.distance(y,x)
    assert a > 0 
    assert b == utils.distance(z,y)
    assert b > 0 
    assert c == utils.distance(z,x)
    assert c > 0 
    assert (a+b)>c


def test_kNN() : 
    with pytest.raises(TypeError):
        attemptedkNN = pyclassify.kNN( [1,2,3])

def test_majority_vote():
    assert utils.majority_vote([0,0,0,0,0,1,1])==0


