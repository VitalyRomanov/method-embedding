from Dataset import *

def test_compact_property():
    assert compact_property(numpy.array([3,2,1])) == {1:0, 2:1, 3:2}
    assert compact_property(numpy.array([6,4,3,7,8,4])) == {3: 0, 4: 1, 6: 2, 7:3, 8:4}