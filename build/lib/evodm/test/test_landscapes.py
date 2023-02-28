import pytest
from evodm.landscapes import *

@pytest.fixture
def ls_N3():
    ls_N3 = Landscape(N=3, sigma = 0.5, num_jumps=2)
    return ls_N3

@pytest.fixture
def ls_N4():
    ls_N4 = Landscape(N=4, sigma = 0.5, num_jumps = 2)
    return ls_N4

@pytest.fixture
def ls_N5():
    ls_N5 = Landscape(N=5, sigma = 0.5, num_jumps = 2)
    return ls_N5


def test_define_adjMutN3i0(ls_N3):
    mut = range(ls_N3.N)
    i = 0
    adjmut = ls_N3.define_adjMut(mut = mut, i = i)
    bools = [i in [1,2,3,4,5,6] for i in adjmut]
    bools.append(len(adjmut) == 6)
    assert all(bools)

def test_define_adjMutN3i1(ls_N3):
    mut = range(ls_N3.N)
    i = 1
    adjmut = ls_N3.define_adjMut(mut = mut, i = i)
    bools = [i in [0,3,5,7] for i in adjmut]
    bools.append(len(adjmut) == 4)
    assert all(bools)

def test_define_adjMutN3i2(ls_N3):
    mut = range(ls_N3.N)
    i = 2
    adjmut = ls_N3.define_adjMut(mut = mut, i = i)
    bools = [i in [0,3,6,7] for i in adjmut]
    bools.append(len(adjmut) == 4)
    assert all(bools)

def test_define_adjMutN3i3(ls_N3):
    mut = range(ls_N3.N)
    i = 3
    adjmut = ls_N3.define_adjMut(mut = mut, i = i)
    bools = [i in [0,1,2,7] for i in adjmut]
    bools.append(len(adjmut) == 4)
    assert all(bools)

def test_define_adjMutN3i7(ls_N3):
    mut = range(ls_N3.N)
    i = 7
    adjmut = ls_N3.define_adjMut(mut = mut, i = i)
    bools = [i in [1,2,3,4,5,6] for i in adjmut]
    bools.append(len(adjmut) == 6)
    assert all(bools)

def test_define_adjMutN4i0(ls_N4):
    mut = range(ls_N4.N)
    i = 0
    adjmut = ls_N4.define_adjMut(mut = mut, i = i)
    bools = [i in [1,2,4,8,3,5,6,9,10,12] for i in adjmut]
    bools.append(len(adjmut) == 10)

    assert all(bools)


def test_define_adjMutN4i1(ls_N4):
    mut = range(ls_N4.N)
    i = 1
    adjmut = ls_N4.define_adjMut(mut = mut, i = i)
    bools = [i in [3,5,9, 0, 7, 11, 13] for i in adjmut]
    bools.append(len(adjmut) == 7)
    assert all(bools)

def test_define_adjMutN4i1(ls_N4):
    mut = range(ls_N4.N)
    i = 2
    adjmut = ls_N4.define_adjMut(mut = mut, i = i)
    bools = [i in [0,10, 6, 3, 11, 14, 7] for i in adjmut]
    bools.append(len(adjmut) == 7)
    assert all(bools)

def test_define_adjMutN4i13(ls_N4):
    mut = range(ls_N4.N)
    i = 13
    adjmut = ls_N4.define_adjMut(mut = mut, i = i)
    bools = [i in [9,5,15,12,8,4,1] for i in adjmut]
    bools.append(len(adjmut) == 7)
    assert all(bools)

def test_define_adjMutN5i0(ls_N5):
    mut = range(ls_N5.N)
    i = 0
    adjmut = ls_N5.define_adjMut(mut = mut, i = i)
    bools = [i in [1,2,4,8,16,24,20,18,17,12,10,9,6,5,3] for i in adjmut]
    bools.append(len(adjmut) == 15)
    assert all(bools)


def test_define_adjMutN5i1(ls_N5):
    mut = range(ls_N5.N)
    i = 1
    adjmut = ls_N5.define_adjMut(mut = mut, i = i)
    bools = [i in [0,3,5,9,17, 7,13,11,19,21,25] for i in adjmut]
    bools.append(len(adjmut) == 11)
    assert all(bools)

def test_define_adjMutN5i3(ls_N5):
    mut = range(ls_N5.N)
    i = 3
    adjmut = ls_N5.define_adjMut(mut = mut, i = i)
    bools = [i in [0,7, 1,2,11,19,15,27,23] for i in adjmut]
    bools.append(len(adjmut) == 9)
    assert all(bools)

def test_define_adjMutN5i15(ls_N5):
    mut = range(ls_N5.N)
    i = 15
    adjmut = ls_N5.define_adjMut(mut = mut, i = i)
    assert len(adjmut) == 11

def test_define_adjMutN5i0jump3(ls_N5):
    mut = range(ls_N5.N)
    i=0
    ls_N5.num_jumps = 3
    adjmut = ls_N5.define_adjMut(mut = mut, i = i)
    assert len(adjmut) == 25

def test_define_adjMutN5i1jump3(ls_N5):
    mut = range(ls_N5.N)
    i=1
    ls_N5.num_jumps = 3
    adjmut = ls_N5.define_adjMut(mut = mut, i = i)
    assert len(adjmut) == 15

def test_define_adjMutN5i0jump4(ls_N5):
    mut = range(ls_N5.N)
    i=0
    ls_N5.num_jumps = 4
    adjmut = ls_N5.define_adjMut(mut = mut, i = i)
    assert len(adjmut) == 30


def test_define_adjMutN5i0jump5(ls_N5):
    mut = range(ls_N5.N)
    i=0
    ls_N5.num_jumps = 5
    adjmut = ls_N5.define_adjMut(mut = mut, i = i)
    assert len(adjmut) == 31

def test_find_max_indices():
    ls = Landscape(N=4, sigma = 0.5, num_jumps = 1)