""" Unit tests for nmrglue/fileio/fileiobase.py module """

import os

import numpy as np
from numpy.testing import assert_array_equal
import nmrglue as ng
import pytest


# Test data.
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
NMRPIPE_1D_FREQ = os.path.join(DATA_DIR, 'nmrpipe_1d_freq.fid')


def test_uc_from_freqscale():
    """
    Test that `uc_from_freqscale` gives equivalent results as `uc_from_udic`.
    """
    from nmrglue.fileio.fileiobase import uc_from_freqscale

    # read frequency test data
    dic, data = ng.pipe.read(NMRPIPE_1D_FREQ)

    # make udic and uc using uc_from_udic
    udic = ng.pipe.guess_udic(dic, data)
    uc = ng.fileiobase.uc_from_udic(udic)

    ppm_scale = uc.ppm_scale()
    uc_from_ppm = uc_from_freqscale(ppm_scale, udic[0]['obs'], 'ppm')
    new_ppm_scale = uc_from_ppm.ppm_scale()
    assert_array_equal(ppm_scale, new_ppm_scale)

    hz_scale = uc.hz_scale()
    uc_from_hz = uc_from_freqscale(hz_scale, udic[0]['obs'], 'hz')
    new_hz_scale = uc_from_hz.hz_scale()
    assert_array_equal(hz_scale, new_hz_scale)

    khz_scale = hz_scale * 1.0e3
    uc_from_khz = uc_from_freqscale(khz_scale, udic[0]['obs'], 'khz')
    new_khz_scale = uc_from_khz.hz_scale() * 1.0e3
    assert_array_equal(khz_scale, new_khz_scale)


# regression test for https://github.com/jjhelmus/nmrglue/issues/113
def test_uc_with_float_size():
    uc = ng.fileiobase.unit_conversion(
        size=64.0, cplx=False, sw=1.0, obs=1.0, car=1.0)
    scale = uc.ppm_scale()
    assert len(scale) == 64


def test_update_uc():
    uc = ng.fileiobase.unit_conversion(
        size=64.0, cplx=False, sw=1.0, obs=1.0, car=1.0)
    uc2 = ng.fileiobase.update_uc(uc, size=10, cplx=True, sw=2.0, car=5.2, obs=3.0)
    assert uc2._size == 10
    assert uc2._cplx is True
    assert abs(uc2._sw - 2.0) < 1e-5
    assert abs(uc2._car - 5.2) < 1e-5
    assert abs(uc2._obs - 3.0) < 1e-5


class DummyDataND(ng.fileiobase.data_nd):
    """
    Minimal data_nd subclass backed by an in-memory array.
    """

    def __init__(self, fshape, order=None):
        self.fshape = tuple(fshape)
        if order is None:
            order = range(len(self.fshape))
        self.order = tuple(order)
        self.cplex = False
        self.big = True
        self.__setdimandshape__()

    def __fgetitem__(self, slices):
        return np.arange(np.prod(self.fshape)).reshape(self.fshape)[slices]

    def __fcopy__(self, order):
        return DummyDataND(self.fshape, order)


# regression test for https://github.com/jjhelmus/nmrglue/issues/286
def test_data_nd_transpose_1d():
    """
    Test that transpose is accepted on one-dimensional data.
    """
    data = DummyDataND(fshape=(5,))
    assert data.transpose().shape == (5,)
    assert data.transpose(0).shape == (5,)
    assert data.transpose(np.int64(0)).shape == (5,)
    assert data.transpose((0,)).shape == (5,)
    assert_array_equal(data.transpose()[:], np.arange(5))


# regression test for https://github.com/jjhelmus/nmrglue/issues/287
@pytest.mark.parametrize("axis", [0.5, 0.0])
def test_data_nd_rejects_float_axes(axis):
    """
    Test that float axis arguments are rejected, as numpy does.
    """
    data = DummyDataND(fshape=(2, 3))
    with pytest.raises(TypeError):
        data.swapaxes(axis, 1)
    with pytest.raises(TypeError):
        data.swapaxes(1, axis)
    with pytest.raises(TypeError):
        data.transpose(axis, 1)
    with pytest.raises(TypeError):
        data.transpose(1, axis)


def test_data_nd_transpose_2d():
    """
    Test that two-dimensional transpose is unchanged.
    """
    data = DummyDataND(fshape=(3, 4))
    expected = np.arange(12).reshape(3, 4)
    assert_array_equal(data.transpose()[:], expected.transpose())
    assert_array_equal(data.transpose(1, 0)[:], expected.transpose(1, 0))
    assert_array_equal(data.transpose((1, 0))[:], expected.transpose(1, 0))
