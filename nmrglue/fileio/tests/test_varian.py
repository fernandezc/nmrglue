"""Tests for the Varian file reader and writer."""

import io

import numpy as np
import pytest

from nmrglue.fileio import varian


@pytest.fixture
def stub_fid_writer(monkeypatch):
    """Replace binary-writing helpers to isolate header correction logic."""
    monkeypatch.setattr(
        varian.fileiobase,
        "open_towrite",
        lambda filename, overwrite=False: io.BytesIO(),
    )
    monkeypatch.setattr(varian, "dic2fileheader", lambda dic: None)
    monkeypatch.setattr(varian, "put_fileheader", lambda fileobj, header: None)
    monkeypatch.setattr(varian, "find_dtype", lambda dic: np.dtype("int32"))
    monkeypatch.setattr(varian, "make_blockheader", lambda dic, index: None)
    monkeypatch.setattr(varian, "dic2blockheader", lambda header: [0, 0, 0])
    monkeypatch.setattr(
        varian,
        "put_block",
        lambda fileobj, trace, nbheaders, blockheader: None,
    )


def test_write_fid_lowmem_corrects_header_dimensions(stub_fid_writer):
    """Mismatched header dimensions are corrected by default."""
    dic = {"np": 2, "nblocks": 1, "nbheaders": 1}
    data = np.ones((2, 3), dtype="complex64")

    with pytest.warns(UserWarning) as warnings:
        varian.write_fid_lowmem("test.fid", dic, data)

    assert [str(item.message) for item in warnings] == [
        "data and np size mismatch",
        "data and block size mismatch",
    ]
    assert dic["np"] == 6
    assert dic["nblocks"] == 2


def test_write_fid_lowmem_can_preserve_header_dimensions(stub_fid_writer):
    """Passing correct=False preserves mismatched header dimensions."""
    dic = {"np": 2, "nblocks": 1, "nbheaders": 1}
    data = np.ones((2, 3), dtype="complex64")

    with pytest.warns(UserWarning):
        varian.write_fid_lowmem("test.fid", dic, data, correct=False)

    assert dic["np"] == 2
    assert dic["nblocks"] == 1
