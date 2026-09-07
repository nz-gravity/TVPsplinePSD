import numpy as np
from tv_pspline_psd.inference import (
    _summary_frequency_chunk, surface_summaries, nested_surface_summaries,
)

def test_large_run_chunk_budget():
    chunk = _summary_frequency_chunk(1600, 2046, 256)
    assert chunk*1600*2046*8 <= 128*1024**2
    assert chunk < 256

def test_chunked_intervals_equal_full_reconstruction():
    rng=np.random.default_rng(321)
    bt=rng.normal(size=(7,3)); bf=rng.normal(size=(11,4))
    draws=rng.normal(size=(23,3,4)); g=rng.normal(size=(23,4))
    full=np.einsum('ta,nab,jb->ntj',bt,draws,bf)
    mean,lo,hi=surface_summaries(draws,bt,bf,freq_chunk=2)
    np.testing.assert_allclose(mean,full.mean(axis=0),atol=1e-14)
    np.testing.assert_allclose([lo,hi],np.percentile(full,[5,95],axis=0),atol=1e-14)
    total=full+(g@bf.T)[:,None,:]
    mean,lo,hi=nested_surface_summaries(g,draws,np.ones(23),bt,bf,freq_chunk=2)
    np.testing.assert_allclose(mean,total.mean(axis=0),atol=1e-14)
    np.testing.assert_allclose([lo,hi],np.percentile(total,[5,95],axis=0),atol=1e-14)
