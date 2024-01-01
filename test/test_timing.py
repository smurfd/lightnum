import numpy as np
import lightnum.lightnum as lp
import time

def test_timing_mod():
  np_s = time.time()
  for _ in range(100000): np.mod([1, 2, 3, 4], [1, 2, 6, 4])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.mod([1, 2, 3, 4], [1, 2, 6, 4])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy mod: {:.4f}ms Lightnum mod: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))
  #assert(lp_t < np_t) # alot better but still to slow # TODO fast
  # Faster than numpy!!!
  # Numpy mod: 147.8453ms
  # Lightnum mod: 121.3479ms

def test_timing_sqrt():
  np_s = time.time()
  for _ in range(100000): np.sqrt([1, 2, 3, 4])
  np_t = (time.time() - np_s) * 1000; lp_s = time.time()
  for _ in range(100000): lp.sqrt([1, 2, 3, 4])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy sqrt: {:.4f}ms Lightnum sqrt: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))
  #assert(lp_t < np_t) # alot better but still to slow # TODO fast
  # Faster than numpy!!!
  # Numpy sqrt: 93.0297ms
  # Lightnum sqrt: 81.8470ms

def test_timing_assert():
  np_s = time.time()
  for _ in range(100000): np.testing.assert_equal([1, 2, 3, 4], [1, 2, 3, 4])
  np_t = (time.time() - np_s) * 1000; lp_s = time.time()
  for _ in range(100000): lp.testing.assert_equal([1, 2, 3, 4], [1, 2, 3, 4])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy asseq: {:.4f}ms Lightnum asseq: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))
  assert(lp_t < lp_s)
  # Faster than numpy!!!
  # Numpy assert_equal: 2682.8160ms
  # Lightnum assert_equal: 312.7999ms

def test_timing_where():
  np_s = time.time()
  for _ in range(100000): np.where(np.arange(10) < 5, np.arange(10), 10*np.arange(10))
  np_t = (time.time() - np_s) * 1000; lp_s = time.time()
  for _ in range(100000): lp.where([True, True, True, True, True, False, False, False, False, False], lp.arange(10), [ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy where: {:.4f}ms Lightnum where: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))
  # slower
  # Numpy where: 201.0920ms
  # Lightnum where: 215.5871ms

def test_timing_max():
  np_s = time.time()
  for _ in range(100000): np.max([[[[2, 2, 3, 4], [1, 2, 3, 4]],[[1, 2, 3, 4], [1, 2, 9, 4]]],[[[1, -1, 3, 4], [1, 2, 3, 4]],[[1, 2, 31, 4], [1, 2, 3, 4]]]])
  np_t = (time.time() - np_s) * 1000; lp_s = time.time()
  for _ in range(100000): lp.max([[[[2, 2, 3, 4], [1, 2, 3, 4]],[[1, 2, 3, 4], [1, 2, 9, 4]]],[[[1, -1, 3, 4], [1, 2, 3, 4]],[[1, 2, 31, 4], [1, 2, 3, 4]]]])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy max: {:.4f}ms Lightnum max: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))
  # slower
  # Numpy max: 389.6551ms
  # Lightnum max: 604.5718ms

def test_timing_zeros():
  np_s = time.time()
  for _ in range(100000): np.zeros([4])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.zeros([4])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy zeros: {:.4f}ms Lightnum zeros: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_zeros_big():
  np_s = time.time()
  for _ in range(100000): np.zeros([3, 3, 3, 3])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.zeros([3, 3, 3, 3])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy zeros: {:.4f}ms Lightnum zeros: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_ones():
  np_s = time.time()
  for _ in range(100000): np.ones([4])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.ones([4])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy ones: {:.4f}ms Lightnum ones: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_ones_big():
  np_s = time.time()
  for _ in range(100000): np.ones([3, 3, 3, 3])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.ones([3, 3, 3, 3])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy ones: {:.4f}ms Lightnum ones: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_zeros_like():
  np_s = time.time()
  for _ in range(100000): np.zeros_like([4])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.zeros_like([4])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy zeros_like: {:.4f}ms Lightnum zeros_like: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_zeros_like_big():
  np_s = time.time()
  for _ in range(100000): np.zeros_like([3, 3, 3, 3])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.zeros_like([3, 3, 3, 3])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy zeros_like: {:.4f}ms Lightnum zeros_like: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_zeros_like_array():
  np_s = time.time()
  for _ in range(100000): np.zeros_like([[0, 1, 2],[3, 4, 5]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.zeros_like([[0, 1, 2],[3, 4, 5]])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy zeros_like: {:.4f}ms Lightnum zeros_like: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_ones_like():
  np_s = time.time()
  for _ in range(100000): np.ones_like([4])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.ones_like([4])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy ones_like: {:.4f}ms Lightnum ones_like: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_ones_like_big():
  np_s = time.time()
  for _ in range(100000): np.ones_like([3, 3, 3, 3])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.ones_like([3, 3, 3, 3])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy ones_like: {:.4f}ms Lightnum ones_like: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_ones_like_array():
  np_s = time.time()
  for _ in range(100000): np.ones_like([[0, 1, 2],[3, 4, 5]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.ones_like([[0, 1, 2],[3, 4, 5]])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy ones_like: {:.4f}ms Lightnum ones_like: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_array():
  np_s = time.time()
  for _ in range(100000): np.array([3,3,3])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.array([3,3,3])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy array: {:.4f}ms Lightnum array: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_exp():
  np_s = time.time()
  for _ in range(100000): np.exp([2])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.exp([2])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy exp: {:.4f}ms Lightnum exp: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_exp2():
  np_s = time.time()
  for _ in range(100000): np.exp2([2])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.exp2([2])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy exp2: {:.4f}ms Lightnum exp2: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_log():
  np_s = time.time()
  for _ in range(100000): np.log([1, 2, 3, 4])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.log([1, 2, 3, 4])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy log: {:.4f}ms Lightnum log: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_log_array():
  np_s = time.time()
  for _ in range(100000): np.log([[1, 2, 3, 4], [1, 2, 3, 4]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.log([[1, 2, 3, 4], [1, 2, 3, 4]])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy log_array: {:.4f}ms Lightnum log_array: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_sum():
  np_s = time.time()
  for _ in range(100000): np.sum((3,3))
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.sum((3,3))
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy sum: {:.4f}ms Lightnum sum: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_sumf():
  np_s = time.time()
  for _ in range(100000): np.sum((3.3,3.3))
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.sum((3.3,3.3))
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy sumf: {:.4f}ms Lightnum sumf: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_min():
  np_s = time.time()
  for _ in range(100000): np.min([[[[2, 2, 3, 4], [1, 2, 3, 4]],[[1, 2, 3, 4], [1, 2, 9, 4]]],[[[1, -1, 3, 4], [1, 2, 3, 4]],[[1, 2, 31, 4], [1, 2, 3, 4]]]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.min([[[[2, 2, 3, 4], [1, 2, 3, 4]],[[1, 2, 3, 4], [1, 2, 9, 4]]],[[[1, -1, 3, 4], [1, 2, 3, 4]],[[1, 2, 31, 4], [1, 2, 3, 4]]]])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy min: {:.4f}ms Lightnum min: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_maximum():
  np_s = time.time()
  for _ in range(100000): np.maximum([[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.maximum([[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy maximum: {:.4f}ms Lightnum maximum: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_minimum():
  np_s = time.time()
  for _ in range(100000): np.minimum([[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.minimum([[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy minimum: {:.4f}ms Lightnum minimum: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_full():
  np_s = time.time()
  for _ in range(100000): np.full(6, 3)
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.full(6, 3)
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy full: {:.4f}ms Lightnum full: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_full_tuple():
  np_s = time.time()
  for _ in range(100000): np.full((6,6,6), 4)
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.full((6,6,6), 4)
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy full_tuple: {:.4f}ms Lightnum full_tuple: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_modf():
  np_s = time.time()
  for _ in range(100000): np.mod([1.1, 2.2, 3.3, 4.4], [1.1, 2.2, 6.6, 4.4])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.mod([1.1, 2.2, 3.3, 4.4], [1.1, 2.2, 6.6, 4.4])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy modf: {:.4f}ms Lightnum modf: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_mod_array():
  np_s = time.time()
  for _ in range(100000): np.mod([[1, 2, 3, 4], [1, 2, 6, 4]], [[2, 1, 7, 4], [1, 3, 6, 9]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.mod([[1, 2, 3, 4], [1, 2, 6, 4]], [[2, 1, 7, 4], [1, 3, 6, 9]])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy mod_array: {:.4f}ms Lightnum mod_array: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_cos():
  np_s = time.time()
  for _ in range(100000): np.cos([1, 2, 3, 4])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.cos([1, 2, 3, 4])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy cos: {:.4f}ms Lightnum cos: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_arctan2():
  np_s = time.time()
  for _ in range(100000): np.arctan2([1, 2, 3, 4], [1, 2, 6, 4])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.arctan2([1, 2, 3, 4], [1, 2, 6, 4])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy arctan2: {:.4f}ms Lightnum arctan2: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_amax():
  np_s = time.time()
  for _ in range(100000): np.amax([1, 2, 3, 4])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.amax([1, 2, 3, 4])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy amax: {:.4f}ms Lightnum amax: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_amax_array():
  np_s = time.time()
  for _ in range(100000): np.amax([(14, 29, 34), (41, 55, 46), (1, 38, 29), (5, 57, 52)])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.amax([(14, 29, 34), (41, 55, 46), (1, 38, 29), (5, 57, 52)])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy amax_array: {:.4f}ms Lightnum amax_array: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_isin():
  np_s = time.time()
  for _ in range(100000): np.isin([1, 2, 3, 4], [1, 2, 6, 4])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.isin([1, 2, 3, 4], [1, 2, 6, 4])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy isin: {:.4f}ms Lightnum isin: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_ceil():
  np_s = time.time()
  for _ in range(100000): np.ceil([1.67, 4.5, 7, 9, 12])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.ceil([1.67, 4.5, 7, 9, 12])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy ceil: {:.4f}ms Lightnum ceil: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_reshape():
  np_s = time.time()
  for _ in range(100000): np.reshape([1, 2, 3, 4, 5, 6, 7, 8], (2, 4))
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.reshape([1, 2, 3, 4, 5, 6, 7, 8], (2, 4))
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy reshape: {:.4f}ms Lightnum reshape: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_reshape_flat():
  np_s = time.time()
  for _ in range(100000): np.reshape([[1, 2, 3, 4], [5, 6, 7, 8]], -1)
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.reshape([[1, 2, 3, 4], [5, 6, 7, 8]], -1)
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy reshape_flat: {:.4f}ms Lightnum reshape_flat: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_reshape_flat_array():
  np_s = time.time()
  for _ in range(100000): np.reshape([[[1, 2, 3, 4], [5, 6, 7, 8]],[[1, 2, 3, 4], [5, 6, 7, 8]]], -1)
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.reshape([[[1, 2, 3, 4], [5, 6, 7, 8]],[[1, 2, 3, 4], [5, 6, 7, 8]]], -1)
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy reshape_flat_array: {:.4f}ms Lightnum reshape_flat_array: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_count_zero():
  np_s = time.time()
  for _ in range(100000): np.count_nonzero([[3,6,9], [3,0,9]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.count_nonzero([[3,6,9], [3,0,9]])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy count_nonzero: {:.4f}ms Lightnum count_nonzero: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_count_zero2():
  np_s = time.time()
  for _ in range(100000): np.count_nonzero([[3,5,9], [0,0,0]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.count_nonzero([[3,5,9], [0,0,0]])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy count_nonzero2: {:.4f}ms Lightnum count_nonzero2: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_allclose():
  np_s = time.time()
  for _ in range(100000): np.allclose([1e10,1e-7], [1.00001e10,1e-8])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.allclose([1e10,1e-7], [1.00001e10,1e-8])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy allclose: {:.4f}ms Lightnum allclose: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_allclose2():
  np_s = time.time()
  for _ in range(100000): np.allclose([1e10,1e-8], [1.00001e10,1e-9])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.allclose([1e10,1e-8], [1.00001e10,1e-9])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy allclose2: {:.4f}ms Lightnum allclose2: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_cbrt():
  np_s = time.time()
  for _ in range(100000): np.cbrt([1,8,27])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.cbrt([1,8,27])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy cbrt: {:.4f}ms Lightnum cbrt: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_copy():
  np_s = time.time()
  for _ in range(100000): np.copy([[1,2,3,4],[5,6,7,8]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.copy([[1,2,3,4],[5,6,7,8]])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy copy: {:.4f}ms Lightnum copy: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_median():
  np_s = time.time()
  for _ in range(100000): np.median([[10, 7, 4], [3, 2, 1]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.median([[10, 7, 4], [3, 2, 1]])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy median: {:.4f}ms Lightnum median: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_arange():
  np_s = time.time()
  for _ in range(100000): np.arange(3, 7)
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.arange(3, 7)
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy arange: {:.4f}ms Lightnum arange: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_flip():
  np_s = time.time()
  for _ in range(100000): np.flip([1,2,3,4,5,6])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.flip([1,2,3,4,5,6])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy flip: {:.4f}ms Lightnum flip: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_split():
  np_s = time.time()
  for _ in range(100000): np.split(np.arange(6), 2)
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.split(lp.arange(6), 2)
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy split: {:.4f}ms Lightnum split: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_tile():
  np_s = time.time()
  for _ in range(100000): np.tile([0,1,2,3,4,5], 2)
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.tile([0,1,2,3,4,5], 2)
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy tile: {:.4f}ms Lightnum tile: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_concatenate():
  np_s = time.time()
  for _ in range(100000): np.concatenate(([1,2,3,4],[4,5,6]))
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.concatenate(([1,2,3,4],[4,5,6]))
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy concatenate: {:.4f}ms Lightnum concatenate: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_cumsum():
  np_s = time.time()
  for _ in range(100000): np.cumsum([[1,2,3], [4,5,6]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.cumsum([[1,2,3], [4,5,6]])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy cumsum: {:.4f}ms Lightnum cumsum: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_matmul():
  np_s = time.time()
  for _ in range(100000): np.matmul([[1, 0], [0, 1]], [[4, 1], [2, 2]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.matmul([[1, 0], [0, 1]], [[4, 1], [2, 2]])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy matmul: {:.4f}ms Lightnum matmul: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_broadcast_to():
  np_s = time.time()
  for _ in range(100000): np.broadcast_to([1, 2, 3], (3, 3))
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.broadcast_to([1, 2, 3], (3, 3))
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy broadcast_to: {:.4f}ms Lightnum broadcast_to: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_outer():
  np_s = time.time()
  for _ in range(100000): np.outer([[1,2,3,4],[5,6,7,8]], [[9,10,11,12],[13,14,15,16]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.outer([[1,2,3,4],[5,6,7,8]], [[9,10,11,12],[13,14,15,16]])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy outer: {:.4f}ms Lightnum outer: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_eye():
  np_s = time.time()
  for _ in range(100000): np.eye(4,4,k=-1)
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.eye(4,4,k=-1)
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy eye: {:.4f}ms Lightnum eye: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_expand_dims():
  np_s = time.time()
  for _ in range(100000): np.expand_dims([[1,2,3,4],[5,6,7,8]], 2)
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.expand_dims([[1,2,3,4],[5,6,7,8]], 2)
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy expand_dims: {:.4f}ms Lightnum expand_dims: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_argmax():
  np_s = time.time()
  for _ in range(100000): np.argmax([[1,2,3,4],[5,6,7,8]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.argmax([[1,2,3,4],[5,6,7,8]])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy argmax: {:.4f}ms Lightnum argmax: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_argmax_axis():
  np_s = time.time()
  for _ in range(100000): np.argmax([[1,2,3,4],[5,6,7,8]], axis=1)
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.argmax([[1,2,3,4],[5,6,7,8]], axis=1)
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy argmax_axis: {:.4f}ms Lightnum argmax_axis: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_transpose():
  np_s = time.time()
  for _ in range(100000): np.transpose([[1,2,3,4],[5,6,7,8]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.transpose([[1,2,3,4],[5,6,7,8]])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy transpose: {:.4f}ms Lightnum transpose: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_stack():
  np_s = time.time()
  for _ in range(100000): np.stack([[1,2,3,4],[5,6,7,8]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.stack([[1,2,3,4],[5,6,7,8]])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy stack: {:.4f}ms Lightnum stack: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_vstack():
  np_s = time.time()
  for _ in range(100000): np.vstack([[1,2,3,4],[5,6,7,8]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.vstack([[1,2,3,4],[5,6,7,8]])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy vstack: {:.4f}ms Lightnum vstack: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_nonzero():
  np_s = time.time()
  for _ in range(100000): np.nonzero([[1,2,3,0],[5,6,7,8]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.nonzero([[1,2,3,0],[5,6,7,8]])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy nonzero: {:.4f}ms Lightnum nonzero: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_squeeze():
  np_s = time.time()
  for _ in range(100000): np.squeeze([[[0, 1, 2], [3, 4, 5], [6, 7, 8]]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.squeeze([[[0, 1, 2], [3, 4, 5], [6, 7, 8]]])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy squeeze: {:.4f}ms Lightnum squeeze: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_clip():
  np_s = time.time()
  for _ in range(100000): np.clip([[1, 2, 3, 4], [5, 6, 7, 8]], 2, 5)
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.clip([[1, 2, 3, 4], [5, 6, 7, 8]], 2, 5)
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy clip: {:.4f}ms Lightnum clip: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_unique():
  np_s = time.time()
  for _ in range(100000): np.unique([[[0, 1, 0, 2], [3, 4, 5, 2], [6, 7, 8, 4]]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.unique([[[0, 1, 0, 2], [3, 4, 5, 2], [6, 7, 8, 4]]])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy unique: {:.4f}ms Lightnum unique: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_triu():
  np_s = time.time()
  for _ in range(100000): np.triu([[1,2,3],[1,2,3],[1,2,3]], 1)
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.triu([[1,2,3],[1,2,3],[1,2,3]], 1)
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy triu: {:.4f}ms Lightnum triu: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_meshgrid():
  np_s = time.time()
  for _ in range(100000): npx,npy=np.meshgrid([1,2,3], [4,5,6])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lpx,lpy=lp.meshgrid([1,2,3], [4,5,6])
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy meshgrid: {:.4f}ms Lightnum meshgrid: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_newaxis():
  np_s = time.time()
  for _ in range(100000): x = np.array([[1,2,3],[1,2,3]]);x[np.newaxis, :]
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.newaxis([[1,2,3],[1,2,3]], 1)
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy newaxis: {:.4f}ms Lightnum newaxis: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_frombuffer():
  np_s = time.time()
  for _ in range(100000): np.frombuffer(b'smurfd', np.uint8)
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.frombuffer(b'smurfd', lp.uint8)
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy frombuffer: {:.4f}ms Lightnum frombuffer: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

def test_timing_promote_types():
  np_s = time.time()
  for _ in range(100000): np.promote_types(lp.uint8, lp.uint32)
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.promote_types(lp.uint8, lp.uint32)
  lp_t = (time.time() - lp_s) * 1000
  print('[{}] Numpy promote_types: {:.4f}ms Lightnum promote_types: {:.4f}ms'.format(np_t > lp_t, np_t, lp_t))

if __name__ == '__main__':
  test_timing_mod()
  test_timing_sqrt()
  test_timing_assert()
  test_timing_where()
  test_timing_max()
  test_timing_zeros()
  test_timing_zeros_big()
  test_timing_ones()
  test_timing_ones_big()
  test_timing_zeros_like()
  test_timing_zeros_like_array()
  test_timing_ones_like()
  test_timing_ones_like_big()
  test_timing_ones_like_array()
  test_timing_array()
  test_timing_exp()
  test_timing_exp2()
  test_timing_log()
  test_timing_log_array()
  test_timing_sum()
  test_timing_sumf()
  test_timing_min()
  test_timing_maximum()
  test_timing_minimum()
  test_timing_full()
  test_timing_full_tuple()
  test_timing_modf()
  test_timing_mod_array()
  test_timing_cos()
  test_timing_arctan2()
  test_timing_amax()
  test_timing_amax_array()
  test_timing_isin()
  test_timing_ceil()
  test_timing_reshape()
  test_timing_reshape_flat()
  test_timing_reshape_flat_array()
  test_timing_count_zero()
  test_timing_count_zero2()
  test_timing_allclose()
  test_timing_allclose2()
  test_timing_cbrt()
  test_timing_copy()
  test_timing_median()
  test_timing_arange()
  test_timing_flip()
  test_timing_split()
  test_timing_tile()
  test_timing_concatenate()
  test_timing_cumsum()
  test_timing_matmul()
  test_timing_broadcast_to()
  test_timing_outer()
  test_timing_eye()
  test_timing_expand_dims()
  test_timing_argmax()
  test_timing_argmax_axis()
  test_timing_transpose()
  test_timing_stack()
  test_timing_vstack()
  test_timing_nonzero()
  test_timing_squeeze()
  test_timing_clip()
  test_timing_unique()
  test_timing_triu()
  test_timing_meshgrid()
  test_timing_newaxis()
  test_timing_frombuffer()
  test_timing_promote_types()
  print("OK!")

"""
# 3.12.1
Numpy mod: 217.2108ms Lightnum mod: 112.6053ms
Numpy sqrt: 99.0338ms Lightnum sqrt: 75.4073ms
Numpy asseq: 2011.5540ms Lightnum asseq: 239.5749ms
Numpy where: 189.6169ms Lightnum where: 139.0600ms
Numpy max: 396.0202ms Lightnum max: 393.2891ms
# 3.8.18
Numpy mod: 155.5903ms Lightnum mod: 159.7881ms
Numpy sqrt: 69.6929ms Lightnum sqrt: 115.0711ms
Numpy asseq: 2525.5029ms Lightnum asseq: 250.3948ms
Numpy where: 194.2251ms Lightnum where: 221.5528ms
Numpy max: 372.7498ms Lightnum max: 569.0811ms
# 3.10.13
Numpy mod: 158.6509ms Lightnum mod: 167.8360ms
Numpy sqrt: 94.6679ms Lightnum sqrt: 125.6351ms
Numpy asseq: 2197.3391ms Lightnum asseq: 249.8381ms
Numpy where: 184.4869ms Lightnum where: 222.3091ms
Numpy max: 386.4939ms Lightnum max: 584.4240ms
# 3.11.7
Numpy mod: 165.9558ms Lightnum mod: 119.1120ms
Numpy sqrt: 95.0959ms Lightnum sqrt: 80.1196ms
Numpy asseq: 1933.2290ms Lightnum asseq: 197.1989ms
Numpy where: 188.3349ms Lightnum where: 176.9741ms
Numpy max: 388.0529ms Lightnum max: 412.5881ms
"""

"""
# 3.8.18
[False] Numpy mod: 155.8857ms Lightnum mod: 174.8359ms
[False] Numpy sqrt: 75.2032ms Lightnum sqrt: 122.9270ms
[True] Numpy asseq: 2711.1790ms Lightnum asseq: 268.5161ms
[False] Numpy where: 203.7611ms Lightnum where: 238.7490ms
[False] Numpy max: 398.2902ms Lightnum max: 608.9921ms
[False] Numpy zeros: 11.5957ms Lightnum zeros: 132.7372ms
[False] Numpy zeros: 14.7903ms Lightnum zeros: 131.9587ms
[False] Numpy ones: 68.6829ms Lightnum ones: 184.5911ms
[False] Numpy ones: 72.8891ms Lightnum ones: 184.3662ms
[True] Numpy zeros_like: 143.8820ms Lightnum zeros_like: 78.6710ms
[True] Numpy zeros_like: 173.2199ms Lightnum zeros_like: 125.7820ms
[True] Numpy ones_like: 136.7123ms Lightnum ones_like: 76.9947ms
[True] Numpy ones_like: 142.9889ms Lightnum ones_like: 78.4171ms
[True] Numpy ones_like: 166.8339ms Lightnum ones_like: 124.7420ms
[True] Numpy array: 20.2610ms Lightnum array: 15.9488ms
[False] Numpy exp: 67.5669ms Lightnum exp: 84.0149ms
[False] Numpy exp2: 66.3183ms Lightnum exp2: 94.5609ms
[False] Numpy log: 74.3680ms Lightnum log: 125.8922ms
[False] Numpy log_array: 122.2351ms Lightnum log_array: 213.9490ms
[True] Numpy sum: 178.8290ms Lightnum sum: 67.8730ms
[True] Numpy sumf: 176.9071ms Lightnum sumf: 69.6490ms
[False] Numpy min: 395.7818ms Lightnum min: 587.7318ms
[False] Numpy maximum: 160.9232ms Lightnum maximum: 392.0009ms
[False] Numpy minimum: 161.1228ms Lightnum minimum: 389.6880ms
[False] Numpy full: 64.5809ms Lightnum full: 121.9273ms
[False] Numpy full_tuple: 71.6619ms Lightnum full_tuple: 1416.4591ms
[False] Numpy modf: 104.3429ms Lightnum modf: 157.3431ms
[False] Numpy mod_array: 162.1730ms Lightnum mod_array: 286.3388ms
[False] Numpy cos: 74.9910ms Lightnum cos: 124.7740ms
[False] Numpy arctan2: 136.2109ms Lightnum arctan2: 158.7331ms
[True] Numpy amax: 182.0769ms Lightnum amax: 110.5902ms
[False] Numpy amax_array: 238.9190ms Lightnum amax_array: 270.8321ms
[True] Numpy isin: 1448.6721ms Lightnum isin: 232.7952ms
[False] Numpy ceil: 67.1961ms Lightnum ceil: 139.2441ms
[True] Numpy reshape: 123.6389ms Lightnum reshape: 118.6888ms
[False] Numpy reshape_flat: 142.0639ms Lightnum reshape_flat: 181.1271ms
[False] Numpy reshape_flat_array: 201.5591ms Lightnum reshape_flat_array: 419.7972ms
[False] Numpy count_nonzero: 75.6023ms Lightnum count_nonzero: 129.8649ms
[False] Numpy count_nonzero2: 76.4189ms Lightnum count_nonzero2: 126.7478ms
[True] Numpy allclose: 1090.0052ms Lightnum allclose: 270.7860ms
[True] Numpy allclose2: 1087.0240ms Lightnum allclose2: 263.4990ms
[False] Numpy cbrt: 74.1611ms Lightnum cbrt: 153.7218ms
[False] Numpy copy: 75.4828ms Lightnum copy: 279.3353ms
[True] Numpy median: 634.4740ms Lightnum median: 147.6188ms
[False] Numpy arange: 21.3697ms Lightnum arange: 72.3603ms
[True] Numpy flip: 70.5721ms Lightnum flip: 70.3030ms
[True] Numpy split: 404.5238ms Lightnum split: 355.6170ms
[True] Numpy tile: 200.9149ms Lightnum tile: 143.1491ms
[True] Numpy concatenate: 87.3339ms Lightnum concatenate: 47.1241ms
[False] Numpy cumsum: 187.2370ms Lightnum cumsum: 346.4518ms
[False] Numpy matmul: 174.7730ms Lightnum matmul: 505.2211ms
[True] Numpy broadcast_to: 186.4452ms Lightnum broadcast_to: 36.0210ms
[False] Numpy outer: 225.1151ms Lightnum outer: 557.7450ms
[False] Numpy eye: 62.8102ms Lightnum eye: 143.0268ms
[False] Numpy expand_dims: 178.6220ms Lightnum expand_dims: 206.5001ms
[False] Numpy argmax: 172.5972ms Lightnum argmax: 207.4831ms
[True] Numpy argmax_axis: 174.9299ms Lightnum argmax_axis: 79.0169ms
[True] Numpy transpose: 125.0939ms Lightnum transpose: 89.0000ms
[True] Numpy stack: 251.9240ms Lightnum stack: 27.1690ms
[True] Numpy vstack: 231.4010ms Lightnum vstack: 26.2380ms
[False] Numpy nonzero: 140.1560ms Lightnum nonzero: 255.1160ms
[True] Numpy squeeze: 159.6859ms Lightnum squeeze: 50.1072ms
[True] Numpy clip: 519.7151ms Lightnum clip: 163.7456ms
[False] Numpy unique: 295.4640ms Lightnum unique: 355.2420ms
[True] Numpy triu: 359.9141ms Lightnum triu: 118.8838ms
[True] Numpy meshgrid: 759.0890ms Lightnum meshgrid: 548.7521ms
[True] Numpy newaxis: 58.3360ms Lightnum newaxis: 36.3681ms
[False] Numpy frombuffer: 18.1580ms Lightnum frombuffer: 67.8012ms
[False] Numpy promote_types: 300.0209ms Lightnum promote_types: 320.0409ms

# 3.12.1
[True] Numpy mod: 230.2213ms Lightnum mod: 117.5480ms
[True] Numpy sqrt: 99.6008ms Lightnum sqrt: 79.3762ms
[True] Numpy asseq: 2093.8570ms Lightnum asseq: 261.2739ms
[True] Numpy where: 195.2121ms Lightnum where: 147.3529ms
[False] Numpy max: 415.4010ms Lightnum max: 417.7480ms
[False] Numpy zeros: 12.5320ms Lightnum zeros: 94.1300ms
[False] Numpy zeros: 14.5099ms Lightnum zeros: 92.8979ms
[False] Numpy ones: 55.3379ms Lightnum ones: 131.5689ms
[False] Numpy ones: 59.1400ms Lightnum ones: 130.0311ms
[True] Numpy zeros_like: 84.7259ms Lightnum zeros_like: 55.9411ms
[True] Numpy zeros_like: 119.9322ms Lightnum zeros_like: 78.2499ms
[True] Numpy ones_like: 78.3811ms Lightnum ones_like: 55.7160ms
[True] Numpy ones_like: 85.7489ms Lightnum ones_like: 56.5860ms
[True] Numpy ones_like: 114.6021ms Lightnum ones_like: 77.7071ms
[True] Numpy array: 22.5039ms Lightnum array: 12.8210ms
[True] Numpy exp: 95.2294ms Lightnum exp: 58.2647ms
[True] Numpy exp2: 92.9151ms Lightnum exp2: 57.5550ms
[True] Numpy log: 102.6580ms Lightnum log: 76.8449ms
[True] Numpy log_array: 151.5102ms Lightnum log_array: 126.3611ms
[True] Numpy sum: 178.2589ms Lightnum sum: 48.4722ms
[True] Numpy sumf: 176.3730ms Lightnum sumf: 50.2343ms
[True] Numpy min: 417.0964ms Lightnum min: 413.1720ms
[False] Numpy maximum: 222.5649ms Lightnum maximum: 387.3870ms
[False] Numpy minimum: 223.8660ms Lightnum minimum: 383.6250ms
[False] Numpy full: 50.2028ms Lightnum full: 90.4551ms
[False] Numpy full_tuple: 58.1613ms Lightnum full_tuple: 1038.3990ms
[True] Numpy modf: 156.8971ms Lightnum modf: 118.4969ms
[True] Numpy mod_array: 224.3230ms Lightnum mod_array: 221.9081ms
[True] Numpy cos: 102.2608ms Lightnum cos: 83.1761ms
[True] Numpy arctan2: 187.4959ms Lightnum arctan2: 121.6571ms
[True] Numpy amax: 184.9649ms Lightnum amax: 78.3322ms
[True] Numpy amax_array: 243.3131ms Lightnum amax_array: 195.4248ms
[True] Numpy isin: 1197.6879ms Lightnum isin: 215.4741ms
[False] Numpy ceil: 91.9199ms Lightnum ceil: 92.8550ms
[True] Numpy reshape: 126.0517ms Lightnum reshape: 67.0180ms
[True] Numpy reshape_flat: 147.4030ms Lightnum reshape_flat: 122.9577ms
[False] Numpy reshape_flat_array: 210.7940ms Lightnum reshape_flat_array: 286.7851ms
[False] Numpy count_nonzero: 64.5962ms Lightnum count_nonzero: 87.5258ms
[False] Numpy count_nonzero2: 64.8041ms Lightnum count_nonzero2: 87.6050ms
[True] Numpy allclose: 908.7381ms Lightnum allclose: 236.7270ms
[True] Numpy allclose2: 903.8949ms Lightnum allclose2: 218.8370ms
[False] Numpy cbrt: 100.7290ms Lightnum cbrt: 119.3018ms
[False] Numpy copy: 69.2298ms Lightnum copy: 165.5350ms
[True] Numpy median: 502.7359ms Lightnum median: 91.7559ms
[False] Numpy arange: 24.3723ms Lightnum arange: 38.4157ms
[True] Numpy flip: 55.5589ms Lightnum flip: 53.7741ms
[True] Numpy split: 327.6112ms Lightnum split: 206.4090ms
[True] Numpy tile: 201.6542ms Lightnum tile: 92.9940ms
[True] Numpy concatenate: 82.5069ms Lightnum concatenate: 26.5820ms
[False] Numpy cumsum: 200.4020ms Lightnum cumsum: 242.6240ms
[False] Numpy matmul: 232.4910ms Lightnum matmul: 348.3641ms
[True] Numpy broadcast_to: 167.0630ms Lightnum broadcast_to: 16.8509ms
[False] Numpy outer: 229.0981ms Lightnum outer: 350.9388ms
[False] Numpy eye: 54.6749ms Lightnum eye: 62.5479ms
[True] Numpy expand_dims: 131.4700ms Lightnum expand_dims: 89.9880ms
[True] Numpy argmax: 178.2310ms Lightnum argmax: 141.2442ms
[True] Numpy argmax_axis: 174.0441ms Lightnum argmax_axis: 52.5651ms
[True] Numpy transpose: 131.6271ms Lightnum transpose: 43.7429ms
[True] Numpy stack: 181.1671ms Lightnum stack: 17.2031ms
[True] Numpy vstack: 169.5602ms Lightnum vstack: 17.0498ms
[True] Numpy nonzero: 149.4930ms Lightnum nonzero: 140.5139ms
[True] Numpy squeeze: 178.1471ms Lightnum squeeze: 27.9961ms
[True] Numpy clip: 233.2160ms Lightnum clip: 78.1019ms
[True] Numpy unique: 282.9480ms Lightnum unique: 247.2942ms
[True] Numpy triu: 314.7509ms Lightnum triu: 59.4840ms
[True] Numpy meshgrid: 701.7579ms Lightnum meshgrid: 399.4441ms
[True] Numpy newaxis: 63.7310ms Lightnum newaxis: 29.3360ms
[False] Numpy frombuffer: 22.1970ms Lightnum frombuffer: 36.0579ms
[False] Numpy promote_types: 299.9942ms Lightnum promote_types: 300.9729ms
"""
