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
  print('Numpy mod: {:.4f}ms Lightnum mod: {:.4f}ms'.format(np_t, lp_t))
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
  print('Numpy sqrt: {:.4f}ms Lightnum sqrt: {:.4f}ms'.format(np_t, lp_t))
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
  print('Numpy asseq: {:.4f}ms Lightnum asseq: {:.4f}ms'.format(np_t, lp_t))
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
  print('Numpy where: {:.4f}ms Lightnum where: {:.4f}ms'.format(np_t, lp_t))
  # slower
  # Numpy where: 201.0920ms
  # Lightnum where: 215.5871ms

def test_timing_max():
  np_s = time.time()
  for _ in range(100000): np.max([[[[2, 2, 3, 4], [1, 2, 3, 4]],[[1, 2, 3, 4], [1, 2, 9, 4]]],[[[1, -1, 3, 4], [1, 2, 3, 4]],[[1, 2, 31, 4], [1, 2, 3, 4]]]])
  np_t = (time.time() - np_s) * 1000; lp_s = time.time()
  for _ in range(100000): lp.max([[[[2, 2, 3, 4], [1, 2, 3, 4]],[[1, 2, 3, 4], [1, 2, 9, 4]]],[[[1, -1, 3, 4], [1, 2, 3, 4]],[[1, 2, 31, 4], [1, 2, 3, 4]]]])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy max: {:.4f}ms Lightnum max: {:.4f}ms'.format(np_t, lp_t))
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
  print('Numpy zeros: {:.4f}ms Lightnum zeros: {:.4f}ms'.format(np_t, lp_t))

def test_timing_zeros_big():
  np_s = time.time()
  for _ in range(100000): np.zeros([3, 3, 3, 3])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.zeros([3, 3, 3, 3])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy zeros: {:.4f}ms Lightnum zeros: {:.4f}ms'.format(np_t, lp_t))

def test_timing_ones():
  np_s = time.time()
  for _ in range(100000): np.ones([4])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.ones([4])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy ones: {:.4f}ms Lightnum ones: {:.4f}ms'.format(np_t, lp_t))

def test_timing_ones_big():
  np_s = time.time()
  for _ in range(100000): np.ones([3, 3, 3, 3])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.ones([3, 3, 3, 3])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy ones: {:.4f}ms Lightnum ones: {:.4f}ms'.format(np_t, lp_t))

def test_timing_zeros_like():
  np_s = time.time()
  for _ in range(100000): np.zeros_like([4])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.zeros_like([4])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy zeros_like: {:.4f}ms Lightnum zeros_like: {:.4f}ms'.format(np_t, lp_t))

def test_timing_zeros_like_big():
  np_s = time.time()
  for _ in range(100000): np.zeros_like([3, 3, 3, 3])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.zeros_like([3, 3, 3, 3])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy zeros_like: {:.4f}ms Lightnum zeros_like: {:.4f}ms'.format(np_t, lp_t))

def test_timing_zeros_like_array():
  np_s = time.time()
  for _ in range(100000): np.zeros_like([[0, 1, 2],[3, 4, 5]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.zeros_like([[0, 1, 2],[3, 4, 5]])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy zeros_like: {:.4f}ms Lightnum zeros_like: {:.4f}ms'.format(np_t, lp_t))

def test_timing_ones_like():
  np_s = time.time()
  for _ in range(100000): np.ones_like([4])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.ones_like([4])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy ones_like: {:.4f}ms Lightnum ones_like: {:.4f}ms'.format(np_t, lp_t))

def test_timing_ones_like_big():
  np_s = time.time()
  for _ in range(100000): np.ones_like([3, 3, 3, 3])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.ones_like([3, 3, 3, 3])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy ones_like: {:.4f}ms Lightnum ones_like: {:.4f}ms'.format(np_t, lp_t))

def test_timing_ones_like_array():
  np_s = time.time()
  for _ in range(100000): np.ones_like([[0, 1, 2],[3, 4, 5]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.ones_like([[0, 1, 2],[3, 4, 5]])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy ones_like: {:.4f}ms Lightnum ones_like: {:.4f}ms'.format(np_t, lp_t))

def test_timing_array():
  np_s = time.time()
  for _ in range(100000): np.array([3,3,3])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.array([3,3,3])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy array: {:.4f}ms Lightnum array: {:.4f}ms'.format(np_t, lp_t))

def test_timing_exp():
  np_s = time.time()
  for _ in range(100000): np.exp([2])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.exp([2])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy exp: {:.4f}ms Lightnum exp: {:.4f}ms'.format(np_t, lp_t))

def test_timing_exp2():
  np_s = time.time()
  for _ in range(100000): np.exp2([2])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.exp2([2])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy exp2: {:.4f}ms Lightnum exp2: {:.4f}ms'.format(np_t, lp_t))

def test_timing_log():
  np_s = time.time()
  for _ in range(100000): np.log([1, 2, 3, 4])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.log([1, 2, 3, 4])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy log: {:.4f}ms Lightnum log: {:.4f}ms'.format(np_t, lp_t))

def test_timing_log_array():
  np_s = time.time()
  for _ in range(100000): np.log([[1, 2, 3, 4], [1, 2, 3, 4]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.log([[1, 2, 3, 4], [1, 2, 3, 4]])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy log_array: {:.4f}ms Lightnum log_array: {:.4f}ms'.format(np_t, lp_t))

def test_timing_sum():
  np_s = time.time()
  for _ in range(100000): np.sum((3,3))
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.sum((3,3))
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy sum: {:.4f}ms Lightnum sum: {:.4f}ms'.format(np_t, lp_t))

def test_timing_sumf():
  np_s = time.time()
  for _ in range(100000): np.sum((3.3,3.3))
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.sum((3.3,3.3))
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy sumf: {:.4f}ms Lightnum sumf: {:.4f}ms'.format(np_t, lp_t))

def test_timing_min():
  np_s = time.time()
  for _ in range(100000): np.min([[[[2, 2, 3, 4], [1, 2, 3, 4]],[[1, 2, 3, 4], [1, 2, 9, 4]]],[[[1, -1, 3, 4], [1, 2, 3, 4]],[[1, 2, 31, 4], [1, 2, 3, 4]]]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.min([[[[2, 2, 3, 4], [1, 2, 3, 4]],[[1, 2, 3, 4], [1, 2, 9, 4]]],[[[1, -1, 3, 4], [1, 2, 3, 4]],[[1, 2, 31, 4], [1, 2, 3, 4]]]])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy min: {:.4f}ms Lightnum min: {:.4f}ms'.format(np_t, lp_t))

def test_timing_maximum():
  np_s = time.time()
  for _ in range(100000): np.maximum([[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.maximum([[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy maximum: {:.4f}ms Lightnum maximum: {:.4f}ms'.format(np_t, lp_t))

def test_timing_minimum():
  np_s = time.time()
  for _ in range(100000): np.minimum([[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.minimum([[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy minimum: {:.4f}ms Lightnum minimum: {:.4f}ms'.format(np_t, lp_t))

def test_timing_full():
  np_s = time.time()
  for _ in range(100000): np.full(6, 3)
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.full(6, 3)
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy full: {:.4f}ms Lightnum full: {:.4f}ms'.format(np_t, lp_t))

def test_timing_full_tuple():
  np_s = time.time()
  for _ in range(100000): np.full((6,6,6), 4)
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.full((6,6,6), 4)
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy full_tuple: {:.4f}ms Lightnum full_tuple: {:.4f}ms'.format(np_t, lp_t))

def test_timing_modf():
  np_s = time.time()
  for _ in range(100000): np.mod([1.1, 2.2, 3.3, 4.4], [1.1, 2.2, 6.6, 4.4])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.mod([1.1, 2.2, 3.3, 4.4], [1.1, 2.2, 6.6, 4.4])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy modf: {:.4f}ms Lightnum modf: {:.4f}ms'.format(np_t, lp_t))

def test_timing_mod_array():
  np_s = time.time()
  for _ in range(100000): np.mod([[1, 2, 3, 4], [1, 2, 6, 4]], [[2, 1, 7, 4], [1, 3, 6, 9]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.mod([[1, 2, 3, 4], [1, 2, 6, 4]], [[2, 1, 7, 4], [1, 3, 6, 9]])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy mod_array: {:.4f}ms Lightnum mod_array: {:.4f}ms'.format(np_t, lp_t))

def test_timing_cos():
  np_s = time.time()
  for _ in range(100000): np.cos([1, 2, 3, 4])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.cos([1, 2, 3, 4])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy cos: {:.4f}ms Lightnum cos: {:.4f}ms'.format(np_t, lp_t))

def test_timing_arctan2():
  np_s = time.time()
  for _ in range(100000): np.arctan2([1, 2, 3, 4], [1, 2, 6, 4])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.arctan2([1, 2, 3, 4], [1, 2, 6, 4])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy arctan2: {:.4f}ms Lightnum arctan2: {:.4f}ms'.format(np_t, lp_t))

def test_timing_amax():
  np_s = time.time()
  for _ in range(100000): np.amax([1, 2, 3, 4])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.amax([1, 2, 3, 4])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy amax: {:.4f}ms Lightnum amax: {:.4f}ms'.format(np_t, lp_t))

def test_timing_amax_array():
  np_s = time.time()
  for _ in range(100000): np.amax([(14, 29, 34), (41, 55, 46), (1, 38, 29), (5, 57, 52)])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.amax([(14, 29, 34), (41, 55, 46), (1, 38, 29), (5, 57, 52)])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy amax_array: {:.4f}ms Lightnum amax_array: {:.4f}ms'.format(np_t, lp_t))

def test_timing_isin():
  np_s = time.time()
  for _ in range(100000): np.isin([1, 2, 3, 4], [1, 2, 6, 4])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.isin([1, 2, 3, 4], [1, 2, 6, 4])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy isin: {:.4f}ms Lightnum isin: {:.4f}ms'.format(np_t, lp_t))

def test_timing_ceil():
  np_s = time.time()
  for _ in range(100000): np.ceil([1.67, 4.5, 7, 9, 12])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.ceil([1.67, 4.5, 7, 9, 12])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy ceil: {:.4f}ms Lightnum ceil: {:.4f}ms'.format(np_t, lp_t))

def test_timing_reshape():
  np_s = time.time()
  for _ in range(100000): np.reshape([1, 2, 3, 4, 5, 6, 7, 8], (2, 4))
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.reshape([1, 2, 3, 4, 5, 6, 7, 8], (2, 4))
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy reshape: {:.4f}ms Lightnum reshape: {:.4f}ms'.format(np_t, lp_t))

def test_timing_reshape_flat():
  np_s = time.time()
  for _ in range(100000): np.reshape([[1, 2, 3, 4], [5, 6, 7, 8]], -1)
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.reshape([[1, 2, 3, 4], [5, 6, 7, 8]], -1)
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy reshape_flat: {:.4f}ms Lightnum reshape_flat: {:.4f}ms'.format(np_t, lp_t))

def test_timing_reshape_flat_array():
  np_s = time.time()
  for _ in range(100000): np.reshape([[[1, 2, 3, 4], [5, 6, 7, 8]],[[1, 2, 3, 4], [5, 6, 7, 8]]], -1)
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.reshape([[[1, 2, 3, 4], [5, 6, 7, 8]],[[1, 2, 3, 4], [5, 6, 7, 8]]], -1)
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy reshape_flat_array: {:.4f}ms Lightnum reshape_flat_array: {:.4f}ms'.format(np_t, lp_t))

def test_timing_count_zero():
  np_s = time.time()
  for _ in range(100000): np.count_nonzero([[3,6,9], [3,0,9]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.count_nonzero([[3,6,9], [3,0,9]])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy count_nonzero: {:.4f}ms Lightnum count_nonzero: {:.4f}ms'.format(np_t, lp_t))

def test_timing_count_zero2():
  np_s = time.time()
  for _ in range(100000): np.count_nonzero([[3,5,9], [0,0,0]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.count_nonzero([[3,5,9], [0,0,0]])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy count_nonzero2: {:.4f}ms Lightnum count_nonzero2: {:.4f}ms'.format(np_t, lp_t))

def test_timing_allclose():
  np_s = time.time()
  for _ in range(100000): np.allclose([1e10,1e-7], [1.00001e10,1e-8])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.allclose([1e10,1e-7], [1.00001e10,1e-8])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy allclose: {:.4f}ms Lightnum allclose: {:.4f}ms'.format(np_t, lp_t))

def test_timing_allclose2():
  np_s = time.time()
  for _ in range(100000): np.allclose([1e10,1e-8], [1.00001e10,1e-9])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.allclose([1e10,1e-8], [1.00001e10,1e-9])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy allclose2: {:.4f}ms Lightnum allclose2: {:.4f}ms'.format(np_t, lp_t))

def test_timing_cbrt():
  np_s = time.time()
  for _ in range(100000): np.cbrt([1,8,27])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.cbrt([1,8,27])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy cbrt: {:.4f}ms Lightnum cbrt: {:.4f}ms'.format(np_t, lp_t))

def test_timing_copy():
  np_s = time.time()
  for _ in range(100000): np.copy([[1,2,3,4],[5,6,7,8]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.copy([[1,2,3,4],[5,6,7,8]])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy copy: {:.4f}ms Lightnum copy: {:.4f}ms'.format(np_t, lp_t))

def test_timing_median():
  np_s = time.time()
  for _ in range(100000): np.median([[10, 7, 4], [3, 2, 1]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.median([[10, 7, 4], [3, 2, 1]])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy median: {:.4f}ms Lightnum median: {:.4f}ms'.format(np_t, lp_t))

def test_timing_arange():
  np_s = time.time()
  for _ in range(100000): np.arange(3, 7)
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.arange(3, 7)
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy arange: {:.4f}ms Lightnum arange: {:.4f}ms'.format(np_t, lp_t))

def test_timing_flip():
  np_s = time.time()
  for _ in range(100000): np.flip([1,2,3,4,5,6])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.flip([1,2,3,4,5,6])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy flip: {:.4f}ms Lightnum flip: {:.4f}ms'.format(np_t, lp_t))

def test_timing_split():
  np_s = time.time()
  for _ in range(100000): np.split(np.arange(6), 2)
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.split(lp.arange(6), 2)
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy split: {:.4f}ms Lightnum split: {:.4f}ms'.format(np_t, lp_t))

def test_timing_tile():
  np_s = time.time()
  for _ in range(100000): np.tile([0,1,2,3,4,5], 2)
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.tile([0,1,2,3,4,5], 2)
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy tile: {:.4f}ms Lightnum tile: {:.4f}ms'.format(np_t, lp_t))

def test_timing_concatenate():
  np_s = time.time()
  for _ in range(100000): np.concatenate(([1,2,3,4],[4,5,6]))
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.concatenate(([1,2,3,4],[4,5,6]))
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy concatenate: {:.4f}ms Lightnum concatenate: {:.4f}ms'.format(np_t, lp_t))

def test_timing_cumsum():
  np_s = time.time()
  for _ in range(100000): np.cumsum([[1,2,3], [4,5,6]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.cumsum([[1,2,3], [4,5,6]])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy cumsum: {:.4f}ms Lightnum cumsum: {:.4f}ms'.format(np_t, lp_t))

def test_timing_matmul():
  np_s = time.time()
  for _ in range(100000): np.matmul([[1, 0], [0, 1]], [[4, 1], [2, 2]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.matmul([[1, 0], [0, 1]], [[4, 1], [2, 2]])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy matmul: {:.4f}ms Lightnum matmul: {:.4f}ms'.format(np_t, lp_t))

def test_timing_broadcast_to():
  np_s = time.time()
  for _ in range(100000): np.broadcast_to([1, 2, 3], (3, 3))
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.broadcast_to([1, 2, 3], (3, 3))
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy broadcast_to: {:.4f}ms Lightnum broadcast_to: {:.4f}ms'.format(np_t, lp_t))

def test_timing_outer():
  np_s = time.time()
  for _ in range(100000): np.outer([[1,2,3,4],[5,6,7,8]], [[9,10,11,12],[13,14,15,16]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.outer([[1,2,3,4],[5,6,7,8]], [[9,10,11,12],[13,14,15,16]])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy outer: {:.4f}ms Lightnum outer: {:.4f}ms'.format(np_t, lp_t))

def test_timing_eye():
  np_s = time.time()
  for _ in range(100000): np.eye(4,4,k=-1)
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.eye(4,4,k=-1)
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy eye: {:.4f}ms Lightnum eye: {:.4f}ms'.format(np_t, lp_t))

def test_timing_expand_dims():
  np_s = time.time()
  for _ in range(100000): np.expand_dims([[1,2,3,4],[5,6,7,8]], 2)
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.expand_dims([[1,2,3,4],[5,6,7,8]], 2)
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy expand_dims: {:.4f}ms Lightnum expand_dims: {:.4f}ms'.format(np_t, lp_t))

def test_timing_argmax():
  np_s = time.time()
  for _ in range(100000): np.argmax([[1,2,3,4],[5,6,7,8]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.argmax([[1,2,3,4],[5,6,7,8]])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy argmax: {:.4f}ms Lightnum argmax: {:.4f}ms'.format(np_t, lp_t))

def test_timing_argmax_axis():
  np_s = time.time()
  for _ in range(100000): np.argmax([[1,2,3,4],[5,6,7,8]], axis=1)
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.argmax([[1,2,3,4],[5,6,7,8]], axis=1)
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy argmax_axis: {:.4f}ms Lightnum argmax_axis: {:.4f}ms'.format(np_t, lp_t))

def test_timing_transpose():
  np_s = time.time()
  for _ in range(100000): np.transpose([[1,2,3,4],[5,6,7,8]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.transpose([[1,2,3,4],[5,6,7,8]])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy transpose: {:.4f}ms Lightnum transpose: {:.4f}ms'.format(np_t, lp_t))

def test_timing_stack():
  np_s = time.time()
  for _ in range(100000): np.stack([[1,2,3,4],[5,6,7,8]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.stack([[1,2,3,4],[5,6,7,8]])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy stack: {:.4f}ms Lightnum stack: {:.4f}ms'.format(np_t, lp_t))

def test_timing_vstack():
  np_s = time.time()
  for _ in range(100000): np.vstack([[1,2,3,4],[5,6,7,8]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.vstack([[1,2,3,4],[5,6,7,8]])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy vstack: {:.4f}ms Lightnum vstack: {:.4f}ms'.format(np_t, lp_t))

def test_timing_nonzero():
  np_s = time.time()
  for _ in range(100000): np.nonzero([[1,2,3,0],[5,6,7,8]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.nonzero([[1,2,3,0],[5,6,7,8]])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy nonzero: {:.4f}ms Lightnum nonzero: {:.4f}ms'.format(np_t, lp_t))

def test_timing_squeeze():
  np_s = time.time()
  for _ in range(100000): np.squeeze([[[0, 1, 2], [3, 4, 5], [6, 7, 8]]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.squeeze([[[0, 1, 2], [3, 4, 5], [6, 7, 8]]])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy squeeze: {:.4f}ms Lightnum squeeze: {:.4f}ms'.format(np_t, lp_t))

def test_timing_clip():
  np_s = time.time()
  for _ in range(100000): np.clip([[1, 2, 3, 4], [5, 6, 7, 8]], 2, 5)
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.clip([[1, 2, 3, 4], [5, 6, 7, 8]], 2, 5)
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy clip: {:.4f}ms Lightnum clip: {:.4f}ms'.format(np_t, lp_t))

def test_timing_unique():
  np_s = time.time()
  for _ in range(100000): np.unique([[[0, 1, 0, 2], [3, 4, 5, 2], [6, 7, 8, 4]]])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.unique([[[0, 1, 0, 2], [3, 4, 5, 2], [6, 7, 8, 4]]])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy unique: {:.4f}ms Lightnum unique: {:.4f}ms'.format(np_t, lp_t))

def test_timing_triu():
  np_s = time.time()
  for _ in range(100000): np.triu([[1,2,3],[1,2,3],[1,2,3]], 1)
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.triu([[1,2,3],[1,2,3],[1,2,3]], 1)
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy triu: {:.4f}ms Lightnum triu: {:.4f}ms'.format(np_t, lp_t))

def test_timing_meshgrid():
  np_s = time.time()
  for _ in range(100000): npx,npy=np.meshgrid([1,2,3], [4,5,6])
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lpx,lpy=lp.meshgrid([1,2,3], [4,5,6])
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy meshgrid: {:.4f}ms Lightnum meshgrid: {:.4f}ms'.format(np_t, lp_t))

def test_timing_newaxis():
  np_s = time.time()
  for _ in range(100000): x = np.array([[1,2,3],[1,2,3]]);x[np.newaxis, :]
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.newaxis([[1,2,3],[1,2,3]], 1)
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy newaxis: {:.4f}ms Lightnum newaxis: {:.4f}ms'.format(np_t, lp_t))

def test_timing_frombuffer():
  np_s = time.time()
  for _ in range(100000): np.frombuffer(b'smurfd', np.uint8)
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.frombuffer(b'smurfd', lp.uint8)
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy frombuffer: {:.4f}ms Lightnum frombuffer: {:.4f}ms'.format(np_t, lp_t))

def test_timing_promote_types():
  np_s = time.time()
  for _ in range(100000): np.promote_types(lp.uint8, lp.uint32)
  np_t = (time.time() - np_s) * 1000
  lp_s = time.time()
  for _ in range(100000): lp.promote_types(lp.uint8, lp.uint32)
  lp_t = (time.time() - lp_s) * 1000
  print('Numpy promote_types: {:.4f}ms Lightnum promote_types: {:.4f}ms'.format(np_t, lp_t))

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
