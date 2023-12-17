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
  assert(lp_t < np_t)
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
  assert(lp_t < np_t)
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

if __name__ == '__main__':
  test_timing_mod()
  test_timing_sqrt()
  test_timing_assert()
  test_timing_where()
  test_timing_max()
  print("OK!")
