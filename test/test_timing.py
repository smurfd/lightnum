import numpy as np
import lightnum.lightnum as lp
import time

def test_timing_mod():
  start = time.time()
  for _ in range(100000): np.mod([1, 2, 3, 4], [1, 2, 6, 4])
  timer1 = (time.time() - start) * 1000
  print('Numpy mod: {0:.4f}ms'.format(timer1))

  start = time.time()
  for _ in range(100000): lp.mod([1, 2, 3, 4], [1, 2, 6, 4])
  timer2 = (time.time() - start) * 1000
  print('Lightnum mod: {0:.4f}ms\n'.format(timer2))
  assert(timer2 < timer1)
  # Faster than numpy!!!
  # Numpy mod: 147.8453ms
  # Lightnum mod: 121.3479ms

def test_timing_sqrt():
  start = time.time()
  for _ in range(100000): np.sqrt([1, 2, 3, 4])
  timer1 = (time.time() - start) * 1000
  print('Numpy sqrt: {0:.4f}ms'.format(timer1))

  start = time.time()
  for _ in range(100000): lp.sqrt([1, 2, 3, 4])
  timer2 = (time.time() - start) * 1000
  print('Lightnum sqrt: {0:.4f}ms\n'.format(timer2))
  assert(timer2 < timer1)
  # Faster than numpy!!!
  # Numpy sqrt: 93.0297ms
  # Lightnum sqrt: 81.8470ms

def test_timing_assert():
  start = time.time()
  for _ in range(100000): np.testing.assert_equal([1, 2, 3, 4], [1, 2, 3, 4])
  timer1 = (time.time() - start) * 1000
  print('Numpy assert_equal: {0:.4f}ms'.format(timer1))

  start = time.time()
  for _ in range(100000): lp.testing.assert_equal([1, 2, 3, 4], [1, 2, 3, 4])
  timer2 = (time.time() - start) * 1000
  print('Lightnum assert_equal: {0:.4f}ms\n'.format(timer2))
  assert(timer2 < timer1)

test_timing_mod()
test_timing_sqrt()
test_timing_assert()
print("OK!")
