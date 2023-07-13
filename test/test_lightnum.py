import numpy as np
import lightnum.lightnum as lp 
import random

def test_zeros():
  lp.testing.assert_equal(lp.zeros([4]), np.zeros([4]))
  lp.testing.assert_equal(lp.zeros([3, 3, 3, 3]), np.zeros([3, 3, 3, 3]).ravel())

def test_ones():
  lp.testing.assert_equal(lp.ones([4]), np.ones([4]))
  lp.testing.assert_equal(lp.ones([3, 3, 3, 3]), np.ones([3, 3, 3, 3]).ravel())

def test_zeros_like():
  lp.testing.assert_equal(lp.zeros_like([3, 3, 3, 3]), np.zeros_like([3, 3, 3, 3]))
  lp.testing.assert_equal(lp.zeros_like([3]), np.zeros_like([3]))
  lp.testing.assert_equal(lp.zeros_like([[0, 1, 2],[3, 4, 5]]), np.zeros_like([[0, 1, 2],[3, 4, 5]]).ravel())

def test_zeros_like():
  lp.testing.assert_equal(lp.ones_like([3, 3, 3, 3]), np.ones_like([3, 3, 3, 3]))
  lp.testing.assert_equal(lp.ones_like([3]), np.ones_like([3]))
  lp.testing.assert_equal(lp.ones_like([[0, 1, 2],[3, 4, 5]]), np.ones_like([[0, 1, 2],[3, 4, 5]]).ravel())

def test_array():
  lp.testing.assert_equal(lp.array([3,3,3]).tolist(), np.array([3,3,3]).tolist())

def test_exp_exp2():
  lp.testing.assert_equal(lp.exp([2]), np.exp([2]))
  lp.testing.assert_equal(lp.exp2([2]), np.exp2([2]))

def test_log():
  lp.testing.assert_equal(lp.log([1, 2, 3, 4]), np.log([1, 2, 3, 4]))
  lp.testing.assert_equal(lp.log([1, 2, 6, 4]), np.log([1, 2, 6, 4]))
  lp.testing.assert_equal(lp.log([[1, 2, 3, 4], [1, 2, 3, 4]]), np.log([[1, 2, 3, 4], [1, 2, 3, 4]]).ravel())

def test_sum():
  lp.testing.assert_equal(lp.sum((3,3)), np.sum((3,3)))

def test_max():
  lp.testing.assert_equal(lp.max([[[[2, 2, 3, 4], [1, 2, 3, 4]],[[1, 2, 3, 4], [1, 2, 9, 4]]],[[[1, -1, 3, 4], [1, 2, 3, 4]],[[1, 2, 31, 4], [1, 2, 3, 4]]]]), np.max([[[[2, 2, 3, 4], [1, 2, 3, 4]],[[1, 2, 3, 4], [1, 2, 9, 4]]],[[[1, -1, 3, 4], [1, 2, 3, 4]],[[1, 2, 31, 4], [1, 2, 3, 4]]]]))

def test_min():
  lp.testing.assert_equal(lp.min([[[[2, 2, 3, 4], [1, 2, 3, 4]],[[1, 2, 3, 4], [1, 2, 9, 4]]],[[[1, -1, 3, 4], [1, 2, 3, 4]],[[1, 2, 31, 4], [1, 2, 3, 4]]]]), np.min([[[[2, 2, 3, 4], [1, 2, 3, 4]],[[1, 2, 3, 4], [1, 2, 9, 4]]],[[[1, -1, 3, 4], [1, 2, 3, 4]],[[1, 2, 31, 4], [1, 2, 3, 4]]]]))

def test_maximum():
  lp.testing.assert_equal(lp.maximum([[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]]), np.maximum([[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]]).ravel())

def test_minimum():
  lp.testing.assert_equal(lp.minimum([[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]]), np.minimum([[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]]).ravel())

def test_empty():
  try: lp.testing.assert_equal(lp.empty(6), np.empty(6))
  except AssertionError as e: print("this is not the same since numpy uses random memory value and i force set to 0")
  try: lp.testing.assert_equal(lp.empty((6,6,6)), np.empty((6,6,6)).ravel())
  except AssertionError as e: print("this is not the same since numpy uses random memory value and i force set to 0")

def test_full():
  lp.testing.assert_equal(lp.full(6, 3), np.full(6, 3))
  lp.testing.assert_equal(lp.full((6,6,6), 4), np.full((6,6,6), 4).ravel())

def test_mod():
  lp.testing.assert_equal(lp.mod([1, 2, 3, 4], [1, 2, 6, 4]), np.mod([1, 2, 3, 4], [1, 2, 6, 4]))
  lp.testing.assert_equal(lp.mod([[1, 2, 3, 4], [1, 2, 6, 4]], [[2, 1, 7, 4], [1, 3, 6, 9]]), np.mod([[1, 2, 3, 4], [1, 2, 6, 4]], [[2, 1, 7, 4], [1, 3, 6, 9]]).tolist())

def test_cos():
  lp.testing.assert_equal(lp.cos([1, 2, 3, 4]), np.cos([1, 2, 3, 4]))

def test_sqrt():
  lp.testing.assert_equal(lp.sqrt([1, 2, 3, 4]), np.sqrt([1, 2, 3, 4]))

def test_arctan2():
  lp.testing.assert_equal(lp.arctan2([1, 2, 3, 4], [1, 2, 6, 4]), np.arctan2([1, 2, 3, 4], [1, 2, 6, 4]))

def test_amax():
  lp.testing.assert_equal(lp.amax([1, 2, 3, 4]), np.amax([1, 2, 3, 4]))
  lp.testing.assert_equal(lp.amax([(14, 29, 34), (41, 55, 46), (1, 38, 29), (5, 57, 52)]), np.amax([(14, 29, 34), (41, 55, 46), (1, 38, 29), (5, 57, 52)]))

def test_isin():
  lp.testing.assert_equal(lp.isin([1, 2, 3, 4], [1, 2, 6, 4]), np.isin([1, 2, 3, 4], [1, 2, 6, 4]).tolist())

def test_ceil():
  lp.testing.assert_equal(lp.ceil([1.67, 4.5, 7, 9, 12]), np.ceil([1.67, 4.5, 7, 9, 12]))

def test_reshape():
  lp.testing.assert_equal(lp.reshape([1, 2, 3, 4, 5, 6, 7, 8], (2, 4)), np.reshape([1, 2, 3, 4, 5, 6, 7, 8], (2, 4)).tolist())
  lp.testing.assert_equal(lp.reshape([[1, 2, 3, 4], [5, 6, 7, 8]], -1), np.reshape([[1, 2, 3, 4], [5, 6, 7, 8]], -1))
  lp.testing.assert_equal(lp.reshape([[[1, 2, 3, 4], [5, 6, 7, 8]],[[1, 2, 3, 4], [5, 6, 7, 8]]], -1), np.reshape([[[1, 2, 3, 4], [5, 6, 7, 8]],[[1, 2, 3, 4], [5, 6, 7, 8]]], -1))

def test_show_randomusage():
  np.random.seed(1337)
  lp.random.seed(1337)

  np.random.randn(2,4)
  lp.random.randn(2,4, dtype=lp.float32)
  lp.random.randn((2,4), dtype=lp.float32)

def test_assert_equal():
  lp.testing.assert_equal([3,6,9], [3,6,9])
  try: lp.testing.assert_equal([3,5,9], [3,6,9]) # Assert failure
  except AssertionError as e: print(str(e), ", but that is expected")

  lp.testing.assert_equal([[1, 2, 3, 4], [5, 6, 7, 8]], [[1, 2, 3, 4], [5, 6, 7, 8]])
  try: lp.testing.assert_equal([[1, 2, 3, 4], [6, 6, 7, 8]], [[1, 2, 3, 4], [5, 6, 7, 8]]) # Assert failure
  except AssertionError as e: print(str(e), ", but that is expected")

def test_assert_allclose():
  lp.testing.assert_allclose([3,6,9], [3,6,9])
  lp.testing.assert_allclose([3,5,9], [3,6,9], atol=1, rtol=1)

def test_count_nonzero():
  lp.testing.assert_equal(lp.count_nonzero([[3,6,9], [3,0,9]]), np.count_nonzero([[3,6,9], [3,0,9]]))
  lp.testing.assert_equal(lp.count_nonzero([[3,5,9], [0,0,0]]), np.count_nonzero([[3,5,9], [0,0,0]]))

def test_allclose():
  lp.testing.assert_equal(lp.allclose([1e10,1e-7], [1.00001e10,1e-8]), np.allclose([1e10,1e-7], [1.00001e10,1e-8]))
  lp.testing.assert_equal(lp.allclose([1e10,1e-8], [1.00001e10,1e-9]), np.allclose([1e10,1e-8], [1.00001e10,1e-9]))

def test_cbrt():
  lp.testing.assert_equal(lp.cbrt([1,8,27]), np.cbrt([1,8,27]))

print("OK!")
