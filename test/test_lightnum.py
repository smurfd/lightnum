import numpy as np
import lightnum.lightnum as lp

def test_zeros(): lp.testing.assert_equal(lp.zeros([4]), np.zeros([4]))
def test_zeros_big(): lp.testing.assert_equal(lp.zeros([3, 3, 3, 3]), np.zeros([3, 3, 3, 3]).ravel())
def test_ones(): lp.testing.assert_equal(lp.ones([4]), np.ones([4]))
def test_ones_big(): lp.testing.assert_equal(lp.ones([3, 3, 3, 3]), np.ones([3, 3, 3, 3]).ravel())
def test_zeros_like(): lp.testing.assert_equal(lp.zeros_like([3]), np.zeros_like([3]))
def test_zeros_like_big(): lp.testing.assert_equal(lp.zeros_like([3, 3, 3, 3]), np.zeros_like([3, 3, 3, 3]))
def test_zeros_like_array(): lp.testing.assert_equal(lp.zeros_like([[0, 1, 2],[3, 4, 5]]), np.zeros_like([[0, 1, 2],[3, 4, 5]]).ravel())
def test_ones_like(): lp.testing.assert_equal(lp.ones_like([3]), np.ones_like([3]))
def test_ones_like_big(): lp.testing.assert_equal(lp.ones_like([3, 3, 3, 3]), np.ones_like([3, 3, 3, 3]))
def test_ones_like_array():lp.testing.assert_equal(lp.ones_like([[0, 1, 2],[3, 4, 5]]), np.ones_like([[0, 1, 2],[3, 4, 5]]).ravel())
def test_array(): lp.testing.assert_equal(lp.array([3,3,3]).tolist(), np.array([3,3,3]).tolist())
def test_exp_exp(): lp.testing.assert_equal(lp.exp([2]), np.exp([2]))
def test_exp_exp2(): lp.testing.assert_equal(lp.exp2([2]), np.exp2([2]))
def test_log(): lp.testing.assert_equal(lp.log([1, 2, 3, 4]), np.log([1, 2, 3, 4]))
def test_log_array(): lp.testing.assert_equal(lp.log([[1, 2, 3, 4], [1, 2, 3, 4]]), np.log([[1, 2, 3, 4], [1, 2, 3, 4]]).ravel())
def test_sum(): lp.testing.assert_equal(lp.sum((3,3)), np.sum((3,3)))
def test_sumf(): lp.testing.assert_equal(lp.sum((3.3,3.3)), np.sum((3.3,3.3)))
def test_max(): lp.testing.assert_equal(lp.max([[[[2, 2, 3, 4], [1, 2, 3, 4]],[[1, 2, 3, 4], [1, 2, 9, 4]]],[[[1, -1, 3, 4], [1, 2, 3, 4]],[[1, 2, 31, 4], [1, 2, 3, 4]]]]), np.max([[[[2, 2, 3, 4], [1, 2, 3, 4]],[[1, 2, 3, 4], [1, 2, 9, 4]]],[[[1, -1, 3, 4], [1, 2, 3, 4]],[[1, 2, 31, 4], [1, 2, 3, 4]]]]))
def test_min(): lp.testing.assert_equal(lp.min([[[[2, 2, 3, 4], [1, 2, 3, 4]],[[1, 2, 3, 4], [1, 2, 9, 4]]],[[[1, -1, 3, 4], [1, 2, 3, 4]],[[1, 2, 31, 4], [1, 2, 3, 4]]]]), np.min([[[[2, 2, 3, 4], [1, 2, 3, 4]],[[1, 2, 3, 4], [1, 2, 9, 4]]],[[[1, -1, 3, 4], [1, 2, 3, 4]],[[1, 2, 31, 4], [1, 2, 3, 4]]]]))
def test_maximum(): lp.testing.assert_equal(lp.maximum([[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]]), np.maximum([[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]]).ravel())
def test_minimum(): lp.testing.assert_equal(lp.minimum([[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]]), np.minimum([[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]]).ravel())
def test_full(): lp.testing.assert_equal(lp.full(6, 3), np.full(6, 3))
def test_full_tuple(): lp.testing.assert_equal(lp.full((6,6,6), 4), np.full((6,6,6), 4).ravel())
def test_mod(): lp.testing.assert_equal(lp.mod([1, 2, 3, 4], [1, 2, 6, 4]), np.mod([1, 2, 3, 4], [1, 2, 6, 4]))
def test_modf(): lp.testing.assert_equal(lp.mod([1.1, 2.2, 3.3, 4.4], [1.1, 2.2, 6.6, 4.4]), np.mod([1.1, 2.2, 3.3, 4.4], [1.1, 2.2, 6.6, 4.4]))
def test_mod_array(): lp.testing.assert_equal(lp.mod([[1, 2, 3, 4], [1, 2, 6, 4]], [[2, 1, 7, 4], [1, 3, 6, 9]]), np.mod([[1, 2, 3, 4], [1, 2, 6, 4]], [[2, 1, 7, 4], [1, 3, 6, 9]]).tolist())
def test_cos(): lp.testing.assert_equal(lp.cos([1, 2, 3, 4]), np.cos([1, 2, 3, 4]))
def test_sqrt(): lp.testing.assert_equal(lp.sqrt([1, 2, 3, 4]), np.sqrt([1, 2, 3, 4]))
def test_arctan2(): lp.testing.assert_equal(lp.arctan2([1, 2, 3, 4], [1, 2, 6, 4]), np.arctan2([1, 2, 3, 4], [1, 2, 6, 4]))
def test_amax(): lp.testing.assert_equal(lp.amax([1, 2, 3, 4]), np.amax([1, 2, 3, 4]))
def test_amax_array(): lp.testing.assert_equal(lp.amax([(14, 29, 34), (41, 55, 46), (1, 38, 29), (5, 57, 52)]), np.amax([(14, 29, 34), (41, 55, 46), (1, 38, 29), (5, 57, 52)]))
def test_isin(): lp.testing.assert_equal(lp.isin([1, 2, 3, 4], [1, 2, 6, 4]), np.isin([1, 2, 3, 4], [1, 2, 6, 4]).tolist())
def test_ceil(): lp.testing.assert_equal(lp.ceil([1.67, 4.5, 7, 9, 12]), np.ceil([1.67, 4.5, 7, 9, 12]))
def test_reshape(): lp.testing.assert_equal(lp.reshape([1, 2, 3, 4, 5, 6, 7, 8], (2, 4)), np.reshape([1, 2, 3, 4, 5, 6, 7, 8], (2, 4)).tolist())
def test_reshape_flat(): lp.testing.assert_equal(lp.reshape([[1, 2, 3, 4], [5, 6, 7, 8]], -1), np.reshape([[1, 2, 3, 4], [5, 6, 7, 8]], -1))
def test_reshape_flat_array(): lp.testing.assert_equal(lp.reshape([[[1, 2, 3, 4], [5, 6, 7, 8]],[[1, 2, 3, 4], [5, 6, 7, 8]]], -1), np.reshape([[[1, 2, 3, 4], [5, 6, 7, 8]],[[1, 2, 3, 4], [5, 6, 7, 8]]], -1))
def test_assert_allclose(): lp.testing.assert_allclose([3,6,9], [3,6,9])
def test_assert_allclose_atol(): lp.testing.assert_allclose([3,5,9], [3,6,9], atol=1, rtol=1)
def test_count_nonzero1(): lp.testing.assert_equal(lp.count_nonzero([[3,6,9], [3,0,9]]), np.count_nonzero([[3,6,9], [3,0,9]]))
def test_count_nonzero3(): lp.testing.assert_equal(lp.count_nonzero([[3,5,9], [0,0,0]]), np.count_nonzero([[3,5,9], [0,0,0]]))
def test_allclose1(): lp.testing.assert_equal(lp.allclose([1e10,1e-7], [1.00001e10,1e-8]), np.allclose([1e10,1e-7], [1.00001e10,1e-8]))
def test_allclose2(): lp.testing.assert_equal(lp.allclose([1e10,1e-8], [1.00001e10,1e-9]), np.allclose([1e10,1e-8], [1.00001e10,1e-9]))
def test_cbrt(): lp.testing.assert_equal(lp.cbrt([1,8,27]), np.cbrt([1,8,27]))
def test_copy(): lp.testing.assert_equal(lp.copy([[1,2,3,4],[5,6,7,8]]), np.copy([[1,2,3,4],[5,6,7,8]]).tolist())
def test_copyto(): b=[[0,0,0,0],[0,0,0,0]]; c=[[1, 2, 3, 4],[2,2,2,2]]; a=np.array(c); np.copyto(a, c); lp.copyto(b, c); lp.testing.assert_equal(b, a.tolist())
def test_median(): lp.testing.assert_equal(lp.median([[10, 7, 4], [3, 2, 1]]), np.median([[10, 7, 4], [3, 2, 1]]))
def test_arange(): lp.testing.assert_equal(lp.arange(3, 7).tolist(), np.arange(3, 7).tolist())
def test_flip(): lp.testing.assert_equal(lp.flip([1,2,3,4,5,6]), np.flip([1,2,3,4,5,6]))
def test_split(): lp.testing.assert_equal(lp.split(lp.arange(6), 2), np.split(np.arange(6), 2))
def test_tile(): lp.testing.assert_equal(lp.tile([0,1,2,3,4,5], 2).tolist(), np.tile([0,1,2,3,4,5], 2).tolist())
def test_concatenate(): lp.testing.assert_equal(lp.concatenate(([1,2,3,4],[4,5,6])), np.concatenate(([1,2,3,4],[4,5,6])))
def test_where(): lp.testing.assert_equal(lp.where([True, True, True, True, True, False, False, False, False, False], lp.arange(10), [ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90]), np.where(np.arange(10) < 5, np.arange(10), 10*np.arange(10)))
def test_cumsum(): lp.testing.assert_equal(lp.cumsum([[1,2,3], [4,5,6]]), np.cumsum([[1,2,3], [4,5,6]]))
def test_matmul(): lp.testing.assert_equal(lp.matmul([[1, 0], [0, 1]], [[4, 1], [2, 2]]), np.matmul([[1, 0], [0, 1]], [[4, 1], [2, 2]]).ravel())
def test_broadcast_to(): lp.testing.assert_equal(lp.broadcast_to([1, 2, 3], (3, 3)), np.broadcast_to([1, 2, 3], (3, 3)).ravel())
def test_outer(): lp.testing.assert_equal(lp.outer([[1,2,3,4],[5,6,7,8]], [[9,10,11,12],[13,14,15,16]]), np.outer([[1,2,3,4],[5,6,7,8]], [[9,10,11,12],[13,14,15,16]]).tolist())
def test_eye(): lp.testing.assert_equal(lp.eye(4,4,k=-1), np.eye(4,4,k=-1).tolist())
def test_expand_dims(): lp.testing.assert_equal(lp.expand_dims([[1,2,3,4],[5,6,7,8]], 2), np.expand_dims([[1,2,3,4],[5,6,7,8]], 2).tolist())
def test_argmax(): lp.testing.assert_equal(lp.argmax([[1,2,3,4],[5,6,7,8]]), np.argmax([[1,2,3,4],[5,6,7,8]]).tolist())
def test_argmax_axis(): lp.testing.assert_equal(lp.argmax([[1,2,3,4],[5,6,7,8]], axis=1), np.argmax([[1,2,3,4],[5,6,7,8]], axis=1).tolist())
def test_show_randomusage(): np.random.seed(1337); lp.random.seed(1337); np.random.randn(2,4); lp.random.randn(2,4); lp.random.randn(2,4, dtype=lp.float32); lp.random.randn((2,4), dtype=lp.float32)
def test_empty():
  try: lp.testing.assert_equal(lp.empty(6), np.empty(6))
  except AssertionError as e: print("this is not the same since numpy uses random memory value and i force set to 0", str(e))
def test_empty_tuple():
  try: lp.testing.assert_equal(lp.empty((6,6,6)), np.empty((6,6,6)).ravel())
  except AssertionError as e: print("this is not the same since numpy uses random memory value and i force set to 0", str(e))
def test_assert_equal():
  lp.testing.assert_equal([3,6,9], [3,6,9])
  try: lp.testing.assert_equal([3,5,9], [3,6,9]) # Assert failure
  except AssertionError as e: print(str(e), ", but that is expected")
  lp.testing.assert_equal([[1, 2, 3, 4], [5, 6, 7, 8]], [[1, 2, 3, 4], [5, 6, 7, 8]])
  try: lp.testing.assert_equal([[1, 2, 3, 4], [6, 6, 7, 8]], [[1, 2, 3, 4], [5, 6, 7, 8]]) # Assert failure
  except AssertionError as e: print(str(e), ", but that is expected")
print("OK!")
