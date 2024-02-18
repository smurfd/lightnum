#!/usr/bin/env python3
import lightnum.lightnum as lp
import numpy as np # only to compare

testing = lp.testing()
r = lp.random()
def test_zeros() -> None: testing.assert_equal(lp.zeros([4]), np.zeros([4]).tolist())
def test_zeros_big() -> None: testing.assert_equal(lp.zeros([3, 3, 3, 3]), np.zeros([3, 3, 3, 3]).tolist())
def test_ones() -> None: testing.assert_equal(lp.ones([4]), np.ones([4]).tolist())
def test_ones_big() -> None: testing.assert_equal(lp.ones([3, 3, 3, 3]), np.ones([3, 3, 3, 3]).tolist())
def test_zeros_like() -> None: testing.assert_equal(lp.zeros_like([3]), np.zeros_like([3]).tolist())
def test_zeros_like_big() -> None: testing.assert_equal(lp.zeros_like([3, 3, 3, 3]), np.zeros_like([3, 3, 3, 3]).tolist())
def test_zeros_like_array() -> None: testing.assert_equal(lp.zeros_like([[0, 1, 2],[3, 4, 5]]), np.zeros_like([[0, 1, 2],[3, 4, 5]]).ravel())
def test_ones_like() -> None: testing.assert_equal(lp.ones_like([3]), np.ones_like([3]).tolist())
def test_ones_like_big() -> None: testing.assert_equal(lp.ones_like([3, 3, 3, 3]), np.ones_like([3, 3, 3, 3]).tolist())
def test_ones_like_array() -> None:testing.assert_equal(lp.ones_like([[0, 1, 2],[3, 4, 5]]), np.ones_like([[0, 1, 2],[3, 4, 5]]).ravel())
def test_array() -> None: testing.assert_equal(lp.array([3,3,3]).tolist(), np.array([3,3,3]).tolist())
def test_exp_exp() -> None: testing.assert_equal(lp.exp([2]), np.exp([2]).tolist())
def test_exp_exp2() -> None: testing.assert_equal(lp.exp2([2]), np.exp2([2]).tolist())
def test_log() -> None: testing.assert_equal(lp.log([1, 2, 3, 4]), np.log([1, 2, 3, 4]).tolist())
def test_log_array() -> None: testing.assert_equal(lp.log([[1, 2, 3, 4], [1, 2, 3, 4]]), np.log([[1, 2, 3, 4], [1, 2, 3, 4]]).tolist())
def test_sum() -> None: testing.assert_equal(lp.sum((3,3)), np.sum((3,3)).tolist())
def test_sumf() -> None: testing.assert_equal(lp.sum((3.3,3.3), dtype=lp.float32), np.sum((3.3,3.3)).tolist())
def test_max() -> None: testing.assert_equal(lp.max([[[[2, 2, 3, 4], [1, 2, 3, 4]],[[1, 2, 3, 4], [1, 2, 9, 4]]],[[[1, -1, 3, 4], [1, 2, 3, 4]],[[1, 2, 31, 4], [1, 2, 3, 4]]]]), np.max([[[[2, 2, 3, 4], [1, 2, 3, 4]],[[1, 2, 3, 4], [1, 2, 9, 4]]],[[[1, -1, 3, 4], [1, 2, 3, 4]],[[1, 2, 31, 4], [1, 2, 3, 4]]]]).tolist())
def test_min() -> None: testing.assert_equal(lp.min([[[[2, 2, 3, 4], [1, 2, 3, 4]],[[1, 2, 3, 4], [1, 2, 9, 4]]],[[[1, -1, 3, 4], [1, 2, 3, 4]],[[1, 2, 31, 4], [1, 2, 3, 4]]]]), np.min([[[[2, 2, 3, 4], [1, 2, 3, 4]],[[1, 2, 3, 4], [1, 2, 9, 4]]],[[[1, -1, 3, 4], [1, 2, 3, 4]],[[1, 2, 31, 4], [1, 2, 3, 4]]]]).tolist())
def test_maximum() -> None: testing.assert_equal(lp.maximum([[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]]), np.maximum([[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]]).tolist())
def test_minimum() -> None: testing.assert_equal(lp.minimum([[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]]), np.minimum([[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]]).tolist())
def test_full() -> None: testing.assert_equal(lp.full(6, 3), np.full(6, 3).tolist())
def test_full_tuple() -> None: testing.assert_equal(lp.full((6,6,6), 4), np.full((6,6,6), 4).tolist())
def test_mod() -> None: testing.assert_equal(lp.mod([1, 2, 3, 4], [1, 2, 6, 4]), np.mod([1, 2, 3, 4], [1, 2, 6, 4]).tolist())
def test_modf() -> None: testing.assert_equal(lp.mod([1.1, 2.2, 3.3, 4.4], [1.1, 2.2, 6.6, 4.4]), np.mod([1.1, 2.2, 3.3, 4.4], [1.1, 2.2, 6.6, 4.4]).tolist())
def test_mod_array() -> None: testing.assert_equal(lp.mod([[1, 2, 3, 4], [1, 2, 6, 4]], [[2, 1, 7, 4], [1, 3, 6, 9]]), np.mod([[1, 2, 3, 4], [1, 2, 6, 4]], [[2, 1, 7, 4], [1, 3, 6, 9]]).tolist())
def test_cos() -> None: testing.assert_equal(lp.cos([1, 2, 3, 4]), np.cos([1, 2, 3, 4]).tolist())
def test_sqrt() -> None: testing.assert_equal(lp.sqrt([1, 2, 3, 4]), np.sqrt([1, 2, 3, 4]).tolist())
def test_arctan2() -> None: testing.assert_equal(lp.arctan2([1, 2, 3, 4], [1, 2, 6, 4]), np.arctan2([1, 2, 3, 4], [1, 2, 6, 4]).tolist())
def test_amax() -> None: testing.assert_equal(lp.amax([1, 2, 3, 4]), np.amax([1, 2, 3, 4]).tolist())
def test_amax_array() -> None: testing.assert_equal(lp.amax([(14, 29, 34), (41, 55, 46), (1, 38, 29), (5, 57, 52)]), np.amax([(14, 29, 34), (41, 55, 46), (1, 38, 29), (5, 57, 52)]).tolist())
def test_isin() -> None: testing.assert_equal(lp.isin([1, 2, 3, 4], [1, 2, 6, 4]), np.isin([1, 2, 3, 4], [1, 2, 6, 4]).tolist())
def test_ceil() -> None: testing.assert_equal(lp.ceil([1.67, 4.5, 7, 9, 12]), np.ceil([1.67, 4.5, 7, 9, 12]).tolist())
def test_reshape() -> None: testing.assert_equal(lp.reshape([1, 2, 3, 4, 5, 6, 7, 8], (2, 4)), np.reshape([1, 2, 3, 4, 5, 6, 7, 8], (2, 4)).tolist())
def test_reshape_flat() -> None: testing.assert_equal(lp.reshape([[1, 2, 3, 4], [5, 6, 7, 8]], -1), np.reshape([[1, 2, 3, 4], [5, 6, 7, 8]], -1).tolist())
def test_reshape_flat_array() -> None: testing.assert_equal(lp.reshape([[[1, 2, 3, 4], [5, 6, 7, 8]],[[1, 2, 3, 4], [5, 6, 7, 8]]], -1), np.reshape([[[1, 2, 3, 4], [5, 6, 7, 8]],[[1, 2, 3, 4], [5, 6, 7, 8]]], -1).tolist())
def test_assert_allclose() -> None: testing.assert_allclose([3,6,9], [3,6,9])
def test_assert_allclose_atol() -> None: testing.assert_allclose([3,5,9], [3,6,9], atol=1, rtol=1)
def test_count_nonzero1() -> None: testing.assert_equal(lp.count_nonzero([[3,6,9], [3,0,9]]), np.count_nonzero([[3,6,9], [3,0,9]]))
def test_count_nonzero3() -> None: testing.assert_equal(lp.count_nonzero([[3,5,9], [0,0,0]]), np.count_nonzero([[3,5,9], [0,0,0]]))
def test_allclose1() -> None: testing.assert_equal(lp.allclose([1e10,1e-7], [1.00001e10,1e-8]), np.allclose([1e10,1e-7], [1.00001e10,1e-8]))
def test_allclose2() -> None: testing.assert_equal(lp.allclose([1e10,1e-8], [1.00001e10,1e-9]), np.allclose([1e10,1e-8], [1.00001e10,1e-9]))
def test_cbrt() -> None: testing.assert_equal(lp.cbrt([1,8,27]), np.cbrt([1,8,27]).tolist())
def test_copy() -> None: testing.assert_equal(lp.copy([[1,2,3,4],[5,6,7,8]]), np.copy([[1,2,3,4],[5,6,7,8]]).tolist())
def test_copyto() -> None:
  b=[[0,0,0,0],[0,0,0,0]]
  c=[[1, 2, 3, 4],[2,2,2,2]]
  a=np.array(c)
  np.copyto(a, c)
  lp.copyto(b, c)
  testing.assert_equal(b, a.tolist())
def test_median() -> None: testing.assert_equal(lp.median([[10, 7, 4], [3, 2, 1]]), np.median([[10, 7, 4], [3, 2, 1]]).tolist())
def test_arange() -> None: testing.assert_equal(lp.arange(3, 7).tolist(), np.arange(3, 7).tolist())
def test_flip() -> None: testing.assert_equal(lp.flip([1,2,3,4,5,6]), np.flip([1,2,3,4,5,6]).tolist())
def test_split() -> None: testing.assert_equal(lp.split(lp.arange(6), 2), np.split(np.arange(6), 2))
def test_tile() -> None: testing.assert_equal(lp.tile([0,1,2,3,4,5], 2).tolist(), np.tile([0,1,2,3,4,5], 2).tolist())
def test_concatenate() -> None: testing.assert_equal(lp.concatenate(([1,2,3,4],[4,5,6])), np.concatenate(([1,2,3,4],[4,5,6])).tolist())
def test_where() -> None: testing.assert_equal(lp.where(np.arange(10) < 5, lp.arange(10).tolist(), [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]), np.where(np.arange(10) < 5, np.arange(10), 10*np.arange(10)).tolist()) #type: ignore
def test_cumsum() -> None: testing.assert_equal(lp.cumsum([[1,2,3], [4,5,6]]), np.cumsum([[1,2,3], [4,5,6]]).tolist())
def test_matmul() -> None: testing.assert_equal(lp.matmul([[1, 0], [0, 1]], [[4, 1], [2, 2]]), np.matmul([[1, 0], [0, 1]], [[4, 1], [2, 2]]).ravel())
def test_broadcast_to() -> None: testing.assert_equal(lp.broadcast_to([1, 2, 3], (3, 3)), np.broadcast_to([1, 2, 3], (3, 3)).tolist())
def test_outer() -> None: testing.assert_equal(lp.outer([[1,2,3,4],[5,6,7,8]], [[9,10,11,12],[13,14,15,16]]), np.outer([[1,2,3,4],[5,6,7,8]], [[9,10,11,12],[13,14,15,16]]).tolist())
def test_eye() -> None: testing.assert_equal(lp.eye(4,4,k=-1), np.eye(4,4,k=-1).tolist())
def test_expand_dims() -> None: testing.assert_equal(lp.expand_dims([[1,2,3,4],[5,6,7,8]], 2), np.expand_dims([[1,2,3,4],[5,6,7,8]], 2).tolist())
def test_argmax() -> None: testing.assert_equal(lp.argmax([[1,2,3,4],[5,6,7,8]]), np.argmax([[1,2,3,4],[5,6,7,8]]).tolist())
def test_argmax_axis() -> None: testing.assert_equal(lp.argmax([[1,2,3,4],[5,6,7,8]], axis=1), np.argmax([[1,2,3,4],[5,6,7,8]], axis=1).tolist())
def test_transpose() -> None: testing.assert_equal(lp.transpose([[1,2,3,4],[5,6,7,8]]), np.transpose([[1,2,3,4],[5,6,7,8]]).tolist())
def test_stack() -> None: testing.assert_equal(lp.stack([[1,2,3,4],[5,6,7,8]]), np.stack([[1,2,3,4],[5,6,7,8]]).tolist())
def test_vstack() -> None: testing.assert_equal(lp.vstack([[1,2,3,4],[5,6,7,8]]), np.vstack([[1,2,3,4],[5,6,7,8]]).tolist())
def test_nonzero() -> None: testing.assert_equal(lp.nonzero([[1,2,3,0],[5,6,7,8]]), np.nonzero([[1,2,3,0],[5,6,7,8]]))
def test_squeeze() -> None: testing.assert_equal(lp.squeeze([[[0, 1, 2], [3, 4, 5], [6, 7, 8]]]), np.squeeze([[[0, 1, 2], [3, 4, 5], [6, 7, 8]]]).tolist())
def test_clip() -> None: testing.assert_equal(lp.clip([[1, 2, 3, 4], [5, 6, 7, 8]], 2, 5), np.clip([[1, 2, 3, 4], [5, 6, 7, 8]], 2, 5).tolist())
def test_unique() -> None: testing.assert_equal(lp.unique([[[0, 1, 0, 2], [3, 4, 5, 2], [6, 7, 8, 4]]]), np.unique([[[0, 1, 0, 2], [3, 4, 5, 2], [6, 7, 8, 4]]]).tolist())
def test_triu() -> None: testing.assert_equal(lp.triu([[1,2,3],[1,2,3],[1,2,3]], 1), np.triu([[1,2,3],[1,2,3],[1,2,3]], 1).tolist())
def test_meshgrid() -> None:
  lpx,lpy=lp.meshgrid([1,2,3], [4,5,6])
  npx,npy=np.meshgrid([1,2,3], [4,5,6])
  testing.assert_equal(lpx, npx.tolist())
  testing.assert_equal(lpy, npy.tolist())
def test_newaxis() -> None:
  x = np.array([[1,2,3],[1,2,3]])
  testing.assert_equal(lp.newaxis([[1,2,3],[1,2,3]], 1), x[np.newaxis, :].tolist())
def test_frombuffer() -> None: testing.assert_equal(lp.frombuffer(b'smurfd', lp.uint8), np.frombuffer(b'smurfd', np.uint8))
def test_promote_types() -> None: assert(str(lp.promote_types(lp.uint8, lp.uint32)) == np.promote_types(np.uint8, np.uint32))
def test_delete() -> None: testing.assert_equal(lp.delete([[1,2,3],[4,5,6],[7, 8, 9]], 1), np.delete([[1,2,3],[4,5,6],[7, 8, 9]], 1))
def test_pad_constant() -> None: testing.assert_equal(lp.pad([1, 2, 3, 4, 5], 2, 'constant', constant_values=(4, 6)), np.pad([1, 2, 3, 4, 5], 2, 'constant', constant_values=(4, 6)).tolist())
def test_pad_edge() -> None: testing.assert_equal(lp.pad([1, 2, 3, 4, 5], 2, 'edge'), np.pad([1, 2, 3, 4, 5], 2, 'edge').tolist())
def test_pad_linear_ramp() -> None: testing.assert_equal(lp.pad([1, 2, 3, 4, 5], 2, 'linear_ramp', end_values=(5, -4)), np.pad([1, 2, 3, 4, 5], 2, 'linear_ramp', end_values=(5, -4)).tolist())
def test_pad_maximum() -> None: testing.assert_equal(lp.pad([1, 2, 3, 4, 5], 2, 'maximum'), np.pad([1, 2, 3, 4, 5], 2, 'maximum').tolist())
def test_pad_mean() -> None: testing.assert_equal(lp.pad([1, 2, 3, 4, 5], 2, 'mean'), np.pad([1, 2, 3, 4, 5], 2, 'mean').tolist())
def test_pad_median() -> None: testing.assert_equal(lp.pad([1, 2, 3, 4, 5], 2, 'median'), np.pad([1, 2, 3, 4, 5], 2, 'median').tolist())
def test_pad_minimum() -> None: testing.assert_equal(lp.pad([1, 2, 3, 4, 5], 2, 'minimum'), np.pad([1, 2, 3, 4, 5], 2, 'minimum').tolist())
def test_pad_reflect() -> None: testing.assert_equal(lp.pad([1, 2, 3, 4, 5], 2, 'reflect'), np.pad([1, 2, 3, 4, 5], 2, 'reflect').tolist())
def test_pad_symmetric() -> None: testing.assert_equal(lp.pad([1, 2, 3, 4, 5], 2, 'symmetric'), np.pad([1, 2, 3, 4, 5], 2, 'symmetric').tolist())
def test_pad_wrap() -> None: testing.assert_equal(lp.pad([1, 2, 3, 4, 5], 2, 'wrap'), np.pad([1, 2, 3, 4, 5], 2, 'wrap').tolist())
def test_require() -> None:
  x = lp.require([1,2,3,5], lp.int32, requirements=['C'])
  y = np.require([1,2,3,5], np.int32, requirements=['C'])
  for i in ['C_CONTIGUOUS', 'F_CONTIGUOUS', 'OWNDATA', 'ALIGNED', 'WRITEBACKIFCOPY']: assert(x.flags[i] == y.flags[i]) #type: ignore
def test_show_randomusage() -> None:
  np.random.seed(1337)
  r.seed(1337)
  np.random.randn(2,4)
  r.randn(2,4)
  r.randn(2,4, dtype=lp.int32)
  r.randn((2,4), dtype=lp.int32)
def test_empty() -> None:
  try: testing.assert_equal(lp.empty(6, dtype=lp.int32), np.empty(6, dtype=np.int32).tolist())
  except AssertionError as e: print("this is not the same since numpy uses random memory value and i force set to 0", str(e))
def test_empty_tuple() -> None:
  try: testing.assert_equal(lp.empty((6,6,6), dtype=lp.int32), np.empty((6,6,6), dtype=np.int32).ravel())
  except AssertionError as e: print("this is not the same since numpy uses random memory value and i force set to 0", str(e))
def test_assert_equal() -> None:
  testing.assert_equal([3,6,9], [3,6,9])
  try: testing.assert_equal([3,5,9], [3,6,9]) # Assert failure
  except AssertionError as e: print(str(e), ", but that is expected")
  testing.assert_equal([[1, 2, 3, 4], [5, 6, 7, 8]], [[1, 2, 3, 4], [5, 6, 7, 8]])
  try: testing.assert_equal([[1, 2, 3, 4], [6, 6, 7, 8]], [[1, 2, 3, 4], [5, 6, 7, 8]]) # Assert failure
  except AssertionError as e: print(str(e), ", but that is expected")
def test_save_load() -> None:
  with open('/tmp/testlp.npy', 'wb') as f:
    lp.save(f, [1,2,3,4,5,6,7,8,9])
    lp.save(f, [10, 11, 12, 13, 14, 15, 16, 17, 18, 191111111111111111])
  with open('/tmp/testlp.npy', 'rb') as f:
    lpx = lp.load(f)
    lpy = lp.load(f)
  with open('/tmp/testnp.npy', 'wb') as f:
    np.save(f, [1,2,3,4,5,6,7,8,9])
    np.save(f, [10, 11, 12, 13, 14, 15, 16, 17, 18, 191111111111111111])
  with open('/tmp/testnp.npy', 'rb') as f:
    npx = np.load(f)
    npy = np.load(f)
  testing.assert_equal(lpx, npx)
  testing.assert_equal(lpy, npy)
  testing.assert_equal(lpx, [1,2,3,4,5,6,7,8,9])
  testing.assert_equal(lpy, [10, 11, 12, 13, 14, 15, 16, 17, 18, 191111111111111111])
def test_load_z() -> None:
  np.savez('/tmp/testnp.npz', name1=np.arange(8).reshape(2, 4).tolist(), name2=np.arange(10).reshape(2, 5).tolist())
  lpz = lp.load('/tmp/testnp.npz') #type: ignore
  npz = np.load('/tmp/testnp.npz')
  testing.assert_equal(lpz['name1'], npz['name1'].tolist()) #type: ignore
  testing.assert_equal(lpz['name2'], npz['name2'].tolist()) #type: ignore

if __name__ == '__main__':
  print("OK!")
