#!/usr/bin/env python3
from __future__ import annotations
from lightnum.helper import helper
from lightnum.array import ndarray
import builtins, time
from typing import List, Callable, Any, Optional, SupportsAbs, SupportsRound, overload

class testing:
  def looper_assert(self, x: List[Any[int, float]] | SupportsRound[Any], y: List[Optional[Any]] | SupportsRound[Any], close: bool=False, cls: bool=False) -> List[Any[int, float]]:
    ret=[]
    if len(x) != len(y): return [False] # this fixes assert, got some errors need to fix
    for i in range(len(x)):
      if isinstance(y, list) and builtins.all(isinstance(j, list) for j in [x[i],y[i]]): ret = testing.looper_assert(self, x[i], y[i])
      elif isinstance(x[i], list) and not isinstance(y, list): ret = testing.looper_assert(self, x[i], y)
      elif isinstance(x[i], tuple): ret = testing.looper_assert(self, x[i], y)
      elif isinstance(x[i], ndarray): ret = testing.looper_assert(self, x[i].tolist(), y[i].tolist())
      elif close and cls: ret = helper.if_round_abs(helper, x[i], y[i], cls=True)
      elif close: ret = helper.if_round_abs(helper, x[i], y[i])
      else: ret.append(True) if round(x[i], 8) == round(y[i], 8) else ret.append(False)
    return ret
  def looper_assert_close(self, x: List[Any[int, float]] | complex | SupportsRound[float], y: List[Any[int, float]] | complex | SupportsRound[float]) -> List[Any[int, float]]: return testing.looper_assert(self, x, y, close=True)
  def looper_assert_close_cls(self, x: List[Any[int, float]], y: List[Any[int, float]]) -> List[Any[int, float]]: return testing.looper_assert(self, x, y, close=True, cls=True)

  def assert_equal(self, x: List[Any[int, float]], y: List[Any[int, float]]) -> None:
    if not builtins.any(isinstance(i, (tuple,list)) for i in [x,y]): assert(x == y)
    elif not builtins.all(testing.looper_assert(self, x, y)): raise AssertionError("Values are not equal")

  def assert_array_equal(self, x: List[Any[int, float]], y: List[Any[int, float]]) -> None:
    if not builtins.all(testing.looper_assert(self, x, y)): raise AssertionError("Values in array are not equal")

  def assert_allclose(self, x: List[Any[int, float]] | complex | SupportsAbs[float], y: List[Any[int, float]] | complex | SupportsAbs[float], rtol: float=1e-07, atol: int=0) -> bool:
    miss, r, rdiff, adiff = 0, [], [], []
    if builtins.all(isinstance(i, int) for i in [x,y]):
      if x != y:
        miss += 1
        if atol: adiff.append(abs(abs(x) - abs(y)))
        rdiff.append(abs(abs(x) - abs(y)) / abs(y))
    else:
      r = testing.looper_assert_close(self, x, y)
      for i in range(0, len(r), 3):
        if r[i] is False: miss += 1
        if r[(i:=i+1)] != 0 and r[i] > atol: adiff.append(r[i]) if atol else adiff.append(0)
        if r[(i:=i+1)] != 0 and r[i] > rtol: rdiff.append(r[i])
    if miss != 0 and adiff != [] and rdiff != []:
      if not r: print("Mismatched elements: {} / {} ({}%)".format(miss, 1, round(miss / (1) * 100, 1)))
      else: print("Mismatched elements: {} / {} ({}%)".format(miss, len(r) // 3, round(miss / (len(r)//3) * 100, 1)))
      print("Max absolute difference: {}\nMax relative difference: {}".format(round(max(adiff), 10), round(max(rdiff), 10)))
      raise AssertionError("Not equal to tolerance rtol={}, atol={}".format(rtol, atol))
    return True

  def timing_test_looper(self, fn1: Callable[..., List[Any]], fn2: Callable[..., List[Any]], s: str, *args: Any, **kwargs: Any) -> None:
    t1 = time.perf_counter()
    for _ in range(100000): fn1(*args, **kwargs)
    t2 = time.perf_counter()
    t3 = time.perf_counter()
    for _ in range(100000): fn2(*args, **kwargs)
    t4 = time.perf_counter()
    print('[{}] Numpy {}: {:.4f}ms Lightnum {}: {:.4f}ms = {}%'.format((t2 - t1) * 1000 > (t4 - t3) * 1000, s, (t2 - t1) * 1000, s, (t4 - t3) * 1000, int((t2 - t1) / (t4 - t3) * 100)))
