from lightnum.helper import helper
from lightnum.array import ndarray
import builtins

class testing:
  def looper_assert(x, y, close=False, cls=False):
    ret=[]
    if len(x) != len(y): return [False] # this fixes assert, got some errors need to fix
    for i in range(len(x)):
      if isinstance(y, list) and builtins.all(isinstance(j, list) for j in [x[i],y[i]]): ret = testing.looper_assert(x[i], y[i])
      elif isinstance(x[i], list) and not isinstance(y, list): ret = testing.looper_assert(x[i], y)
      elif isinstance(x[i], tuple): ret = testing.looper_assert(x[i], y)
      elif isinstance(x[i], ndarray): ret = testing.looper_assert(x[i].tolist(), y[i].tolist())
      elif close and cls: ret = helper.if_round_abs(x[i], y[i], cls=True)
      elif close: ret = helper.if_round_abs(x[i], y[i])
      else: ret.append(True) if round(x[i], 8) == round(y[i], 8) else ret.append(False)
    return ret
  def looper_assert_close(x, y): return testing.looper_assert(x, y, close=True)
  def looper_assert_close_cls(x, y): return testing.looper_assert(x, y, close=True, cls=True)

  def assert_equal(x, y):
    if not builtins.any(isinstance(i, (tuple,list)) for i in [x,y]): assert(x == y)
    elif not builtins.all(testing.looper_assert(x, y)): raise AssertionError("Values are not equal")

  def assert_array_equal(x, y):
    if not builtins.all(testing.looper_assert(x, y)): raise AssertionError("Values in array are not equal")

  def assert_allclose(x, y, rtol=1e-07, atol=0):
    miss, r, rdiff, adiff = 0, [], [], []
    if builtins.all(isinstance(i, int) for i in [x,y]):
      if x != y:
        miss += 1
        if atol: adiff.append(abs(abs(x) - abs(y)))
        rdiff.append(abs(abs(x) - abs(y)) / abs(y))
    else:
      r = testing.looper_assert_close(x, y)
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
