from lightnum.helper import helper
import builtins

class testing:
  def assert_equal(x, y):
    if not builtins.any(isinstance(i, (tuple,list)) for i in [x,y]): assert(x == y)
    elif not builtins.all(helper.looper(helper, x, y=y, ret=[], ass=True)): raise AssertionError("Values are not equal")

  def assert_array_equal(x, y):
    if not builtins.all(helper.looper(helper, x, y=y, ret=[], ass=True)): raise AssertionError("Values in array are not equal")

  def assert_allclose(x, y, rtol=1e-07, atol=0):
    miss, r, rdiff, adiff = 0, [], [], []
    if builtins.all(isinstance(i, int) for i in [x,y]):
      if x != y:
        miss += 1
        if atol: adiff.append(abs(abs(x) - abs(y)))
        rdiff.append(abs(abs(x) - abs(y)) / abs(y))
    else:
      r = helper.looper(helper, x, y=y, ret=[], ass=True, asscls=True)
      for i in range(0, len(r), 3):
        if r[i] is False: miss += 1
      for i in range(1,len(r),3):
        if r[i] != 0 and r[i] > atol:
          if atol: adiff.append(r[i])
          else: adiff.append(0)
      for i in range(2, len(r), 3):
        if r[i] != 0 and r[i] > rtol: rdiff.append(r[i])
    if miss != 0 and adiff!=[] and rdiff!=[]:
      if not r: print(f"Mismatched elements: {miss} / {1} ({round(miss / (1) * 100, 1)}%)")
      else: print(f"Mismatched elements: {miss} / {len(r)//3} ({round(miss / (len(r)//3) * 100, 1)}%)")
      print("Max absolute difference:", round(max(adiff), 10))
      print("Max relative difference:", round(max(rdiff), 10))
      raise AssertionError(f"Not equal to tolerance rtol={rtol}, atol={atol}")
    return True
