from lightnum.helper import helper
import builtins

class testing:
  def assert_equal(x, y):
    if not builtins.any(isinstance(i, (tuple,list)) for i in [x,y]): assert(x == y)
    elif not builtins.all(helper.looper_assert(x, y)): raise AssertionError("Values are not equal")

  def assert_array_equal(x, y):
    if not builtins.all(helper.looper_assert(x, y)): raise AssertionError("Values in array are not equal")

  def assert_allclose(x, y, rtol=1e-07, atol=0):
    miss, r, rdiff, adiff = 0, [], [], []
    if builtins.all(isinstance(i, int) for i in [x,y]):
      if x != y:
        miss += 1
        if atol: adiff.append(abs(abs(x) - abs(y)))
        rdiff.append(abs(abs(x) - abs(y)) / abs(y))
    else:
      r = helper.looper_assert_close(x, y)
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
