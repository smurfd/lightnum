import builtins
import ctypes
import math
import random as rnd
import array as arr

# types
float16 = ctypes.c_float
float32 = ctypes.c_float
float64 = ctypes.c_double
int8 = ctypes.c_int8
int16 = ctypes.c_int16
int32 = ctypes.c_int32
int64 = ctypes.c_int64
uint8 = ctypes.c_uint8
uint16 = ctypes.c_uint16
uint32 = ctypes.c_uint32
uint64 = ctypes.c_uint64

# constants
inf = math.inf
nan = math.nan

# TODO0: refactor helper functions to better names and maby squash into fewer functions
# TODO1: make dtype=xxx do something
class array(object):
  def __init__(self, x=0): self.x = x
  def __getattr__(self, name): return getattr(self.x, name)
  def __repr__(self): return repr(self.x)
  def __str__(self): return str(self.x)
  def flatten(self, x): return reshape(x, -1)
  def astype(self, x): return [x(self.x[i]).value for i in range(len(self.x))]
  def tolist(self): return list(self.x)
  def numpy(self): return list(self.x)

class asarray(array): pass

class ndarray(array):
  def __init__(self, x=0):
    if len(x) == 1:
      if type(x) is not int and type(x[0]) is not list: self.x = [0 for _ in range(x[0])]
      else: self.x = [0 for a in range(len(x)) for b in range(len(x[a])) if a != b]
    else: self.x = [x[i] for i in range(len(x))]

  def astype(self, x):
    if len(self.x) == 1: return [0 for _ in range(self.x[0])]
    else: return [x(self.x[i]).value for i in range(len(self.x))]

class helper():
  def sum(x,y): return x + y
  def mod(x, y): return x % y
  def exp2(x): return 2 ** x
  def cbrt(x): return round(x**(1/3.),2)
  # helper function to loop through multidimentional lists
  def loop(self, x, call, y=0):
    ret = []
    for i in range(len(x)):
      if type(y) is list and type(x[i]) is list and type(y[i]) is list: ret.append(self.loop(self, x[i], call, y[i]))
      elif type(x[i]) is list: ret.append(self.loop(self, x[i], call, y=y))
      else:
        if type(x) is tuple: return call(x)
        if type(y) is tuple or type(y) is list: ret.append(call(x[i], y[i]))
        else: ret.append(call(x[i]))
    return ret

  # helper function to loop through multidimentional lists to make check
  def loopcheck(self, x, ret = 0, y = 0, max=False, min=False, isin=False, ass=False, asscls=False, cls=False, count=False):
    if type(x) is bool: return x
    for i in range(len(x)):
      if type(y) is list and type(x[i]) is list and type(y[i]) is list: ret = self.loopcheck(self, x[i], ret, y[i], max=max, min=min, isin=isin, ass=ass, asscls=asscls, count=count)
      elif type(x[i]) is list and type(y) is not list: ret = self.loopcheck(self, x[i], ret, y, max=max, min=min, isin=isin, ass=ass, asscls=asscls, count=count)
      elif type(x[i]) is tuple: ret = self.loopcheck(self, x[i], ret, y, max=max, min=min, isin=isin, ass=ass, asscls=asscls, count=count)
      else:
        if ass:
          if round(x[i], 8) == round(y[i], 8): ret.append(True)
          else: ret.append(False)
          if asscls:
            ret.append(abs(abs(x[i]) - abs(y[i])))
            if y[i]: ret.append(abs(abs(x[i]) - abs(y[i])) / abs(y[i]))
            if cls: ret.append(abs(y[i]))
        if count and (x[i] != 0 and x[i] != False): ret = ret + 1
        if max and ret <= x[i]: ret = x[i]
        if min and ret >= x[i]: ret = x[i]
        if isin and x[i] == y[i]: ret.append(True)
        elif isin and x[i] != y[i]: ret.append(False)
    return ret

  def box2x(self, x, fill=0): return [empty(x[-1], fill) for _ in range(x[len(x) - 2])]

  # tuple (1,2,3,4,5)
  #                ^- # per row
  #              ^- # rows per box
  #            ^- # boxes
  def boxloop(self, x, fill=0, like=False):
    ret = []
    tmp = []
    if type(x) is int: return [fill] * x
    if type(x) is list:
      if like and type(x[0]) is not list: return [fill] * len(x)
      elif like:
        for a in range(len(x)):
          for b in range(len(x[a])): tmp.append(fill)
          ret.append(tmp)
          tmp=[]
        return ret
      if len(x) == 1: return [fill] * x[0]
      else: ret = [fill] * len(x)
    if len(x) <= 2: return self.box2x(self, x, fill) # if onedimentional [2, 4]
    for l in range(len(x)-2,-1,-1): # len - 2 because we use the box2x for each row
      for i in range(x[l]): tmp.append(self.box2x(self, x, fill))
      ret.append(tmp)
    return ret.pop()

  def minmax(self, x, y, max=False, min=False):
    ret = []
    tmp = []
    for i in range(len(x)):
      if type(x[i]) is list: tmp.append(self.minmax(self, x[i], y[i], max, min))
      else:
        if max and y[i] >= x[i]: tmp.append(y[i])
        elif min and y[i] <= x[i]: tmp.append(y[i])
        else: tmp.append(x[i])
      ret.extend(tmp)
      tmp=[]
    return ret

class random():
  def seed(x, dtype=int32): return rnd.seed(x)
  def randint(x, dtype=int32): return rnd.randint(x)
  def randn(*args, dtype=int32):
    ret=[]
    if type(args) is tuple and len(args) == 1:
      if len(args[0]) != 1:
        for j in range(len(args[0])):
          for i in range(args[0][j]): ret.append(i)
      else:
        for i in range(len(args[0])): ret.append(i)
    else:
      for i in args: ret.append(i)
    return ndarray(helper.boxloop(helper, ret, fill=rnd.random()))
  def random(size, dtype=float32): return random.randn(size)
  def rand(*args, dtype=int32): return random.randn(args, dtype=dtype)
  def choise(x, dtype=int32): return rnd.choise(x)
  def RandomState(x, dtype=int32): return rnd.RandomState(x)
  def default_rng(x, dtype=int32): return rnd.default_rng(x)

# math
def log(x, dtype=int32): return helper.loop(helper, x, math.log) # Seems to work
def exp(x, dtype=int32): return helper.loop(helper, x, math.exp) # Seems to work
def exp2(x, dtype=int32): return helper.loop(helper, x, helper.exp2) # Seems to work
def cbrt(x, dtype=int32): return helper.loop(helper, x, helper.cbrt) # Seems to work
def sum(x, y=0, dtype=int32): # Seems to work
  if dtype == int32: return helper.loop(helper, x, builtins.sum, y=y)
  else: return helper.loop(helper, x, math.fsum, y=y)
def mod(x, y, dtype=int32): # Seems to work
  if dtype == int32: return helper.loop(helper, x, helper.mod, y)
  else: helper.loop(helper, x, math.fmod, y)
def prod(x, y=0, dtype=int32): return helper.loop(helper, x, math.prod)
def zeros(s, d=0, dtype=float32): return helper.boxloop(helper, s, fill=d)
def zeros_like(s, d=0, dtype=int32): return empty(s, fill=0, like=True) # Seems to work
def ones(s, d=1, dtype=int32): return zeros(s, d) # Seems to work # call the same function as zeros but set it to 1 instead
def ones_like(s, d=1, dtype=int32): return empty(s, fill=1, like=True) # Seems to work
def max(x): return helper.loopcheck(helper, x, max=True) # Seems to work
def min(x): return helper.loopcheck(helper, x, min=True) # Seems to work
def maximum(x, y): return helper.minmax(helper, x, y, max=True) # Seems to work
def minimum(x, y): return helper.minmax(helper, x, y, min=True) # Seems to work
def empty(x, fill=0, like=False): return helper.boxloop(helper, x, fill=fill, like=like) # Seems to work
def full(x, fill): return helper.boxloop(helper, x, fill=fill) # Seems to work
def cos(x, dtype=int32): return helper.loop(helper, x, math.cos) # Seems to work
def sqrt(x, dtype=int32): return helper.loop(helper, x, math.sqrt) # Seems to work
def arctan2(x, y, dtype=int32): return helper.loop(helper, x, math.atan2, y) # Seems to work
def amax(x, dtype=int32): return helper.loopcheck(helper, x, max=True) # Kinda works
def isin(x, y, dtype=int32): return helper.loopcheck(helper, x, y=y, ret=[], isin=True) # Seems to work
def ceil(x, dtype=int32): return helper.loop(helper, x, math.ceil) # Seems to work
def count_nonzero(x): return helper.loopcheck(helper, x, ret=0, count=True) # Seems to work
def not_equal(x, y): return not testing.assert_equal(x, y) # Seems to work
def array_equal(x, y): return testing.assert_equal(x, y) # Seems to work
def arange(x,y,z, dtype=int32): return ndarray(helper.boxloop(helper, ret, fill=rnd.random()))
def any(x): return builtins.any(helper.loopcheck(helper, x, ret=0, count=True)) # Seems to work
def all(x): return builtins.all(helper.loopcheck(helper, x, ret=0, count=True)) # Seems to work

def allclose(x, y, rtol=1e-05, atol=1e-08): # Seems to work
  r = []
  diff = []
  r = helper.loopcheck(helper, x, y=y, ret=[], ass=True, asscls=True, cls=True)
  for i in range(1,len(r),4):
    if (r[i] <= atol + rtol * r[i + 2]) == False: return False
  return True

def reshape(l, shape): # Seems to work
  ncols, nrows = 0, 0
  ret = []
  if shape == -1:
    if type(l) is not list and type(l) is not tuple: ncols, nrows = l, 1
    else: ncols, nrows = len(l), 1
  elif type(shape) is tuple: nrows, ncols = shape
  else: ncols, nrows = len(l), 1
  for r in range(nrows):
    row = []
    for c in range(ncols):
      if shape == -1 and type(l) is list and type(l[c]) is not float and type(l[c]) is not int: row.extend(reshape(l[c], -1))
      elif shape == -1: row.extend(l); break
      else: row.append(l[ncols * r + c])
    if shape == -1: ret.extend(row)
    else: ret.append(row)
  return ret

class ctypeslib: # kindof
  def as_array(x, shape): return arr.array('i', x)

class testing:
  def assert_equal(x, y):
    if type(x) is not tuple and type(y) is not tuple and type(x) is not list and type(y) is not list: assert(x == y) # meaning either int or float
    elif not builtins.all(helper.loopcheck(helper, x, y=y, ret=[], ass=True)): raise AssertionError("Values are not equal")

  def assert_array_equal(x, y):
    if not builtins.all(helper.loopcheck(helper, x, y=y, ret=[], ass=True)): raise AssertionError("Values in array are not equal")

  def assert_allclose(x, y, rtol=1e-07, atol=0):
    miss = 0
    r = []
    rdiff = []
    adiff = []
    if type(x) is int and type(y) is int:
      if x != y:
        miss = miss + 1
        if atol: adiff.append(abs(abs(x) - abs(y)))
        rdiff.append(abs(abs(x) - abs(y)) / abs(y))
    else:
      r = helper.loopcheck(helper, x, y=y, ret=[], ass=True, asscls=True)
      for i in range(0,len(r),3):
        if r[i] == False: miss = miss + 1
      for i in range(1,len(r),3):
        if r[i] != 0 and r[i] > atol:
          if atol: adiff.append(r[i])
          else: adiff.append(0)
      for i in range(2,len(r),3):
        if r[i] != 0 and r[i] > rtol: rdiff.append(r[i])
    if miss != 0 and adiff!=[] and rdiff!=[]:
      if not r: print(f"Mismatched elements: {miss} / {1} ({round(miss / (1) * 100, 1)}%)")
      else: print(f"Mismatched elements: {miss} / {len(r)//3} ({round(miss / (len(r)//3) * 100, 1)}%)")
      print("Max absolute difference:", round(max(adiff), 10))
      print("Max relative difference:", round(max(rdiff), 10))
      raise AssertionError(f"Not equal to tolerance rtol={rtol}, atol={atol}")
    return True

# TODO2
class lib:
  class stride_tricks:
    def as_strided(): pass

def concatenate(): pass
def expand_dims(): pass
def eye(): pass
def frombuffer(): pass
def arange(): pass
def stack(): pass
def tile(): pass
def argmax(): pass
def outer(): pass
def clip(): pass
def pad(): pass
def squeeze(): pass
def broadcast_to(): pass
def nonzero(): pass
def unique(): pass
def promote_types(): pass
def multiply(): pass
def set_printoptions(): pass
def triu(): pass
def where(): pass
def dtype(): pass
def memmap(): pass
def require(): pass
def set_printoptions(): pass
def split(): pass
def moveaxis(): pass
def transpose(): pass
def rollaxis(): pass
def matmul(): pass
def argsort(): pass
def newaxis(): pass
def meshgrid(): pass
def delete(): pass
def vstack(): pass
def copy(): pass
def save(): pass
def load(): pass
def vstack(): pass
def median(): pass
def load(): pass
def cumsum(): pass
def flip(): pass
def copyto(): pass
