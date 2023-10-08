import ctypes
import math

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
bool_ = ctypes.c_bool

# constants
inf = math.inf
nan = math.nan

class helper():
  def typ(x, dtype=int32): return dtype(x).value
  def sum(x, y): return x + y
  def mod(x, y): return x % y
  def exp2(x): return 2 ** x
  def cbrt(x): return round(x**(1 / 3.), 2)

  # helper function to return a row of a list
  def getrow(self, x, fill=0): return [self.looper(helper, x[-1], fill=fill, like=False, noloop=True) for _ in range(x[len(x) - 2])]

  # tuple (1,2,3,4,5)
  #                ^- # per row
  #              ^- # rows per box
  #            ^- # boxes
  # helper function to loop through multidimentional lists to check
  def looper(self, x, ret = 0, y = 0, max=False, min=False, maxi=False, mini=False, isin=False, ass=False, asscls=False, cls=False, count=False, countzero=False, add=False, fill=0, like=False, noloop=False, loop=False, call=None, dtype=int32):
    if loop:
      if y and not isinstance(x, list) and not isinstance(x, tuple): return call(x, y)
      if y: return [self.looper(self, i, call=call, y=j, loop=True) for i, j in zip(x, y)]
      if not isinstance(x, list): return call(x)
      return [self.looper(self, i, call=call, y=y, loop=True) for i in x]

    if noloop:
      if type(x) is int: return [fill] * x
      if type(x) is list:
        if like and type(x[0]) is not list: return [fill] * len(x)
        elif like: return [fill for a in range(len(x)) for b in range(len(x[a]))]
        elif len(x) == 1: return [fill] * x[0]
        else: return [fill] * len(x)
      if len(x) <= 2: return self.getrow(self, x, fill) # if onedimentional [2, 4]
      return [[self.getrow(self, x, fill) for i in range(x[l])] for l in range(len(x) - 2, -1, -1)].pop()

    if type(x) is bool: return x
    if (mini or maxi): tmp=[]; ret=[]
    for i in range(len(x)):
      if (maxi or mini) and type(x[i]) is list and type(y[i]) is list: tmp.append(self.looper(self, x[i], ret, y[i], maxi=maxi, mini=mini))
      elif type(y) is list and type(x[i]) is list and type(y[i]) is list: ret = self.looper(self, x[i], ret, y[i], max=max, min=min, isin=isin, ass=ass, asscls=asscls, count=count, countzero=countzero, add=add)
      elif type(x[i]) is list and type(y) is not list: ret = self.looper(self, x[i], ret, y, max=max, min=min, isin=isin, ass=ass, asscls=asscls, count=count, countzero=countzero, add=add)
      elif type(x[i]) is tuple: ret = self.looper(self, x[i], ret, y, max=max, min=min, isin=isin, ass=ass, asscls=asscls, count=count, countzero=countzero, add=add)
      else:
        if ass and round(x[i], 8) == round(y[i], 8): ret.append(True)
        elif ass: ret.append(False)
        if ass and asscls: ret.append(abs(abs(x[i]) - abs(y[i])))
        if ass and asscls and y[i]: ret.append(abs(abs(x[i]) - abs(y[i])) / abs(y[i]))
        if ass and asscls and cls: ret.append(abs(y[i]))
        if count and (x[i] != 0 and x[i] is not False): ret = ret + 1
        elif countzero and (x[i] == 0): ret = ret + 1
        elif add: ret = ret + x[i]
        elif (max and ret <= x[i]) or (min and ret >= x[i]): ret = x[i]
        elif (mini or maxi): tmp.append(y[i]);tmp.append(x[i]); ret.extend(tmp); tmp = []
        elif isin and x[i] == y[i]: ret.append(True)
        elif isin and x[i] != y[i]: ret.append(False)
    return ret

def reshape(l, shape):
    ncols, nrows, ret = 0, 0, []
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
