import ctypes as ct
import math

# types
float16 = ct.c_float
float32 = ct.c_float
float64 = ct.c_double
int8 = ct.c_int8
int16 = ct.c_int16
int32 = ct.c_int32
int64 = ct.c_int64
uint8 = ct.c_uint8
uint16 = ct.c_uint16
uint32 = ct.c_uint32
uint64 = ct.c_uint64
bool_ = ct.c_bool

# constants
inf = math.inf
nan = math.nan

cts = [ct.c_float, ct.c_float, ct.c_double, ct.c_int8, ct.c_int16, ct.c_int32, ct.c_int64, ct.c_uint8, ct.c_uint16, ct.c_uint32, ct.c_uint64, ct.c_bool]

class ctstruct(ct.Structure):
  def __eq__(self, other): # ==
    for fld in self._fields_:
      if not isinstance(fld, tuple(cts)):
        if str(self._fields_[1]) != str(other._fields_[1]): return False
      else:
        if getattr(self, fld[0]) != getattr(other, fld[0]): return False
    return True

  def __ne__(self, other): # !=
    for fld in self._fields_:
      if not isinstance(fld, tuple(cts)):
        if str(self._fields_[1]) != str(other._fields_[1]): return True
      else:
        if getattr(self, fld[0]) != getattr(other, fld[0]): return True
    return False

  def __lt__(self, other): # <
    for fld in self._fields_:
      if not isinstance(fld, tuple(cts)):
        if str(self._fields_[1]) < str(other._fields_[1]): return True
      else:
        if getattr(self, fld[0]) < getattr(other, fld[0]): return True
    return False

  def __le__(self, other): # <=
    for fld in self._fields_:
      if not isinstance(fld, tuple(cts)):
        if str(self._fields_[1]) <= str(other._fields_[1]): return True
      else:
        if getattr(self, fld[0]) <= getattr(other, fld[0]): return True
    return False

  def __gt__(self, other): # >
    for fld in self._fields_:
      if not isinstance(fld, tuple(cts)):
        if str(self._fields_[1]) > str(other._fields_[1]): return True
      else:
        if getattr(self, fld[0]) > getattr(other, fld[0]): return True
    return False

  def __ge__(self, other): # >=
    for fld in self._fields_:
      if not isinstance(fld, tuple(cts)):
        if str(self._fields_[1]) >= str(other._fields_[1]): return True
      else:
        if getattr(self, fld[0]) >= getattr(other, fld[0]): return True
    return False

class dtype(ctstruct):
  _fields_ = []
  def __init__(self, x="None", dtype=ct.c_uint8):
    self._fields_ = [x, dtype]
  def __repr__(self): return repr(self._fields_[1])
  def value(self):
    if not isinstance(tuple(self._fields_), tuple(cts)):
      return self._fields_[0]
    return self._fields_[1]
