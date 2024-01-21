class array(object):
  def __init__(self, x=0): self.x = ndarray(x)
  def __getattr__(self, name): return getattr(self.x, name)
  def __repr__(self): return repr(self.x)
  def __str__(self): return str(self.x)
  def __getitem__(self, itm): return self.x.__getitem__(itm)
  def __sub__(self, x): return ndarray([i - x for i in self.x])
  def __truediv__(self, x): return [i / x for i in self.x]
  def __floordiv__(self, x): return [i // x for i in self.x]
  def __call__(self, x): return ndarray(x)
  def __len__(self, x): return len(self.x) if "Image" not in str(type(x)) else len(x.size)
  def astype(self, x): return [x(self.x[i]).value for i in range(len(self.x))]
  def tolist(self): return list(self.x)
  def numpy(self): return list(self.x)

class asarray(array): pass

class ndarray(array):
  def __init__(self, x=0):
    if len(x) == 1:
      if not isinstance(x, int) and not isinstance(x[0], list): self.x = [0 for _ in range(x[0])]
      else: self.x = [0 for a in range(len(x)) for b in range(len(x[a])) if a != b]
    else: self.x = [x[i] for i in range(len(x))]
  def __len__(self): return len(self.x)
  def __truediv__(self, x): return [i // x for i in self.x]
  #def __getitem__(self, itm): return ndarray(self.x.__getitem__(itm))
  def astype(self, x): return [0 for _ in range(self.x[0])] if len(self.x) == 1 else [x(self.x[i]).value for i in range(len(self.x))]
