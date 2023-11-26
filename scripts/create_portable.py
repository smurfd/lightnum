import re

def create_portable_getdata(fn):
  data = []
  with open(fn, "r") as f:
    for line in f:
      if not line.startswith("from lightnum"): # We dont need those imports since its all in one file
        fnd = re.search(r"from lightnum.[\Sa-z]*", line)
        if fnd: line = line.replace(line[fnd.span()[0]:fnd.span()[1]], "from lightnum")
        data.append(line)
  return data

def create_portable_writedata(fn, data):
  imp = []; dat = []
  for d in data:
    if d.startswith("import"): imp.append(d)
    else: dat.append(d)
  imp = list(set(imp)) # hack to get unique list
  for i, u in enumerate(imp): dat.insert(i, u)
  with open(fn, "w") as f:
    for d in dat: f.write(str(d))

if __name__ == '__main__':
  data = []
  data.extend(create_portable_getdata("lightnum/dtypes.py"))
  data.extend(create_portable_getdata("lightnum/helper.py"))
  data.extend(create_portable_getdata("lightnum/array.py"))
  data.extend(create_portable_getdata("lightnum/random.py"))
  data.extend(create_portable_getdata("lightnum/testing.py"))
  data.extend(create_portable_getdata("lightnum/lightnum.py"))

  create_portable_writedata("lightnum.py", data)
