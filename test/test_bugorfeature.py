import time

def aaa(): pass
def bbb(a=0, b=0, c=0, d=0, e=0, f=0, g=0, h=0, i=0, j=0, k=0, l=0, m=0, n=0, o=0, p=0): pass

def test_aaa():
  start = time.time()
  for _ in range(100000): aaa()
  timer1 = (time.time() - start) * 1000
  print('Function with no parameters: {0:.4f}ms'.format(timer1))

def test_bbb():
  start = time.time()
  for _ in range(100000): bbb()
  timer1 = (time.time() - start) * 1000
  print('Function with many parameters: {0:.4f}ms'.format(timer1))

test_aaa()
test_bbb()
