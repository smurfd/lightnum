import os

# Run this in your source folder to replace "import numpy as np" with "import lightnum.lightnum as np"
def scantree(path):
  for entry in os.scandir(path):
    if entry.is_dir(follow_symlinks=False): yield from scantree(entry.path)
    else: yield entry

def replace(rwith, rfind):
  for entry in scantree(os.path.dirname(os.path.abspath(__file__))):
    if entry.is_file() and os.fsdecode(entry).endswith('.py'):
      found = False
      with open(entry) as f:
        content = f.readlines()
        for i, line in enumerate(content):
          if line.lstrip().startswith(rfind):
            found = True
            content[i] = rwith + '\n'

      if found:
        with open(entry, 'w') as f:
          f.write(''.join(content))

if __name__ == '__main__':
  replace('import lightnum.lightnum as np', 'import numpy as np')
