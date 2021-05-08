import os
import errno
from sys import platform

def mkdir_if_missing(dir_path):
  try:
    os.makedirs(dir_path)
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise


def make_symlink_if_not_exists(real_path, link_path):
  '''
  param real_path: str the path linked
  param link_path: str the path with only the symbol
  '''
  print(link_path)
  real_path = os.path.normpath(real_path)
  link_path = os.path.normpath(link_path)

  if platform == 'win32':
    cmd = 'mklink /J "{1}" "{0}"'
  else:
    path_to_create = real_path
    cmd = 'ln -s {0} {1}'

  os.makedirs(real_path, exist_ok=True)
  os.system(cmd.format(real_path, link_path))
