import numpy as np


def load(fpath, Data):
  # Load data
  b = np.load(fpath + 'High' + Data + '.npz')
  print(b.files)

  HZ = b['Z']
  print("HZ.shape :", HZ.shape)

  Hnu_Z = b['nu_Z']
  print("Hnu_Z.shape:  ", Hnu_Z.shape)

  b = np.load(fpath + 'Low' + Data + '.npz')
  print(b.files)
  LZ = b['Z']
  print("LZ.shape :", LZ.shape)

  Lnu_Z = b['nu_Z']
  print("LZ.shape :", Lnu_Z.shape)

  return (HZ, Hnu_Z, LZ, Lnu_Z)
