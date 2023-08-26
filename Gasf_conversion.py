import pandas as pd
import numpy as np
import os
from pyts.image import GramianAngularField


data_dir="Youtube/vid" # directory where 1D traces are stored (Format : Youtube/vid*)
save_dir="Youtube_csv/vid" # directorty store GASF converted data

for i in range(20):
  vid=data_dir+str(i+1)
  path_to_save = os.path.abspath(save_dir+str(i+1))
  if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)
  k=1
  for file in os.scandir(vid): # load data and convert to GASF
    df = pd.read_csv(file,usecols=[' addr2_bytes']) # the column to read the data
    df = df.replace(np.nan, 0)
    data = df.to_numpy(dtype='float')
    pts=[]
    # GASF formation
    for l in data:
      for j in l:
        pts.append(j)
    num_of_samples_per_bin=4 #to bin the video data
    slice_index=0
    points=[]
    for j in range(int(len(pts)/num_of_samples_per_bin)):
      points.append(np.sum(pts[j*num_of_samples_per_bin:(j+1)*num_of_samples_per_bin]))
    X = np.array([points])
    # Compute Gramian angular fields
    gasf = GramianAngularField(sample_range=(0,1),method='summation')
    X_gasf = gasf.fit_transform(X)
    # Gamma corretion
    gasf_csv2=X_gasf[0]*0.5 + 0.5
    gamma=0.25
    gasf_csv2=np.power(gasf_csv2,gamma)
    gasf_csv2 = pd.DataFrame(data=gasf_csv2)
    gasf_csv2.to_csv(path_to_save + "/vid"+str(i+1)+"_" + str(k) + '.csv', header=False, index=False)
    k=k+1
