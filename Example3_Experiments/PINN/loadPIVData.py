import numpy as np
import scipy as sc
import matplotlib
from matplotlib import pyplot as plt
import os
from os.path import dirname
from pathlib import Path

import fdTools as fdt

class GridData:
    XX = None
    YY = None
    TT = None
    UU = None
    VV = None
    WZ = None
    FX = None
    FY = None
    Re = None
    

def loadTrainingData():
  
  D = sc.io.loadmat('../PIVData.mat')
  
  
  Data = GridData()
  Data.XX = D['XX']
  Data.YY = D['YY']
  Data.TT = D['TT']
  Data.UU = D['UU']
  Data.VV = D['VV']
  Data.WZ = D['WZ']
  Data.FX = D['FX']
  Data.FY = D['FY']
  Data.Re = D['Re']
  return Data





