import random as rnd
import os
import pandas as pd
import csv
import datetime as dt
import trimesh
import numpy as np
from matplotlib.scale import scale_factory
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from Objetos import Volcano

class Balistico:
    def __init__(self, volcano:Volcano) -> None:
        self.volcano = volcano
        self.trayectory =[]
        self.velocity

