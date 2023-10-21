# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 14:19:12 2023

@author: ZAID
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
x= np.arange(-10, 11)

#%%
plt.plot(x)
#%%
x2 = x**2
x3 = x ** 3
#%%
plt.plot(x)
plt.plot(x2)
plt.plot(x3)
plt.show()
#%%