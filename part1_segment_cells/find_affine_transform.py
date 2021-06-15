# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("after_rotate_and_rescale.csv")
fitter = LinearRegression()

np_fit = df[['old_row','old_col','new_row','new_col']].to_numpy()
np_fit_x = np_fit[:,:2]
np_fit_y = np_fit[:,2:]

fitter.fit(np_fit_x,np_fit_y)
# print(fitter.coef_)
# print(fitter.intercept_)
# print(np.linalg.det(fitter.coef_))

matrix_affine = np.zeros((3,3))
matrix_affine[:2,:2] = fitter.coef_
matrix_affine[:2,2]  = fitter.intercept_
matrix_affine[2,2]   = 1
matrix_inv = np.linalg.inv(matrix_affine)
matrix_inv.tofile("transform_inverse.dat")

# %% 
np_fit1 = df[df['spectral_img']=='glu-0_field-1'][['old_row','old_col','new_row','new_col']].to_numpy()
np_fit1_x = np_fit1[:,:2]
np_fit1_y = np_fit1[:,2:]

fitter.fit(np_fit1_x,np_fit1_y)
print(fitter.coef_)
print(fitter.intercept_)
print(np.linalg.det(fitter.coef_))

# %%
np_fit2 = df[df['spectral_img']=='glu-100_field-5'][['old_row','old_col','new_row','new_col']].to_numpy()
np_fit2_x = np_fit2[:,:2]
np_fit2_y = np_fit2[:,2:]

fitter.fit(np_fit2_x,np_fit2_y)
print(fitter.coef_)
print(fitter.intercept_)
print(np.linalg.det(fitter.coef_))