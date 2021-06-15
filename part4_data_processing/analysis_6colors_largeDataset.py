# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go


# %% 
# for one single cell file and one single organelle file
# used for EYrainbow Leucine 1stTry 
organelles = ["pex3","vph1","sec61","sec7","tom70","erg6"]

file_cell = r"D:\Documents\FILES\lab_Images\OrganelleScience\EY279xSixColor_Leucine\EY279xSixColor_Leucine_csv\cell.csv"
file_orga = r"D:\Documents\FILES\lab_Images\OrganelleScience\EY279xSixColor_Leucine\EY279xSixColor_Leucine_csv\organelle_rename.csv"

df_cell = pd.read_csv(str(Path(file_cell)))
df_cell.rename(columns={'leucine':'condition','cell_idx':"label-cell",'cell_area':'area'},inplace=True)
df_cell["hour"] = 3

df_orga = pd.read_csv(str(Path(file_orga)))
df_orga.rename(columns={'leucine':"condition",'orga':'organelle','cell_idx':"label-cell",'orga_size':'volume-organelle'},inplace=True)
df_orga["hour"] = 3
df_orga[df_orga["organelle"]=="vph1"].loc[:,"volume-organelle"] = np.pi*np.sqrt(df_orga[df_orga["organelle"]=="vph1"].loc[:,"volume-organelle"]**3)/6.
# %% [markdown]
# ## Read the files of cells and organelles.
# 
# During the last section the column name of cell labels were named 'label', which may be confused with organelle labels, so there is a rename command which can be commented out.
# 
# ### Variables In:
# - `folder_cell`: str of the name of the folder with cell csvs.
# - `folder_orga`: str of the name of the folder with organelle csvs.
# 
# ### Variables Local:
# - `files_cell`: generator of paths to all the cell csv files.
# - `files_orga`: generator of paths to all the organelle csv files.
# 
# ### Variables Out:
# - `df_cell`: properties of cells with meaningless indices
# - `df_orga`: properties of organelles with meaningless indices
# %%
# for folder where one file for each image.
organelles = ["pex3","vph1","sec61","sec7","tom70","erg6"]

folder_cell = r"D:\Documents\FILES\lab_Images\OrganelleScience\EY289xSixColor_Glucose_LargerData\EY289xSixColor_Glucose_LargerData_csv\cells"
folder_orga = r"D:\Documents\FILES\lab_Images\OrganelleScience\EY289xSixColor_Glucose_LargerData\EY289xSixColor_Glucose_LargerData_csv\organelles"

files_cell = Path(folder_cell).glob("*.csv")
df_cell = pd.concat((pd.read_csv(file) for file in files_cell))
files_orga = Path(folder_orga).glob("*.csv")
df_orga = pd.concat((pd.read_csv(file) for file in files_orga))

# df_cell = df_cell[df_cell["hour"]==3] # added afterwards
# df_orga = df_orga[df_orga["hour"]==3]

# %% [markdown]
# ## Get the statistics of organelles in each cell by `.groupby()`
# 
# ### Variables In:
# - `df_orga`: see last section
# 
# ### Variables Local:
# - `df_orga_meanbycell`: pd.Series of the mean of organelles in each cell
# - `df_orga_numbycell`:  pd.Series of the count of organelles in each cell
# 
# ### Variables Out:
# - `df_orga_percell`: pd.DataFrame indexed by ["condition","hour","field","organelle","label-cell",'mean','count','total']

# %%
df_orga = df_orga[df_orga["volume-pixel"]>10]
df_orga['depth'] = df_orga['bbox-3'] - df_orga['bbox-0']
df_orga = df_orga[df_orga["depth"]>0]

df_orga["volume-organelle"] = df_orga["volume-pixel"]
df_orga.loc[df_orga["organelle"]=="vph1","volume-organelle"] = df_orga.loc[df_orga["organelle"]=="vph1","volume-bbox"]

# %%
df_orga_meanbycell = df_orga[["condition","field","organelle","label-cell","volume-organelle"]].groupby(["condition","field","label-cell","organelle"])["volume-organelle"].apply(np.average)
df_orga_numbycell = df_orga[["condition","field","organelle","label-cell","volume-organelle"]].groupby(["condition","field","label-cell","organelle"])["volume-organelle"].count()

df_orga_percell = pd.DataFrame({"mean":df_orga_meanbycell,"count":df_orga_numbycell})
df_orga_percell["total"] = df_orga_percell["mean"] * df_orga_percell["count"]
df_orga_percell.reset_index(inplace=True)

# %% [markdown]
# ## Add cell properties into grouped organelle data
# 
# This is do-able iff two DataFrames have the same index. So `.set_index(inplace=True)` and `.reset_index(inplace=True)` are important.
#
# ### Variables In:
# - `df_cell`:         same as before
# - `df_orga_percell`
# 
# ### Variables Out:
# - `df_orga_percell`: columns added, now ["condition","hour","field","label-cell","organelle","mean","count","total",'volume-cell','fraction-total']  
# %%
df_orga_percell.set_index(["condition","field","label-cell"],inplace=True)
df_cell.set_index(["condition","field","label-cell"],inplace=True)

df_orga_percell["volume-cell"] = np.pi*np.sqrt(df_cell["area"]**3)/6.
df_orga_percell["fraction-total"] = df_orga_percell["total"]/df_orga_percell["volume-cell"]

df_cell.reset_index(inplace=True)
df_orga_percell.reset_index(inplace=True)

# %% [markdown] 
# ## Wash NaN before PCA
# 
# The dataset after excluding all nan is small, so we need a better way including data points where some components are nan, to enlarge the dataset
# 
# ### Variables In:
# - `df_orga_percell`
# 
# ### Variables Local:
# - `idx`
# - `orga`
# - `df_pca`: DataFrame indexed by ["condition","hour","field","label-cell",*organelles,'na_count']
# 
# ### Variables Out:
# - `df_pca_washed`: df_pca that exclude cells with too many nan and the  filled with zeros.
# %%
idx = df_orga_percell.groupby(["condition","field","label-cell"]).count().index
df_orga_percell.set_index(["condition","field","label-cell"],inplace=True)
df_pca = pd.DataFrame(index=idx,columns=organelles)
for orga in organelles:
    df_pca[orga] = df_orga_percell.loc[df_orga_percell["organelle"]==orga,"fraction-total"]
df_pca.reset_index(inplace=True)
df_pca["condition"] = df_pca["condition"]/100.
df_orga_percell.reset_index(inplace=True)
df_pca["na_count"] = df_pca.isna().sum(axis=1)

# df_pca_washed = df_pca[df_pca["na_count"]<3] 
df_pca_washed = df_pca
df_pca_washed = df_pca_washed.fillna(0.)
# %%
np.bincount(df_pca["na_count"])
# array([2777, 1327,  699,  385,  200,  123], dtype=int64)
# array([405, 387, 190,  64,  55,  31,  24], dtype=int64)
# array([1767, 2096, 1705, 1242,  629,  357], dtype=int64) glucose-large
# %% [markdown]
# ## Principal Component Analysis
# ### Variables In:
# -
# ### Variables Local:
# - `pca`: pca object
# - `np_pca`: numpy array of the N*7 numbers
# - `df_pca_washed`: add 3 columns, the projections onto 3 most significant axes
# - `base0`,`base1`,`base2`: base 
# %%
from sklearn.decomposition import PCA

np_pca = df_pca_washed[["condition",*organelles]].to_numpy()
pca = PCA(n_components=7)
pca.fit(np_pca)
pca_components = [comp if comp[0]>0 else -comp for comp in pca.components_ ]
df_components = pd.DataFrame(pca_components,columns=["condition",*organelles])
fig_parallel = px.parallel_coordinates(df_components,dimensions=["condition",*organelles],color=df_components.index,labels=df_components.columns)

base0 = pca_components[0]
base1 = pca_components[1]
base2 = pca_components[2]

df_pca_washed["proj0"] = df_pca_washed.apply(lambda x:np.dot(base0,x.loc[["condition",*organelles]]),axis=1)
df_pca_washed["proj1"] = df_pca_washed.apply(lambda x:np.dot(base1,x.loc[["condition",*organelles]]),axis=1)
df_pca_washed["proj2"] = df_pca_washed.apply(lambda x:np.dot(base2,x.loc[["condition",*organelles]]),axis=1)

# %%
# fig_parallel = go.Figure(data=
#     go.Bar(
#         base = dict(color = df_components.index,
#                    showscale = True),
#         dimensions = [dict(range=[-1,1],label=dim,values=df_components[dim]) for dim in df_components.columns]
#     )
# )
fig_parallel = px.bar()
fig_parallel.write_html("plots/pca_components.html")

fig01 = px.scatter(df_pca_washed,x="proj0",y="proj1",color="condition")
fig02 = px.scatter(df_pca_washed,x="proj0",y="proj2",color="condition")
fig12 = px.scatter(df_pca_washed,x="proj1",y="proj2",color="condition")

fig01.write_html("plots/proj01.html")
fig02.write_html("plots/proj02.html")
fig12.write_html("plots/proj12.html")

def pca_on_df(df:pd.DataFrame,fields:list,n_axes=7):
    np_pca = df[fields].to_numpy()
    pca = PCA(n_components=n_axes)
    pca.fit(np_pca)
    pca.components_
    return pca.components_

# %% [markdown]
# ## Minibatch of Input to PCA
# 
# Make a (4,3) subplot where the dataset is randomly split into fourths and plot the projections onto the first 3 axes. 
# 
# ### Variables In:
# - `np_pca`
# 
# ### Variables Local:
# - `np_idx`
# - `n_sample`: the number of sub-bathces
# - `fig,axes`: containers of the subplots
# - `sub_idx`: the indices of the batch
# - `sub_pca`: the subset of data that goes through the pca 
# %%
list_pca_components = [pca_components]
np.random.seed(42)
np_idx = np.random.permutation(np_pca.shape[0])
n_sample = int(len(np_pca)/4)
for i in range(4):
    sub_idx = np_idx[i*n_sample:min((i+1)*n_sample,len(np_pca))]
    sub_pca = np_pca[sub_idx,:]
    pca.fit(sub_pca)
    pca_components = [comp if comp[0]>0 else -comp for comp in pca.components_]
    list_pca_components.append(pca_components)
    base0 = pca_components[0]
    base1 = pca_components[1]
    base2 = pca_components[2]

    df_pca_washed["proj0"] = df_pca_washed.apply(lambda x:np.dot(base0,x.loc[["condition",*organelles]]),axis=1)
    df_pca_washed["proj1"] = df_pca_washed.apply(lambda x:np.dot(base1,x.loc[["condition",*organelles]]),axis=1)
    df_pca_washed["proj2"] = df_pca_washed.apply(lambda x:np.dot(base2,x.loc[["condition",*organelles]]),axis=1)
    
    df_pca_sub = df_pca_washed.iloc[sub_idx,:]

    fig01 = px.scatter(df_pca_sub,
                       x="proj0",y="proj1",
                       color="condition",
                       title=f"Batch {i+1}")
    fig02 = px.scatter(df_pca_sub,
                       x="proj0",y="proj2",
                       color="condition",
                       title=f"Batch {i+1}")
    fig12 = px.scatter(df_pca_sub,
                       x="proj1",y="proj2",
                       color="condition",
                       title=f"Batch {i+1}")
    fig01.write_html(f"plots/batch{i+1}_proj01.html")
    fig02.write_html(f"plots/batch{i+1}_proj02.html")
    fig12.write_html(f"plots/batch{i+1}_proj12.html")


# %%
df_many_components = [pd.DataFrame(many,columns=["condition",*organelles]) for many in list_pca_components]
for i,many in enumerate(df_many_components):
    many["batch"]=i-1
df_components_compare = pd.concat(df_many_components)
fig_parallel_compare = go.Figure(data=
    go.Parcoords(
        line = dict(color = df_components_compare.index,
                   showscale = True),
        dimensions = [dict(range=[-1,1],label=dim,values=df_components_compare[dim]) for dim in ["condition",*organelles]]+[dict(range=[-1,3],label="batch",values=df_components_compare["batch"])]+[dict(range=[0,6],label="index",values=df_components_compare.index)]
    )
)
fig_parallel_compare.write_html("plots/pca_components_compare.html")
# %%
