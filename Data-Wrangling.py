import pandas as pd
import zipfile
import numpy as np

zf = zipfile.ZipFile('../../deep-solar-dataset.zip')
#df = pd.read_csv(zf.open('deepsolar_tract.csv'))
df = pd.read_csv(zf.open('deepsolar_tract.csv'), encoding='latin-1')



#pd.options.display.max_seq_items = 2000



df = df.drop_duplicates(keep='first')
df = df.drop('Unnamed: 0', axis=1)



# transform float-strings to float
df.electricity_price_transportation = df.electricity_price_transportation.replace(' ', np.nan)
df.electricity_price_transportation = df.electricity_price_transportation.astype(float)
df.electricity_price_transportation = df.electricity_price_transportation.replace(np.nan, df.electricity_price_transportation.mean())



# change boolean columns to integers
df_nocat.voting_2012_dem_win = df_nocat.voting_2012_dem_win.astype(int)
df_nocat.voting_2016_dem_win = df_nocat.voting_2016_dem_win.astype(int)



df.columns
df.describe()
df.info()
df.sample(20)
len(df['county'].unique())
df.loc[df['county'] == 'Oklahoma County']
df.iloc[:,range(1,5)].sample(10)



# column types
[i for i, x in enumerate(list(df.dtypes)) if x == object]
df.iloc[:10, [i for i, x in enumerate(list(df.dtypes)) if x == object]]



# frequency table
from collections import Counter



# joins
pd.merge(df, df3, on='geolocation_zip_code_prefix', how='outer').head()



# change inf to nan, calculate mean, impute missing values with mean
for column in df_nocat.columns:
    df_nocat[column] = df_nocat[column].replace([np.inf, -np.inf], np.nan)
    colmean = df_nocat[column].mean()
    df_nocat[column] = df_nocat[column].replace(np.nan, colmean)
    
    
 
# no nans or infs
np.any(np.isnan(df_nocat))
np.all(np.isfinite(df_nocat))



# pca calculation
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
#from sklearn import preprocessing
pca = PCA(n_components = 2)
#pca.fit(X)
#pca.fit(preprocessing.normalize(df_nocat))
principalComponents = pca.fit_transform(StandardScaler().fit_transform(df_nocat))
#print(pca.explained_variance_ratio_)
#print(pca.singular_values_) 

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['pc1', 'pc2'])



# basic pca viz
import matplotlib.pyplot as plt
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
ax.scatter(principalDf.pc1, principalDf.pc2, c = colors, alpha=0.5)

ax.grid()



# bin continuous variable
tile_count_binned = np.digitize(df_nocat.tile_count, df_nocat.tile_count.quantile([1/3,2/3]))



# multi-color pca viz
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = [0, 1, 2]
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = tile_count_binned == target
    ax.scatter(principalDf.loc[indicesToKeep, 'pc1']
               , principalDf.loc[indicesToKeep, 'pc2']
               , c = color
               , s = 50
              , alpha = 0.05)
ax.legend(targets)
ax.grid()



# clustering
from sklearn.cluster import KMeans

# Number of clusters
kmeans = KMeans(n_clusters=2)
# Fitting the input data
kmeans = kmeans.fit(X)
# Getting the cluster labels
labels = kmeans.predict(X)
# Centroid values
centroids = kmeans.cluster_centers_



# feature correlations
import matplotlib.pyplot as plt
plt.matshow(df.corr())



import matplotlib.pyplot as plt
import seaborn as sns
corr = df_nocat.corr()
plt.style.use('bmh')
plt.figure(figsize=(12, 10))

sns.heatmap(corr[(corr >= 0.2) | (corr <= -0.2)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=False, annot_kws={"size": 8}, square=True);



from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
#scatter_matrix(df[['tile_count', 'solar_system_count']].sample(1000), alpha=0.2, figsize=(6, 6))
scatter_matrix(df_nocat.iloc[:,range(1,5)].sample(1000), alpha=0.2, figsize=(8, 8))
plt.show()



# defining dependent/independent variables; partitioning into training/test sets
Y1 = df['tile_count']
Y2 = df['solar_system_count']
Y3 = df['total_panel_area']
X = df_nocat.drop(['tile_count', 'solar_system_count', 'total_panel_area'], axis=1)



msk = np.random.rand(len(X)) < 0.9

Y1_train = Y1[msk]
Y1_test = Y1[~msk]

Y2_train = Y2[msk]
Y2_test = Y2[~msk]

Y3_train = Y3[msk]
Y3_test = Y3[~msk]

X_train = X[msk]
X_test = X[~msk]
#train_samples = random.sample(range(df.shape[0]))



# machine learning
from sklearn import linear_model
alpha_list_coarse = 10 ** np.arange(-10, 11, 1, dtype='float')
ridge = linear_model.RidgeCV(alphas = alpha_list_coarse, cv=10)
ridge.fit (X_train, Y1_train)

alpha_list_fine = linspace(ridge.alpha_/10, ridge.alpha_*10, num=101, endpoint=True, retstep=False)
ridge = linear_model.RidgeCV(alphas = alpha_list_fine, cv=10)
ridge.fit (X_train, Y_train)
ridge.fit (X_train, Y1_train)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=1000, max_features='sqrt', min_samples_leaf=round(len(X_train)/100), random_state=0)
rf.fit(X_train, Y1_train)



# evaluating predictions
predictions = rf.predict(X_test)
Y1_test.head(10)
predictions[:10]
plt.hist(predictions)
plt.hist(Y1_test)
