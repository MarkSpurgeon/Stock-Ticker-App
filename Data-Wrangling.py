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


# change inf to nan, calculate mean, impute missing values with mean
for column in df_nocat.columns:
    df_nocat[column] = df_nocat[column].replace([np.inf, -np.inf], np.nan)
    colmean = df_nocat[column].mean()
    df_nocat[column] = df_nocat[column].replace(np.nan, colmean)
    
    
 
# no nans or infs
np.any(np.isnan(df_nocat))
np.all(np.isfinite(df_nocat))



