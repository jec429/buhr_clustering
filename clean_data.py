import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from preprocessing import OneHotEncoder

np.set_printoptions(threshold=np.inf)


# df = pd.read_excel('EERosterWFDNAandBUHRPartnerswTenure.xlsx', sheet_name='WFDNA and BUHR')
df = pd.read_excel('test.xlsx', sheet_name='Sheet1')
df.info()
df = df.drop('Employee Count', axis=1)
df = df.drop('MonthYear', axis=1)
print(df['BUHR Partner'].value_counts())
BUHRs = df['BUHR Partner'].astype('category').cat.codes
df = df.drop('BUHR Partner', axis=1)
df.drop_duplicates(subset="WWID", keep='last', inplace=True)

data_x = df

for c in data_x.columns:
    if data_x[c].dtype == object:
        data_x[c] = data_x[c].fillna('Missing')
        print(c)
        data_x[c] = data_x[c].astype('category')
    else:
        data_x[c] = data_x[c].fillna(-999)

data_x['Years of Service Buckets'] = data_x['Years of Service Buckets'].cat.codes

data_x_numeric = OneHotEncoder().fit_transform(df)

# data_x_numeric.info()

kmeans = KMeans(n_clusters=4)
kmeans.fit(data_x_numeric)
# print(kmeans.cluster_centers_)
print(kmeans.labels_)
print(np.array(BUHRs))
