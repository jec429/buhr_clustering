import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from preprocessing import OneHotEncoder


# np.set_printoptions(threshold=np.inf)


df = pd.read_excel('EERosterWFDNAandBUHRPartnerswTenure.xlsx', sheet_name='WFDNA and BUHR')
# df = pd.read_excel('test.xlsx', sheet_name='Sheet1')
df.info()
df = df.drop('Employee Count', axis=1)
df = df.drop('MonthYear', axis=1)
df = df.drop('Terminated?', axis=1)
df = df.drop('Termination Reason', axis=1)
df = df.drop('Manager Name (Direct Reports)', axis=1)
df['Location Code'] = df['Location Code'].str.split(' ').str.get(0).str.strip()
df['Pay Grade'] = df['Pay Grade'].str.replace('51', '100').str.replace('+', '').astype(int)
df.drop_duplicates(subset="WWID", keep='last', inplace=True)

df2 = pd.read_excel('RG0029_Workers_with_HRBP_2019-08-13_14_41_IST.xlsx', skiprows=8)
df2['WWID'] = df2["Worker's WWID"]
df2 = df2.drop("Worker's WWID", axis=1)
df2 = df2.drop("Worker's Legal Name", axis=1)
df2 = df2.drop("Worker", axis=1)
df2 = df2.drop("Worker's Manager", axis=1)
df2 = df2.drop("MRC", axis=1)


df3 = pd.merge(df, df2, on='WWID', how='outer')
df3 = df3.drop("WWID", axis=1)

print(df3['BUHR Partner'].value_counts())
BUHRs = df3['BUHR Partner'].astype('category').cat.codes
df3 = df3.drop('BUHR Partner', axis=1)
df3 = df3.drop('HRBP', axis=1)
df3 = df3.drop('HRBP Country', axis=1)

df3.info()
data_x = df3[:100]

for c in data_x.columns:
    if data_x[c].dtype == object:
        data_x[c] = data_x[c].fillna('Missing')
        print(c)
        data_x[c] = data_x[c].astype('category')
        #data_x[c] = data_x[c].cat.codes
    else:
        data_x[c] = data_x[c].fillna(-999)

data_x['Years of Service Buckets'] = data_x['Years of Service Buckets'].cat.codes


data_x_numeric = OneHotEncoder().fit_transform(data_x)

data_x_numeric.info()
print(data_x_numeric.head())

# sys.exit()


kmeans = KMeans(n_clusters=40)
kmeans.fit(data_x_numeric)
# print(kmeans.cluster_centers_)
print(kmeans.labels_)
print(np.array(BUHRs))
