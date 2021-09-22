import pandas as pd

df = pd.read_csv('April 17, 2021.csv')
df.info()
df.columns

df_raw = df[['RequestID', 'StartDate', 'EndDate', 'AgencyName',
       'CategoryDescription',
       'SelectionMethodDescription', 'ContractAmount']]


# Function for searching the short notation by using RequestID number:
df_shorttitle = df[['RequestID','ShortTitle']]
def notation(ID_num):
    filt = df_shorttitle['RequestID'] == ID_num
    print(df_shorttitle.loc[filt])

notation(20071004002)

# Clean dataset

df_raw.isnull().sum()
df_raw.shape
df_clean = df_raw.dropna(how = 'any')
df_clean = df_clean[df_clean['ContractAmount'] > 0]
df_clean.info()
df_clean.columns
df_clean.to_csv('Data/cleandataset.csv')
df_clean = pd.read_csv('Data/cleandataset.csv')

# Bids counted by category:
CategoryDescription = df_clean['CategoryDescription'].value_counts()
categorysummary_df = pd.DataFrame({'Counts of Bids':CategoryDescription})
categorysummary_df.reset_index(inplace=True)
categorysummary_df.rename(columns={'index':'Category'},inplace=True)
categorysummary_df.to_csv('Data/CategorySummary.csv',index = False)

# group by Category Description

category_group = df_clean.groupby(['CategoryDescription'])
Goods = category_group.get_group('Goods')
Goods.value_counts()
Goods.to_csv('Data/Goods.csv', index = False)
HumanClientServices = category_group.get_group('Human Services/Client Services')

OtherServices = category_group.get_group('Services (other than human services)')
Construction = category_group.get_group('Construction/Construction Services')
GoodsServices = category_group.get_group('Goods and Services')
ConstructionServices = category_group.get_group('Construction Related Services')

# Analyze by category:
# Selection Methods for each category
# BM_goods_df
BM = Goods['SelectionMethodDescription'].value_counts(normalize=True)
BM_goods_df = pd.DataFrame({'Goods':BM})

# BM_HumanClientServices_df
BM = HumanClientServices['SelectionMethodDescription'].value_counts(normalize=True)
BM_HumanClientServices_df = pd.DataFrame({'HumanClientServices':BM})

# BM_OtherServices_df
BM = OtherServices['SelectionMethodDescription'].value_counts(normalize=True)
BM_OtherServices_df = pd.DataFrame({'OtherServices':BM})

# BM_Construction_df
BM = Construction['SelectionMethodDescription'].value_counts(normalize=True)
BM_Construction_df = pd.DataFrame({'Construction':BM})

# BM_GoodsServices_df
BM = GoodsServices['SelectionMethodDescription'].value_counts(normalize=True)
BM_GoodsServices_df = pd.DataFrame({'GoodsServices':BM})

# BM_ConstructionServices_df
BM = ConstructionServices['SelectionMethodDescription'].value_counts(normalize=True)
BM_ConstructionServices_df = pd.DataFrame({'ConstructionServices':BM})

# Combined Results
result = pd.concat([BM_goods_df, BM_HumanClientServices_df, BM_OtherServices_df, BM_Construction_df,BM_GoodsServices_df,BM_ConstructionServices_df], axis=1)
result.to_csv('Data/SelectionMethods.csv',index = True)


