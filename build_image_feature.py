from extractImage import *
from precompute import *
from sklearn.preprocessing import LabelEncoder

store = "coldstorage"
input_file = store+"_path.csv"

img_root = store+"_img"
df = pd.read_csv(input_file, header = 0)
### Preprocessing  start ###
###
# Subset dataframe to just columns category_path and name
df = df.loc[:,['category_path','name','img_file']]
# Make a duplicate of input df
df_original=df
df_dedup=df.drop_duplicates(subset='name')
# print(len(np.unique(df_dedup['name'])))
df=df_dedup
#drop paths that have 1 member
df_count = df.groupby(['category_path']).count()
df_count = df_count[df_count == 1]
df_count = df_count.dropna()
df = df.loc[~df['category_path'].isin(list(df_count.index))]
df = df.drop(df[df['category_path'].isnull()].index)
df = df.reset_index(drop=True)
print("Uniqued df by name : "+str(len(df['name'])))
####### label encoder
number = LabelEncoder()
y = df['category_path']
x = df['img_file']
y = y.astype(str)

# cat_map = build_heirarchy_label(y) # use for label mapping
# cat_map.to_csv(store+"_map.csv") 
cat_map = pd.read_csv(store+"_map.csv", header = 0)
y = y.str.split("->")
y = map_label(y,cat_map)
y = y.reset_index()
y,remove_index = reduce_hierachical_class(y)
print(y)
print(remove_index)
##### level 3
y_level = y[(y['third_name']!=99)&(y['third_value']!=99)]
groupby = ['top_name','second_name','third_name']
freq_y = pd.DataFrame(y_level.groupby(groupby).size(),columns=['size'])
freq_y['size'] = freq_y['size']/len(y)
freq_y = freq_y.sort('size',ascending=False)
print(freq_y)
print(freq_y[freq_y['size']<0.01])
###### level 2
y_level = y[(y['second_name']!=99)&(y['second_value']!=99)]
groupby = ['top_name','second_name']
freq_y = pd.DataFrame(y_level.groupby(groupby).size(),columns=['size'])
freq_y['size'] = freq_y['size']/len(y)
freq_y = freq_y.sort('size',ascending=False)
print(freq_y)
print(freq_y[freq_y['size']<0.01])
###### level 1
y_level = y
groupby = ['top_name']
freq_y = pd.DataFrame(y_level.groupby(groupby).size(),columns=['size'])
freq_y['size'] = freq_y['size']/len(y)
freq_y = freq_y.sort('size',ascending=False)
print(freq_y)
print(freq_y[freq_y['size']<0.01])
### reduce x follow y
x = x.reset_index()
x = x.drop(x.index[remove_index])
print(x[0:10])

# y = number.fit_transform(y)
print(np.shape(df['img_file']))
for i in range(0,10):
  train,test,label_train,label_test = extractImageFeature(x,img_root,label=y,opt="sift",split=True,random_state=2000,save=True,GROUP=i)