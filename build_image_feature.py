from extractImage import *
from precompute import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold



store = "coldstorage"
input_file = store+".csv"
opt="sift"

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
x = df[['img_file','name']]
y = y.astype(str)

## init mapping
cat_map = build_heirarchy_label(y) # use for label mapping
cat_map.to_csv(store+"_"+opt+"_map.csv") 
cat_map = pd.read_csv(store+"_"+opt+"_map.csv", header = 0)

y = y.str.split("->")
y_map = map_label(y,cat_map)
# y = y.reset_index()
y_map,remove_index = reduce_hierachical_class(y_map,cat_map)
print(y_map)

## re-mapping
dump_hire_file(y_map,"image_feature/"+store+"_"+opt+"_map.txt")

### reduce x follow y_map
x = x.reset_index()
x = x.drop(x.index[remove_index])
y = y_map['node_label']
x_img = img2vec(x['img_file'],img_root,opt='contextual',random_state = 2000)
x_txt = x['name']


INDEX = split_data(x,y)

# y_map = number.fit_transform(y_map)
for i in range(0,10):

  train_img = x[INDEX[GROUP]['train']]
  test_img = x[INDEX[GROUP]['test']]
  label_train = y[INDEX[GROUP]['train']]
  label_test = y[INDEX[GROUP]['test']]
  train,test,label_train,label_test = extractImageFeature(train_img,test_img,label_train,label_test,img_root,opt=opt)