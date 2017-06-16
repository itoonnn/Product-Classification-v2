#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from anytree import Node, RenderTree, AsciiStyle
from sklearn.preprocessing import Normalizer
import operator,sys

def uprint(*objects, sep=' ', end='\n', file=sys.stdout):
    enc = file.encoding
    if enc == 'UTF-8':
        print(*objects, sep=sep, end=end, file=file)
    else:
        f = lambda obj: str(obj).encode(enc, errors='backslashreplace').decode(enc)
        print(*map(f, objects), sep=sep, end=end, file=file)

def build_heirarchy_label(label):
  cat_map = pd.DataFrame(columns=[
    'level',
    'node_parent',
    'node_name',
    'node_label'
  ])
  seq_label = label.str.split("->")
  idx = 0
  for cat in seq_label:
    parent = None
    for i,node in enumerate(cat):
      if(len(cat_map[cat_map['node_name']==node])==0):
        item = {
          'level':i,
          'node_parent':parent,
          'node_name':node,
          'node_label':idx
        }
        cat_map = cat_map.append(item,ignore_index=True)
        # print(cat_map)
        idx+=1
      parent = cat_map[cat_map['node_name']==node]['node_label'].values[0]
  # uprint(RenderTree(rootNum,style=AsciiStyle()))
  # uprint(RenderTree(root,style=AsciiStyle()))
  return cat_map
def map_label(y,cmap):
  print("mapping label")
  y_map = []
  for i in range(len(y)):
    y_map.append(cmap[cmap['node_name']==y[i][-1]]['node_label'].values[0])
  return y_map


def reduce_class(x,y,threshold = 0.01,other=False):
  print("Reduce Class")
  y_size = len(y)
  freq_y = Counter(y)
  freq_y = sorted(freq_y.items(), key=operator.itemgetter(1),reverse=True)
  SUM = 0
  count = 0
  removed_class = []
  ## fy[0] = class, fy[1] = freq
  for fy in freq_y:
    freq = fy[1]/y_size
    if(freq < threshold):
      count += 1
      SUM += fy[1]
      removed_class.append(fy[0])
      # print(fy,freq)

  print("exist class : ",len(freq_y)-count)
  print("remove amount : ",SUM)
  print("remove rate : ",SUM/y_size)
  print("removed class\n",removed_class)
  for i in range(y_size):
    if y[i] in removed_class:
      if(other):     ############ other
        y[i] = 9999.0
      else:
        y[i] = None
        x[i] = None
  if(other):         ############ other
    for i in range(len(y_test)):
      if y_test[i] in removed_class:
        y_test[i] = 9999.0
  x = x[~np.isnan(x).all(1)]
  y = y[~np.isnan(y)]
  if(other):
    return x,y,y_test
  else:
    return x,y

def feature_selection(train,test,threshold = 0.9):
  print("PCA")
  pca = PCA(svd_solver='full',random_state=2000)
  pca = pca.fit(train)
  explained_variance = pca.explained_variance_ratio_
  SUM = 0
  n_components = 0
  for var in explained_variance:
    if(SUM <= threshold):
      SUM += var
      n_components += 1
    else:  
      break
  print(SUM,n_components)
  pca = PCA(n_components = n_components, svd_solver='full', random_state=2000)
  train = pca.fit_transform(train)
  test = pca.transform(test)
  return train,test


def reduceHCProcess(y,level,cat_map,threshold = 0.01):
  y_size = len(y)
  reduce_index = []
  print("### LEVEL ",level," ###")

  y_level = y[y['level']==level]
  if(len(y_level)<=0):                                                                 # if that level has no item, return itself
    return y,reduce_index
  freq_y = pd.DataFrame(y_level.groupby(['node_label']).size(),columns=['size'])
  freq_y['size_ratio'] = freq_y['size']/y_size
  freq_y = freq_y.sort('size',ascending=False)
  print("Small Classes")
  reduce_freq_y = freq_y[freq_y['size_ratio']<threshold]
  SUM = reduce_freq_y['size_ratio'].sum()
  print("sumerized freq :",SUM)
  print("shape :",np.shape(reduce_freq_y))
  if(SUM > 0): ### Terminate Condition/
  #   break
    reduce_class = reduce_freq_y.reset_index(level=['node_label'])
    reduce_data = y.join(reduce_class.set_index('node_label'), on='node_label')
    reduce_data = reduce_data.dropna()
    print("================== reduce_data ================")
    # print("class:",reduce_class)
    # print("data:",reduce_data.index)
    if level > 0:
      for i in reduce_data.index:
        # if(y.loc[i,'level']!=level):
        y.loc[i,'node_label'] = y.loc[i,'node_parent']
        y.loc[i,'node_name'] = cat_map[cat_map['node_label']==y.loc[i,'node_label']]['node_name'].values[0]
        y.loc[i,'node_parent'] = cat_map[cat_map['node_label']==y.loc[i,'node_label']]['node_parent'].values[0]
        y.loc[i,'level'] = level-1
        if(level-1 == 0):
          y.loc[i,'node_parent'] = None
    else:
      y = y.drop(y.index[reduce_index])
      print("Number of Removed rows:",len(reduce_index))
    print("==================================")
  
  # print(pd.DataFrame(y.groupby(['node_label']).size(),columns=['size']))
  return y,reduce_index

def reduce_hierachical_class(y,cat_map,threshold = 0.01,other=False):
  print("Reduce Class")
  # y = y.reset_index()
  # y = y.fillna(99)
  y = pd.DataFrame(y,columns=['node_label'])
  y = y.join(cat_map.set_index('node_label'), on='node_label')
  y = y[['node_label','node_name','node_parent','level']]
  for level in range(2,-1,-1):
    y,remove_index = reduceHCProcess(y,level,cat_map,threshold = threshold)
  # # print(y)
  return y,remove_index
def dump_hire_file(y_map,output):
  print(pd.DataFrame(y_map.groupby(['node_label']).size(),columns=['size']))
  y_map = y_map[~np.isnan(y_map['node_parent'])][['node_parent','node_label']]
  print(y_map)  
  y_map = y_map.drop_duplicates(subset='node_label')
  y_map.to_csv(output,header=False,index=False,sep=' ') 


def feature_selection(train,test,threshold = 0.9):
  print("PCA")
  pca = PCA(svd_solver='full',random_state=2000)
  pca = pca.fit(train)
  explained_variance = pca.explained_variance_ratio_
  SUM = 0
  n_components = 0
  for var in explained_variance:
    if(SUM <= threshold):
      SUM += var
      n_components += 1
    else:  
      break
  print(SUM,n_components)
  pca = PCA(n_components = n_components, svd_solver='full', random_state=2000)
  train = pca.fit_transform(train)
  test = pca.transform(test)
  return train,test

def split_data(x,y,random_state=2000):
  SSK = StratifiedKFold(n_splits=10,random_state=random_state)
  INDEX = []
  for train_index, test_index in SSK.split(x,y):
    INDEX.append({'train':train_index,'test':test_index})
  # train = x[INDEX[GROUP]['train']]
  # test = x[INDEX[GROUP]['test']]
  # label_train = y[INDEX[GROUP]['train']]
  # label_test = y[INDEX[GROUP]['test']]
  return INDEX

  
# def feature_selection(train,test):
#   score_before = 0
#   rc_before = 0
#   best_rc = 1
#   best_n_component = 0
#   for i in range(10,100,5):
#     pca = PCA(n_components=int(np.shape(train)[1]*(i/100.0)),svd_solver='full',random_state=2000)
#     score = np.mean(cross_val_score(pca, train))
#     rc = (score-score_before)/score
#     rrc = (rc-rc_before)/rc
#     print((i/100.0),"\t",score,"\t",rc,"\t",rrc)

#     if rc < 0.01:
#       break
#     else:
#       best_rc,best_n_component = rc,(i/100.0)
#     rc_before = rc
#     score_before = score
#   print(best_n_component,"\t",best_rc)
#   pca = PCA(n_components=int(np.shape(train)[1]*best_n_component),svd_solver='full',random_state=2000)
#   train = pca.fit_transform(train)
#   test = pca.transform(test)
#   return train,test
