import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.utils
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.doc2vec import LabeledSentence,TaggedDocument
from gensim.models import Doc2Vec
from gensim import corpora
import scipy
import random
import time
import os,sys



def extract_tfid(doc):
  # Remove numbers in product name
  doc = doc.str.replace("[^a-zA-Z]", " ")
  vectorizer = TfidfVectorizer(ngram_range=(1,2),stop_words={'english'},min_df=0.001,max_df=1.0)
  x = vectorizer.fit_transform(doc)
  return x,vectorizer

def extract_w2v(doc,label):
  documents = doc
  model = gensim.models.doc2vec.Doc2Vec(
        dm=0, # DBOW
        hs=1,
        size=10, 
        alpha=0.01, 
        min_alpha=0.0001,
        window=15, 
        min_count=1,
        workers=8)
  sentences = [[word for word in document.split() if word not in STOPWORDS and len(word)>1] for document in documents]
  
  #####################################################
  bigram_transformer = gensim.models.Phrases(sentences)
  documents = [TaggedDocument(words = bigram_transformer[sentences[i]] if len(bigram_transformer[sentences[i]])>0 else [], tags = [i]) for i in range(len(sentences))]
  # build model
  model.build_vocab(documents)

  # train model
  for epoch in range(10):
    random.shuffle(documents)
    model.train(documents)
    print("epoch : ",epoch)
  

  model.save("test.model")
  model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
  return model,documents

def extractTextFeature(data,label=[],opt="tfid",split=False,random_state = 2000,save=False,GROUP = 0):
  x = data
  if(not split):
    if(opt=="tfid"):
      x,vectorizer = extract_tfid(x)
    elif(opt=="w2v"):
      x,token = extract_w2v(x,label)
    return x,token
  else:
    y = label
    #### split train test
    SSK = StratifiedKFold(n_splits=10,random_state=random_state)
    INDEX = []
    for train_index, test_index in SSK.split(x,y):
      INDEX.append({'train':train_index,'test':test_index})
    train = x[INDEX[GROUP]['train']]
    test = x[INDEX[GROUP]['test']]
    label_train = y[INDEX[GROUP]['train']]
    label_test = y[INDEX[GROUP]['test']]
    if(opt=="tfid"):
      train,vectorizer = extract_tfid(train)
      test = vectorizer.transform(test)
    elif(opt=="w2v"):
      train,vectorizer = extract_tfid(train)
      test = vectorizer.transform(test)
    return train,test,label_train,label_test

# # Specify input csv file
print("file")
print("coldstorage_path.csv == 1")
print("giant_path.csv == 2")
print("redmart_path.csv == 3")
input_file = input()
input_file = int(input_file)
if input_file == 1:
  input_file = "coldstorage_path.csv"
elif input_file == 2:
  input_file = "giant_path.csv"
elif input_file == 3:
  input_file = "redmart_path.csv"
img_root = input_file.replace("_path.csv","")+"_img"
print("SEED")
SEED = 2000
GROUP = int(input())
print(input_file)


df = pd.read_csv(input_file, header = 0)
# Subset dataframe to just columns category_path and name
df = df.loc[:,['category_path','name']]
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
df = df.reset_index(drop=True)
df['name'] = df['name'].str.replace("[^a-zA-Z]", " ")
df['name'] = df['name'].str.lower()

print("Uniqued df by name : "+str(len(df['name'])))
x,token = extractTextFeature(df['name'],label=df['category_path'],opt='w2v')

print(token[0][0])
v1=x.infer_vector(token[0][0])
x = Doc2Vec.load('test.model')
v2=x.infer_vector(token[0][0])
x = Doc2Vec.load('test.model')
v3=x.infer_vector(token[0][0])
print(v1)
print(v2)
print(v3)
# print(x.docvecs['sent_0'])
# for key in x.docvecs:
#   print(key)

