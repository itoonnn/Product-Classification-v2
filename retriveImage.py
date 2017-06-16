# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os,sys
import urllib.request as ur
from time import sleep

FILE = "coldstorage.csv"
FOLDER_NAME = "coldstorage_img"

try:
  os.mkdir(FOLDER_NAME)
except WindowsError:
  pass

def retreiveImg(url,folder):
  fname = url.split("/")[-1]
  fname = fname.split("?")[0]
  f = open(folder+"/"+fname,'wb')
  req = ur.Request(url)
  req.add_header("User-Agent", "Mozilla/5.0 (X11; U; Linux i686) Gecko/20071127 Firefox/2.0.0.11")
  f.write(ur.urlopen(req).read())
  f.close()

dataFile = pd.read_csv(FILE, header = 0, encoding = "ISO-8859-1")
#add new column
temp = pd.DataFrame()
temp['img_file'] = dataFile['image'].str.split("/").str[-1]
temp = temp['img_file'].str.split("?").str[0]
dataFile['img_file'] = temp
dataFile.to_csv(FILE)
#drop duplicate
dataFile = dataFile.loc[:,['image','name']]
dataFile_dedup=dataFile.drop_duplicates(subset='name')
dataFile=dataFile_dedup
dataFile = dataFile.reset_index(drop=True)
# print(dataFile['image'][0].split("/")[-1])
for imgurl in dataFile['image']:
  print(imgurl)
  try:
    fname = imgurl.split("/")[-1]
    fname = fname.split("?")[0]
    if(os.path.isfile(FOLDER_NAME+"/"+fname)):
      print("EXISTING")
    else:
      # sleep(3)
      retreiveImg(imgurl,FOLDER_NAME)
      print("DOWNLOAD...")
  except:
    print("ERROR")
    pass