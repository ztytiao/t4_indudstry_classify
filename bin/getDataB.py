import jieba
import pandas as pd
import numpy as np
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import time
import os
import re

def readExcel(url):
    df=pd.read_excel(url,na_values='')
    return df

def writeExcel(df,url=''):
    write = pd.ExcelWriter(url)
    df.to_excel(write, sheet_name='Sheet1',merge_cells=False)
    write.save()

def genDF(df_Base,df_IPC):
    columns=df_Base['合并后'].unique()
    index=df_IPC['NAME'].unique()
    return pd.DataFrame(None,index=index,columns=columns)
    pass

def checkVect(df_Input,vectkey,vectipc):
    coripc=__ipc(df_Input['IPC'])
    corkey=__jiebabatch(df_Input['ABSTRACT'])
    matkey=vectkey.transform(corkey)
    matipc=vectipc.transform(coripc)
    matkey = np.where(matkey.toarray() > 0, 1, 0)
    matipc = np.where(matipc.toarray() > 0, 1, 0)
    return matkey,matipc

def addNewPT(key,newkey,ipc,newipc,Cpy,Cat,df_result=[],s_Cat=[],corkeyNo=[]):
    '''输出两个：1. 公司各个行业属性的专利记录，2. 专利对应的属性'''
    if df_result==[]:
        df_result=pd.DataFrame(None,columns=['Company','Cat','Val'])
    for i in range(newkey.shape[0]):
        print('Processing--------------',i)
        onekey=newkey[i]
        oneipc=newipc[i]
        companyName=Cpy[i]
        mat=np.dot(key, onekey) * np.dot(ipc, oneipc)
        matcat=Cat[(mat-corkeyNo)>=0].unique()
        cat=[]#用于记录ip的属性
        for j in matcat:
            df_result=df_result.append({'Company':companyName,'Cat':j,'Val':1},ignore_index=True)
            cat.append(j)
        s_Cat.append(' '.join(cat))
    return df_result,s_Cat
    pass


def __jiebabatch(summary):
    '''每行增加none，使得没有的也能匹配上'''
    cor=[]
    for j,i in enumerate(summary):
        if j%1000==0:
            print('ABSTRACT@----------',j)
        subcor = ['None']
        if type(i)!=type(1.1):
            b=jieba.cut(i)
            for j in b:
                subcor.append(j)
            joint=' '
            subcor=joint.join(subcor)
        else:
            subcor='None'
        cor.append(subcor)
    return cor
    pass

def __ipc(ipc):
    '''每行增加none，使得没有的也能匹配上'''
    cor=[]
    lv0=re.compile(u'\D+[0-9]{1,3}')
    lv1=re.compile(u'\D+[0-9]{1,3}\D+')
    lv2=re.compile(u'\D+[0-9]{1,3}\D+[0-9]{1,3}')
    for j,i in enumerate(ipc):
        if j%1000==0:#todo
            print('IPC@---------',j)
        subcor=['none']
        if type(i)!=type(1.1):#写入ipc三级域名  modified--0828
            subcor.append(lv0.findall(i)[0])
            subcor.append(lv1.findall(i)[0])
            if lv2.findall(i):
                subcor.append(lv2.findall(i)[0])
            a = i.split(sep='_')
            if len(a)>1:
                subcor.append(i)
        joint = ' '
        subcor = joint.join(subcor)
        cor.append(subcor)
    return cor
    pass

def addDict(df):
    '''根据分类标准在结巴分词中加入新关键词'''
    word=set(df['keywords'].unique())
    word1=set(df['keywords1'].unique())
    word2 = set(df['keyward2'].unique())
    word3 = set(df['notkeywords'].unique())
    word=word | word1 | word2 | word3
    for i in word:
        jieba.add_word(str(i),freq=100)
    pass

def getBaseCor(df):
    '''用于tdidf'''
    corkey=[]
    coripc = []
    corkeyNo=[]
    for i in df.index:
        subcor=[]
        no=0
        if type(df.loc[i,'keywords'])!=type(1.1):
            subcor.append(df.loc[i,'keywords'])
            no=no+1
            if type(df.loc[i, 'keywords1']) != type(1.1):
                subcor.append(df.loc[i, 'keywords1'])
                no=no+1
        else:
            subcor=['None']
            no=1
        joint=' '
        subcor=joint.join(subcor)
        corkey.append(subcor)
        corkeyNo.append(no)

        subcor=[]
        if type(df.loc[i, 'IPC']) != type(1.1):
            subcor.append(df.loc[i, 'IPC'])
            # a=df.loc[i,'IPC'].split(sep='/')
            # if len(a)>1:
            #     subcor.append(a[0])
        else:
            subcor=['None']
        joint=' '
        subcor=joint.join(subcor)
        coripc.append(subcor)

    return corkey,coripc,np.array(corkeyNo)

def TFtraintfidf(corkey,coripc):
    '''计算分词矩阵'''
    vectorizerkey = CountVectorizer(analyzer='word',token_pattern='(?u)\\b\\w+\\b')
    #CountVectorizer(analyzer='word',token_pattern='(?u)\\b\\w\\w+\\b')
    key=vectorizerkey.fit_transform(corkey)
    vectorizeripc = CountVectorizer()
    ipc=vectorizeripc.fit_transform(coripc)
    return np.where(key.toarray()>0,1,0),vectorizerkey,np.where(ipc.toarray()>0,1,0),vectorizeripc

def readFilesName(dir):
    name=[]
    for parent,dirnames,filenames in os.walk(dir):
        for filename in filenames:
            url=parent+filename
            name.append(url)
    return name

if __name__=="__main__":
    #1.inial
    docdir = os.path.abspath('..') + '\\doc\\'
    name=readFilesName(docdir+'raw\\')
    IndustryFileName='industry_sep.xlsx'
    print('Initialize Finished')#todo

    #df_IPC=readExcel(docdir+IPCFileName)
    #2.read file
    for j,i in enumerate(name):
        df_Input=readExcel(i)
        df_Input['IPC']=df_Input['IPC'].str.replace('/','_')
        df_Base=readExcel(docdir+IndustryFileName)
        df_Base['IPC']=df_Base['IPC'].str.replace('/','_')
        addDict(df_Base)

        print('Data_Read_Finished')#todo
        #3.run
        start=time.clock()
        corkey, coripc,corkeyNo=getBaseCor(df_Base)
        print('Classify Finished')#todo
        key, vectkey, ipc, vectipc=TFtraintfidf(corkey,coripc)
        newkey,newipc=checkVect(df_Input,vectkey,vectipc)
        print('Analysising new input finished')#todo
        Cpy=df_Input['APPLICATION']
        Cat=df_Base['合并后']
        w,cat=addNewPT(key, newkey, ipc, newipc, Cpy, Cat, df_result=[],corkeyNo=corkeyNo)

        end=time.clock()
        dur=end-start

        df_f=w.groupby(['Company','Cat'])['Cat'].count()
        #4.export excel
        #for check
        df_Input['Cat']=cat
        writeExcel(df_Input, docdir + 're\\check0828' + str(j) + '.xlsx')
        writeExcel(df_f,docdir+'re\\result0828'+str(j)+'.xlsx')
        print(dur)


