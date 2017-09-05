import jieba
import pandas as pd
import numpy as np
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
import time
import os
import re

def readFilesName(dir):
    name=[]
    for parent,dirnames,filenames in os.walk(dir):
        for filename in filenames:
            url=parent+filename
            name.append(url)
    return name

def readExcel(url):
    df=pd.read_excel(url,na_values='',index_col=None)
    return df

def combine_Abstract(df,df_Base,df_Test):
    '''combine abstract with same cat'''
    df=df.where(df.notnull(),1)
    dft=df[df['Cat']==1]
    df=df[df['Cat']!=1]
    df_Test=pd.concat([df_Test,dft],axis=0,ignore_index=True)
    df_Base=pd.concat([df_Base,df],axis=0,ignore_index=True)
    return df_Base, df_Test
    # for i in df.index:
    #     if df.loc[i,'Cat'] in df_Base['Cat'].values:
    #         index_Cat=df_Base[df_Base['Cat']==df.loc[i,'Cat']].index[0]
    #         df_Base.loc[index_Cat,'ABSTRACT']=df_Base.loc[index_Cat,'Cat']+df.loc[i,'ABSTRACT']
    #     else:
    #         item_new={'Cat':df.loc[i,'Cat'],'ABSTRACT':df.loc[i,'ABSTRACT']}
    #         df_Base=df_Base.append(item_new,ignore_index=True)
    # return

def digitize_Cat(df_Base):
    Cat=pd.DataFrame(df_Base['Cat'].unique().tolist(),columns=['Cat'])
    Cat['id']=Cat.index
    df_Base2=pd.merge(df_Base,Cat,'left',on='Cat')
    return df_Base2
    pass

def __TFgetjiebaSeries(ds):
    cor=[]
    for i in ds.index:
        b=jieba.cut(ds[i])
        subcor = []
        for c in b:
            subcor.append(c)
        joint = ' '
        subcor = joint.join(subcor)
        cor.append(subcor)
    return cor

def TFtrainandtest(x,y,clf=GaussianNB(),test_size=0.2,ada=0,nestimators=100):
    '''x，y,clf,test_size,ada=0为不使用ada，IPCmat默认不使用'''
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=test_size,random_state=0)
    xtraincor=__TFgetjiebaSeries(xtrain)
    xtestcor=__TFgetjiebaSeries(xtest)

    #获取分词矩阵
    word, weighttrain, vector, transformer = TFtraintfidf(xtraincor)
    weighttest = TFtesttfidf(xtestcor, vector, transformer)

    #ada boost
    if ada>0:
        clff=adaboost(clf,weighttrain.tolist(),ytrain.tolist(),learning_rate=1,n_estimators=nestimators)
    else:
        clff= TFtraining(weighttrain.tolist(), ytrain.tolist())

    y1test,y2test=TFtesting(weighttest.tolist(),ytest.tolist(),clff)
    y1train,y2train=TFtesting(weighttrain.tolist(),ytrain.tolist(),clff)
    comparetest=pd.DataFrame({'real':y1test,'predict':y2test,'same':(y1test-y2test)})
    comparetest.index = ytest.index
    comparetrain = pd.DataFrame({'real': y1train, 'predict': y2train, 'same': (y1train - y2train)})
    comparetrain.index = ytrain.index
    return comparetest,comparetrain
def TFtrainandtest_Real(xtrain,xtest,ytrain,clf=GaussianNB(),test_size=0.2,ada=0,nestimators=100):
    '''x，y,clf,test_size,ada=0为不使用ada，IPCmat默认不使用'''
    #xtrain,xtest,ytrain
    xtraincor=__TFgetjiebaSeries(xtrain)
    xtestcor=__TFgetjiebaSeries(xtest)

    #获取分词矩阵
    word, weighttrain, vector, transformer = TFtraintfidf(xtraincor)
    weighttest = TFtesttfidf(xtestcor, vector, transformer)

    #ada boost
    if ada>0:
        clff=adaboost(clf,weighttrain.tolist(),ytrain.tolist(),learning_rate=1,n_estimators=nestimators)
    else:
        clff= TFtraining(weighttrain.tolist(), ytrain.tolist())

    ypredict=clff.predict(weighttest.tolist())


    return ypredict

def TFtesttfidf(cor,vectorizer,transformer):
    '''根据已经获取的，计算测试集的权重'''
    test = transformer.transform(vectorizer.transform(cor))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵  
   # word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语  
    weight = test.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重 
    return weight
def TFtraintfidf(cor):
    '''获取词语集合，以及权重'''
    vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频  
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值  
    train = transformer.fit_transform(vectorizer.fit_transform(cor))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵  
    word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语  
    weight = train.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重  
    return word,weight,vectorizer,transformer
def adaboost(clf,X_train,y_train,learning_rate=1,n_estimators=100):
    ada_real = AdaBoostClassifier(base_estimator=clf, learning_rate=learning_rate, n_estimators=n_estimators,
                                  algorithm='SAMME.R')
    ada_real.fit(X_train, y_train)
    return ada_real
def TFtraining(x,y,clf=MultinomialNB()):
    clff = clf.fit(x, y)
    return clff
def TFtesting(x,y,clf):
    ypredict=clf.predict(x)
    return y,ypredict


if __name__=="__main__":
    #1.inial
    docdir = os.path.abspath('..') + '\\doc\\'
    name=readFilesName(docdir+'mid\\')
    df_Base = pd.DataFrame(None, columns=['Cat', 'ABSTRACT'])
    df_Test = pd.DataFrame(None, columns=['Cat', 'ABSTRACT'])
    #2.get the ready Data
    for i in name:
        df=readExcel(i)
        df_Base,df_Test= combine_Abstract(df,df_Base,df_Test)

    #3.train the data
    df_Base=digitize_Cat(df_Base)
    ypredict = TFtrainandtest_Real(xtrain=df_Base['ABSTRACT'],xtest=df_Test['ABSTRACT'], ytrain= df_Base['id'],
                                       clf=GaussianNB(), test_size=0.7, ada=1, nestimators=100)

    #4.export result
    


    pass