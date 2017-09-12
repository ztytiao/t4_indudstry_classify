import jieba
import pandas as pd
import random
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
    df['ABSTRACT']=df['ABSTRACT'].astype(str)
    df['ABSTRACT']=df['ABSTRACT']+' END'
    return df

def combine_Abstract(df,df_Base,df_Test):
    '''combine abstract with same cat'''
    df=df.where(df.notnull(),1)
    dft=df[df['Cat']==1]
    df=df[df['Cat']!=1]

    df_Base=pd.concat([df_Base,df],axis=0,ignore_index=True)
    df_Test=pd.concat([df_Test,dft],axis=0,ignore_index=True)
    return df_Base, df_Test
def combine_Abstract_B(df,df_Base,df_Test):
    '''combine abstract with same cat'''
    df=df.where(df.notnull(),1)
    dft=df[df['Cat']==1]
    df=df[df['Cat']!=1]
    df_Temp=pd.DataFrame()
    for i in df.index:
        cat=df.loc[i,'Cat'].split(' ')
        for j in cat:
            df_Temp=df_Temp.append({'ID':df.loc[i,'ID'],
                                    'TITLE':df.loc[i,'TITLE'],
                                    'ABSTRACT':df.loc[i,'ABSTRACT'],
                                    'APPLICATION':df.loc[i,'APPLICATION'],
                                    'IPC':df.loc[i,'IPC'],
                                    'Cat': j
                                    },ignore_index=True)
        pass
    df_Base=pd.concat([df_Base,df_Temp],axis=0,ignore_index=True)
    df_Test=pd.concat([df_Test,dft],axis=0,ignore_index=True)
    return df_Base, df_Test

def digitize_Cat(df_Base):
    Cat=pd.DataFrame(df_Base['Cat'].unique().tolist(),columns=['Cat'])
    Cat['id']=Cat.index
    df_Base2=pd.merge(df_Base,Cat,'left',on='Cat')
    return df_Base2,Cat
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

def TFtrain(xtrain,ytrain,clf=GaussianNB(),test_size=0.2,ada=0,nestimators=100):
    '''x，y,clf,test_size,ada=0为不使用ada，IPCmat默认不使用'''
    #xtrain,xtest,ytrain
    xtraincor=__TFgetjiebaSeries(xtrain)
    #xtestcor=__TFgetjiebaSeries(xtest)

    #获取分词矩阵
    word, weighttrain, vector, transformer = TFtraintfidf(xtraincor)
    #weighttest = TFtesttfidf(xtestcor, vector, transformer)

    #ada boost
    if ada>0:
        clff=adaboost(clf,weighttrain.tolist(),ytrain.tolist(),learning_rate=1,n_estimators=nestimators)
    else:
        clff= TFtraining(weighttrain.tolist(), ytrain.tolist())
    #ypredict = clff.predict(weighttest.tolist())
    return clff,vector,transformer

def TFtesttfidf(cor,vectorizer,transformer):
    '''根据已经获取的，计算测试集的权重'''
    if vectorizer.transform(cor).shape[0]<1:
        weight=np.zeros([cor.__len__(),vectorizer.transform(cor).shape[1]])
        pass
    else:
        test = transformer.transform(vectorizer.transform(cor))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵  
       # word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语  
        weight = test.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重 
    return weight
def TFtraintfidf(cor):
    '''获取词语集合，以及权重'''
    vectorizer = CountVectorizer()  # TODO 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频  
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值  
    train = transformer.fit_transform(vectorizer.fit_transform(cor))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵  
    word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语  
    print('Size of feature is',len(word))
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

def SplitTrainning(df_Base,size_Train,df_Predict,size_Predict,Cat,repeatNo):
    '''预测'''
    del df_Predict['Cat']
    df=pd.DataFrame(index=df_Predict['ID'],columns=['TITLE','predict'])
    df['predict']=' '
    df['TITLE']=df_Predict['TITLE'].tolist()
    batch_Predict=int(df_Predict.shape[0]/size_Predict)#多批次预测
    for j in range(repeatNo):
        '''构建多个小分类器'''
        print('Now Traning the ',j,' th classiflier')
        tic=time.clock()
        toc=time.clock()
        Y=[]
        indSeleted = random.sample(df_Base.index.tolist(), size_Train)
        df_Train = df_Base.loc[indSeleted]
        clff, vector, transformer=TFtrain(xtrain=df_Train['ABSTRACT'],  ytrain=df_Train['id'],
                                       clf=GaussianNB(), test_size=0.7, ada=1, nestimators=50)
        print('***************Trainning finished, and Start predicting***************')
        for i in range(batch_Predict):
            indSeleted2=range(i*size_Predict,(i+1)*size_Predict)
            df_T=df_Predict.iloc[indSeleted2]
            xtestcor = __TFgetjiebaSeries(df_T['ABSTRACT'])
            weighttest = TFtesttfidf(xtestcor, vector, transformer)
            ypredict =clff.predict(weighttest.tolist())
            Y=Y+ypredict.tolist()
            print('Predict',i*size_Predict,' th record-------------------and takes',time.clock())
        if batch_Predict * size_Predict<df_Predict.shape[0]:
            #the rest item
            indSeleted2 = range(batch_Predict * size_Predict, df_Predict.shape[0])
            df_T = df_Predict.loc[indSeleted2]
            xtestcor = __TFgetjiebaSeries(df_T['ABSTRACT'])
            weighttest = TFtesttfidf(xtestcor, vector, transformer)
            ypredict = clff.predict(weighttest.tolist())
            Y = Y + ypredict.tolist()
        Y=[str(i) for i in Y]
        df['predict']=df['predict']+' '+Y
        print('Total time',time.clock(),'This round takes',time.clock()-toc)
        toc=time.clock()
    return df
    pass

def writeExcel(df,url='',row_Max=500000):
    No=int(df.shape[0]/row_Max)
    for i in range(No):
        u=url+'file'+str(i)+'.xlsx'
        write = pd.ExcelWriter(u)
        df[i*row_Max:(i+1)*row_Max].to_excel(write, sheet_name='Sheet1',merge_cells=False)
        write.save()

    u = url + 'file' + str(No) + '.xlsx'
    write = pd.ExcelWriter(u)
    df[No * row_Max:df.shape[0]].to_excel(write, sheet_name='Sheet1', merge_cells=False)
    write.save()

def TrainByIPC(df_Base,size_Train,df_Predict,size_Predict,Cat,repeatNo):
    df_Base['l1_IPC']=df_Base['IPC'].str.slice(0,4)
    df_Predict['l1_IPC']=df_Predict['IPC'].str.slice(0,4)
    IPCs=df_Base['IPC'].str.slice(0,4).unique()
    df_T=pd.DataFrame()
    for i in IPCs:
        df_Base_temp=df_Base[df_Base['l1_IPC']==i]
        df_Predict_temp = df_Predict[df_Predict['l1_IPC'] == i]
        size_Train_temp=min(size_Train,df_Base_temp.shape[0])
        size_Predict_temp=min(size_Predict,df_Predict_temp.shape[0])
        if size_Predict_temp>0 and size_Train_temp>10:
            print('Now,Processing IP in', i,'with train size',df_Base_temp.shape[0],'and predict size',df_Predict_temp.shape[0])
            df_Temp=SplitTrainning(df_Base_temp,size_Train_temp,df_Predict_temp,size_Predict_temp,Cat,repeatNo)
            df_T=pd.concat([df_T,df_Temp],axis=0,ignore_index=False)

    return df_T

if __name__=="__main__":
    #1.inial
    docdir = os.path.abspath('..') + '\\doc\\'
    name=readFilesName(docdir+'mid\\')
    dest=docdir+'re2\\'
    df_Base = pd.DataFrame()
    df_Test = pd.DataFrame()

    #2.get the ready Data
    print('-------------Start Reading Data--------------')
    j=0
    for i in name:
        df=readExcel(i)
        df_Base,df_Test= combine_Abstract(df,df_Base,df_Test)
        j=j+1
        print(j/len(name),'Data Loaded')

    #3.train the data
    print('-------------Start Training Data--------------')
    df_Base,Cat=digitize_Cat(df_Base)

    df=TrainByIPC(df_Base,400,df_Test,500,Cat,1)
    # 上面有三个数字，第一个代表训练集大小，第二个代表预测集大小，第三个表示有多少个弱分类器
    #4.export result
    print(df)
    writeExcel(df, url=dest, row_Max=500000)#结果输出，其中数字代表一个文件里面有多少条记录，会拆分
    write = pd.ExcelWriter(dest+'Cat.xlsx')
    Cat.to_excel(write, sheet_name='Sheet1', merge_cells=False)
    write.save()
    pass