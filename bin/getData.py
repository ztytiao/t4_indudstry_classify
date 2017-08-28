import os
import pandas as pd
import copy
import jieba
import numpy as np
import time


def readExcel(url):
    df=pd.read_excel(url,na_values='')
    return df

def writeExcel(df,url=''):
    write = pd.ExcelWriter(url)
    df.to_excel(write, sheet_name='Sheet1')
    write.save()

def genDF(df_Base,df_IPC):
    columns=df_Base['合并后'].unique()
    index=df_IPC['NAME'].unique()
    return pd.DataFrame(None,index=index,columns=columns)
    pass

def check(df_Input,df_Base,df_result=[]):
    ''''''
    if df_result==[]:
        df_result=pd.DataFrame(None,columns=['Company','Cat'])
    for i in df_Input.index:
        company=df_Input.loc[i,'APPLICANT']
        ipc=df_Input.loc[i,'IPC']
        summary=df_Input.loc[i,'ABSTRACT']
        cor=__jieba(summary)
        df=__checkOne(company,df_Base,ipc,cor)
        df_result=pd.concat([df_result,df],axis=0)
    return df_result
    pass

def __checkOne(company,df_Base,ipc,cor):
    df=pd.DataFrame(None,columns=['Company','Cat'])
    for i in df_Base.index:
        ipc_cat=df_Base.loc[i,'IPC']
        key_cat=df_Base.loc[i,'keywords']
        key_cat1=df_Base.loc[i,'keywords1']
        #key_cat2=df_Base.loc[i,'keyward2']
        keyNo_cat=df_Base.loc[i,'notkeywords']
        if (ipc==str(ipc_cat) or ipc.split(sep='/')[0]==str(ipc_cat))\
                and (type(key_cat)==type(1.1) or key_cat in cor)\
                and (type(key_cat1)==type(1.1) or key_cat1 in cor)\
                and (type(keyNo_cat)==type(1.1) or keyNo_cat not in cor):
            df=df.append({'Company':company,'Cat':df_Base.loc[i,'合并后']},ignore_index=True)
    #and
    return df

def __jieba(summary):
    '''根据abstract，进行分词，返回矩阵'''
    b=jieba.cut(summary)
    # re=[]
    # for i in b:
    #     re.append(i)
    return b
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

if __name__=="__main__":
    #1.inial
    docdir = os.path.abspath('..') + '\\doc\\'
    IndustryFileName='industry_sep.xlsx'
    IPCFileName='ipctest1.xlsx'
    ABSFileName=''
    IPCABSFileName='ipc_abs_test.xlsx'

    #df_IPC=readExcel(docdir+IPCFileName)
    #2.read file

    df_Input=readExcel(docdir+IPCABSFileName)
    df_Base=readExcel(docdir+IndustryFileName)
    addDict(df_Base)

    #3.run
    start=time.clock()
    w=check(df_Input,df_Base,df_result=[])
    end=time.clock()
    dur=end-start

    #4.export excel
    writeExcel(w,docdir+'result01.xlsx')
    print(dur)


