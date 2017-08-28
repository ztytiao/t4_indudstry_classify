import pandas as pd
import os

def readExcel(url):
    df=pd.read_excel(url,na_values='')
    return df


def calNOs(df_Cat,df_Ind):
    df_Cat['Y']=0
    for i in df_Ind.index:
        for j in df_Cat.index:
            if str(df_Ind.loc[i,'IPC']) in str(df_Cat.loc[j,'Unnamed: 2']):
                df_Cat.loc[j,'Y']=1
    return df_Cat
    pass

def sumNo(df):
    pass

if __name__=="__main__":
    docdir = os.path.abspath('..') + '\\doc\\'
    IndustryFileName = 'industry_sep.xlsx'
    CatFileName='IPC-COUNTV0821.xlsx'

    df_Cat=readExcel(docdir+CatFileName) #IPC总量 6902114  Unnamed: 2
    df_Ind=readExcel(docdir+IndustryFileName) #行业 合并后   IPC keywords keywords1  keyward2 notkeywords

    pass