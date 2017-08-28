import os
import pandas as pd

def readExcel(url):
    df=pd.read_excel(url,na_values='')
    return df

def readFilesName(dir):
    name=[]
    for parent,dirnames,filenames in os.walk(dir):
        for filename in filenames:
            url=parent+filename
            name.append(url)
    return name

def writeExcel(df,url=''):
    write = pd.ExcelWriter(url)
    df.to_excel(write, sheet_name='Sheet1',merge_cells=False)
    write.save()

def combineDF(name):
    df=pd.DataFrame(None,columns=['Company', 'Cat', 'Cat.1'])
    for i in name:
        df_temp=readExcel(i)
        print('Reading------',i,'--------Finished')
        df=pd.concat([df,df_temp],axis=0)
    return df

def processDF(df,method='mean'):
    df_detail_orig = df.groupby(['Company', 'Cat'])['Cat.1'].sum()
    df_detail=df_detail_orig.reset_index()
    df_detail.columns=['Company', 'Cat', 'SubNo.']

    df_company_orig=df.groupby(['Company'])['Cat.1'].sum()
    df_company=df_company_orig.reset_index()
    df_company.columns = ['Company', 'TotalNo.']
    df = pd.merge(df_detail, df_company, 'left')
    df['Percentage']=df['SubNo.']/df['TotalNo.']
    return df


if __name__=="__main__":
    #1.inial
    docdir = os.path.abspath('..') + '\\doc\\'
    name=readFilesName(docdir+'re\\')

    #2. read and combine df
    df=combineDF(name)

    #3. process df
    print('Start Processing--------------------')
    df_re=processDF(df)
    print('exporting---------------------------')
    #4. export df
    writeExcel(df_re, docdir+'Fresult01.xlsx')