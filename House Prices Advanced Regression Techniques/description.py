
# coding: utf-8

# In[ ]:

"""
__file__

    description.py

__description__

    This file provides functions to describe dataframe,e.g.,df.columns,df_missing_columns

__author__

    zhang
    
"""
import pandas as pd
import numpy as np
import time
import os

save_path = '../feature_dicvide/"
if !os.path.isdir(“save_path”) :
    os.mkdir(“save_path”) 

    
class describe_df(object):
    number = 0
    def __init__(self, df):
        self.df = df
        
    def feature(self):
        df = self.df
        return list(df.columns)
    def ms_noms_cols(self, save=False):
        """
        @return description:
        mscols 返回含有缺失值的特征（list）
        nomscols 返回不含有缺失值的特征（list）
        missing_describe 返回每个特征缺失状况（dataframe）
        """
        df = self.df
        describe_df.number += 1
        types = df.dtypes
        
        total_missing = df.isnull().sum()
        percent_missdtypesing = df.isnull().sum() / df.isnull().count()
        missing_describe = pd.concat([types, total_missing, percent_missdtypesing], axis = 1, keys = ['Types', 'Total', 'Percent'])
        missing_describe.to_csv("missing_describe" + time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())) + ".csv")
        
        mscols = list(missing_describe[missing_describe['Total']>0].index)
        nomscols = list(missing_describe[missing_describe['Total'] == 0].index)
        
        if save:
            mscols_file = save_path + "mscol" + time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())) + ".txt"
            nomscols_file = save_path + "nomscol" + time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())) + ".txt"
        
            ms_string = ','join(mscols)
            noms_string = ','join(nomscols)
        
            with open(mscols_file,'a') as f1:
                f1.write(ms_string)
            
            with open(nomscols_file 'a')as f2:
                f2.write(noms_string)
            
        return mscols, nomscols, missing_describe
    
    def valtype_divide(self, save=False):
        df = self.df
        ob = list(df.columns[df.dtypes=='object'])
        noob = list(df.columns[df.dtypes!='object'])
        
        int64type = list(df.columns[df.dtypes=='int64'])
        int32type = list(df.columns[df.dtypes=='int32'])
        
        inttype = int64type + int32type
        
        float64type = list(df.columns[df.dtypes=='float64'])
        float32type = list(df.columns[df.dtypes=='float32'])
        
        floattype = float64type + float32type
        
        if save :
            ob_file = save_path + "object_type" + time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())) + ".txt"
            noob_file = save_path + "noobject_type" + time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())) + ".txt"
        
            ob_string = ','join(ob)
            noob_string = ','join(noob)
        
            with open(ob_file,'a') as f1:
                f1.write(ob_string)
            
            with open(noob_file 'a')as f2:
                f2.write(noob_string)
        
        
        return ob,noob,inttype,floattype
    
    def type_msnoms(self, save=False, osave=False):
        ob, noob, _1, _2 = valtype_divide(osave)
        mscols, nomscols, missing_describe = ms_noms_cols(osave)
        
        obms = list(set(ob) &  set(mscols))
        obnoms = list(set(ob) &  set(nomscols))
        
        noobms = list(set(noob) &  set(mscols))
        noobnoms = list(set(noob) &  set(nomscols))
        
        if save:
            ob_file = save_path + "object_type" + time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())) + ".txt"
            noob_file = save_path + "noobject_type" + time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())) + ".txt"
        
            ob_string = ','join(ob)
            noob_string = ','join(noob)
            
        
            with open(ob_file,'a') as f1:
                f1.write(ob_string)
            
            with open(noob_file 'a')as f2:
                f2.write(noob_string)

        
        return missing_describe, obms, obnoms, noobms, noobnoms
        
    
    @staticmethod
    def contain_describe(df1, df2, name1, name2):
        """
        @return description:
        contain_describe df1 和 df2 特征差异（dataframe）
        """
        col1=set(df1.columns)
        col2=set(df2.columns)
        col=list(col1|col2)
        df1_contain=[]
        df2_contain=[]
        for i in col:
            if i in col1:
                df1_contain.append(1)
            else:
                df1_contain.append(0)
            if i in col2:
                df2_contain.append(1)
            else:
                df2_contain.append(0)
        d = {'col' : col,
             name1 : df1_contain,
             name2 : df2_contain}
        contain_describe=pd.DataFrame(d)
        contain_describe.to_csv("contain_describe"+time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))+".csv")
        return contain_describe

