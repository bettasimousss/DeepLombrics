# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:35:49 2020

@author: saras

This script contains specific loss functions and metrics for some supported compositional architectures
(See lego_blocks.py)
"""


import pandas as pd
import numpy as np


date="0509"

### Input file ###
file_taxonomy="application/data/taxo/taxo_drilobase.csv"
file_groups="application/data/taxo/species_taxa_id.csv"
file_retained="application/data/taxo/retained_taxa.csv"
file_database='application/data/Vers2022/database.csv'
file_grid='application/data/current/grid.csv'
file_env='application/data/current/current_env_prep.csv'

### Output files
file_prep='application/vers22_out/bdd_prep_%s.csv'%date


'''
Preprocessing earthworm database 
1- Normalize ids to match nomenclature used in the model
2- Assign grid cell to coordinate for quick evaluation from projections directly
3- Import environmental data
'''

#######################
'''
Name normalization
'''
#######################
full_taxonomy=pd.read_csv(file_taxonomy,sep=',',decimal='.')
old_new=pd.read_csv(file_groups,sep=',',decimal='.',index_col=0).set_index('taxa_id')
subids=old_new['sp_id'].unique().tolist()#pd.read_csv(file_retained,sep=',',decimal='.').query('supertaxa==0')['gid'].tolist()

def name2id(x,y):
    df=full_taxonomy.query('scientificNameBouche==@x')
    if len(df)>0:
        return df.index[0]
    else:
        df=full_taxonomy.query('scientificNameDrilobase==@y')
        if len(df)>0:
            return df.index[0]
    
    return np.nan

### Asign ids to current observations used for evaluation
db=pd.read_csv(file_database,sep=',',decimal='.')
db['dbid']=db.index.tolist()
Yc=db.query('scNameBouche==scNameBouche')

occur_ids=[name2id(x, y) for x,y in Yc[['scNameBouche','scNameDrilobase']].values.tolist()]
Yc['tid']=occur_ids

Yc['sp_id']=Yc['tid'].apply(lambda x: old_new.loc[full_taxonomy.loc[int(x),'Id_taxon'],'sp_id'] if x==x else -1)
Yc['taxa_id']=Yc['tid'].apply(lambda x: full_taxonomy.loc[int(x),'Id_taxon'] if x==x else -1)

#Yc.to_csv('application/vers22_out/prefinal_db.csv',sep=',',decimal='.')

## Correct mismatches 

#unmatched=Yc.query('Id_taxon!=Id_taxon')

#Y_eval=Yc.query('Id_taxon==Id_taxon')#[['Longitude_WGS84','Latitude_WGS84','Id_taxon']]
#Y_eval['Id_gtaxa']=[old_new.loc[int(x),'new_id'] if int(x) in old_new.index else -1 for x in Y_eval['Id_taxon']]

Y_eval=Yc.query('sp_id!=-1 & Longitude_WGS84==Longitude_WGS84 & Latitude_WGS84==Latitude_WGS84 & (Longitude_WGS84!=0 | Latitude_WGS84!=0)')

#######################
'''
Coordinates assignement
'''
#######################
grid=pd.read_csv(file_grid,sep=',',decimal='.',index_col=0).astype(float)
minx, miny=grid[['X','Y']].min(axis=0)
maxx,maxy=grid[['X','Y']].max(axis=0)

def lonlat2pixel(lon,lat):
    if (minx<=lon<=maxx) &(miny<=lat<=maxy):
        gid=0
        ## Get closest X, Y
        dists=(grid-(lon,lat)).abs().sum(axis=1)
        gid=dists.argmin()
    
    elif (lon==lat==0):
        gid=-1
    else:
        print(lon, lat)
        Warning('Out of geographic extent')
        gid=-1
    
    return gid

coords=Y_eval[['Longitude_WGS84','Latitude_WGS84']].astype(float).values.tolist()
nngrid=[lonlat2pixel(x,y) for x,y in coords]
Y_eval['nngrid']=nngrid


#######################
'''
Environmental data assignement
'''
#######################
env_curr= pd.read_csv(file_env,sep=",",decimal=".",index_col=0) 
db_ev=Y_eval.query('nngrid!=-1').join(env_curr,on='nngrid') 
#X_ev=env_curr.iloc[Y_ev['nngrid'].values.tolist(),:]

### Updating land cover with provided values 
'''
1. Encoding provided values 
2. Keeping map values where not provided
'''

clc_classes=[11,12,13,14,21,22,23,24,31,32,33,41,42,51,52,99]
clc_codes=np.arange(len(clc_classes))

clc_map={clc_classes[i]:clc_codes[i] for i in range(len(clc_classes))}

def clc_encoder(obs,ext):
    if obs!=obs: ##=np.nan
        code=int(ext)
    else: ### encode
        code=clc_map.get(int(obs))
    return code

db_ev['landcov']=[clc_encoder(a,b) for a,b in db_ev[['CLC','clc']].values.tolist()]
db_ev.to_csv(file_prep,sep=',',decimal='.')

print("Reminder to update land cover name from landcov to clc to match training data on the saved file")