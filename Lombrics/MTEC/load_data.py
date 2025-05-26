# -*- coding: utf-8 -*-
"""
Created on Tue May 19 13:54:55 2020

@author: saras
"""

import pandas as pd
import numpy as np

import json

import itertools

from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from mca import MCA

from skmultilearn.model_selection import IterativeStratification
from skmultilearn.model_selection.iterative_stratification import iterative_train_test_split

from Bio import Phylo

from scipy.special import logit
from statsmodels.stats.moment_helpers import cov2corr

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow_addons as tfa

from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

#from deep_mvp import *
import sys
sys.path.append('../')
from mtlvae import *
from eval_functions import *


##########Y Reproducibility #############
rs=0
np.random.seed(rs)


##### Disable script portions ######
use_spat=False
prep_phylo=True
prep_fct=True

########################################################################################################
'''
                                        Data paths 
'''
########################################################################################################
folder_data="data/"
folder_taxo="data/taxo/"
folder_phylo="data/phylogeny/"
folder_traits="data/traits/"
folder_model="train/model/"

folder_log="train/log/"

### Geographic data
file_coord=folder_data+"past/coords_clean.csv"

### Observations data
file_obs_bouche=folder_data+"past/lien_taxa_station.csv"
file_eval="vers22_out/bdd_prep.csv"
file_eval_sp="vers22_out/bdd_splines.csv"

### Taxonomic, phylogenetic and functional databases
file_taxo=folder_taxo+"taxo_drilobase.csv"
file_traits=folder_traits+"prep_traits.csv"
file_phylocodes=folder_phylo+"taxa_codes.csv"
file_phylotree=folder_phylo+"tree.newick"
file_phylodist=folder_phylo+"phylo_distance.csv"

### Environmental data
file_penv_raw=folder_data+"past/env_raw.csv"
file_penv_prep=folder_data+"past/env_prep.csv"
file_splines=folder_data+"past/spatial_splines.csv"

########################################################################################################
'''
                Load evaluation data
'''      
########################################################################################################  
eval_db=pd.read_csv(file_eval,sep=',',decimal='.',index_col=0)
eval_splines=pd.read_csv(file_eval_sp,sep=',',decimal='.',index_col=0)

########################################################################################################
'''
                            Environmental data preparation                            
'''
########################################################################################################
env_raw=pd.read_csv(file_penv_raw,sep=',',decimal='.',index_col=0)
env_df=pd.read_csv(file_penv_prep,sep=',',decimal='.',index_col=0)
spat_df=pd.read_csv(file_splines,sep=',',decimal='.',index_col=0)

###### Select environmental variables ############

def generate_datasets():
    env_dataset=[]
    clim_choice=['climall','climvif']
    
    clim_all=['bio_%d'%i for i in range(1,20)]
    clim_vif=['bio_1','bio_3','bio_7','bio_8','bio_9','bio_15','bio_19']
    
    
    ###### Categories #########
    cat_names={
            'crusting':(['NS','VW','W','M','S','VS'],
                        [0,1,2,3,4,5]),
            
            'erodi':(['NS','VW','W','M','S','VS'],
                     [0,1,2,3,4,5]),
            
            'wr':(['NA','D3D1','W3-6D1','W6D11','W11'],
                  [0,1,2,3,4]),
            
            'parmado':(['clastic-sedim','sediment','igneous','metamorpho','alluvWeathSlope','glacial','eolian','organic','anthropo'],
                       [0,1,2,3,4,5,6,7,8]),
            
            'clc':(['Urban','Transport','MineDump','ArtifVeg',
                    'Arable','PermaCrop','Pasture','HeterAgri',
                    'Forest','ScrubHerbVeg','OpenSpacNVeg',
                    'InlandWet','CoastalWet',
                    'ContinWater','MarineWater',
                    'NA'],
                   [11,12,13,14,21,22,23,24,31,32,33,41,42,51,52,99])
            }
    
    for i, clim in enumerate([clim_all,clim_vif]):
        num=clim+['pH','Carbone','CN','clay','silt']
    
        cat=['crusting','erodi','wr','parmado','clc']
        ordi=['awc_top', 'bs_top', 'cec_top', 'dgh', 'dimp','pd_top']
        bina=['proxi_eau_fast']
        
        num_vars=num+bina+ordi
        cat_vars=cat
        
        for numprep,catprep in [('raw','onehot'),
                     ('raw','embed'),
                     ('decomp','decomp'),
                     ('decomp','onehot'),
                     ('decomp','embed')]:
            
            config={'varsel':clim_choice[i],'num_vars':num_vars,'cat_vars':cat_vars}
    
            ###### Preprocessing vs end-to-end ############
            config['prep']=(numprep,catprep)
            config['num_pvars']=[]
            config['cat_pvars']=[]
            
            if numprep=='raw':
                X_num=env_df[num_vars]
                Xc_num=eval_db[num_vars]
                config['num_pvars']+=num_vars
                
            if numprep=='decomp':
                ### Preprocess with PCA + MCA
                exp_var=0.95
                pca=PCA(n_components=exp_var,svd_solver='full',random_state=rs)
                pca.fit(env_df[num_vars])
                X_num=pd.DataFrame(data=pca.transform(env_df[num_vars]),columns=['pca_%i' for i in range(pca.n_components_)],index=env_df.index)
                Xc_num=pd.DataFrame(data=pca.transform(eval_db[num_vars]),columns=['pca_%i' for i in range(pca.n_components_)],index=eval_db.index)
                
                config['num_pvars']+=['pca_%d'%i for i in range(pca.n_components_)]
            
            if catprep=='decomp':
                mca=MCA(DF=env_df[cat_vars].astype(int).astype(str),cols=cat_vars,ncols=len(cat_vars))
                X_cat=pd.DataFrame(data=mca.fs_r(N=mca.rank),columns=['mca_%i' for i in range(mca.rank)],index=env_df.index)
                
                mca=MCA(DF=eval_db[cat_vars].astype(int).astype(str),cols=cat_vars,ncols=len(cat_vars))
                Xc_cat=pd.DataFrame(data=mca.fs_r(N=mca.rank),columns=['mca_%i' for i in range(mca.rank)],index=eval_db.index)
                
                config['num_pvars']+=['mca_%d'%i for i in range(mca.rank)]
                
            if catprep=='onehot':
                all_cols=[c+'_%d'%k for c in cat_vars for k,_ in enumerate(cat_names[c][1])]  
                X_cat=pd.get_dummies(env_df[cat_vars].astype(int),columns=cat_vars).reindex(columns=all_cols).fillna(0)
                Xc_cat=pd.get_dummies(eval_db[cat_vars].astype(int),columns=cat_vars).reindex(columns=all_cols).fillna(0)
                config['num_pvars']+=all_cols
            
            if catprep=='embed':
                X_cat=env_df[cat_vars]
                Xc_cat=eval_db[cat_vars]
                config['cat_pvars']=cat_vars
                
            
            config['data']=pd.DataFrame(data=np.concatenate([X_num,X_cat],axis=1),columns=config['num_pvars']+config['cat_pvars'],index=env_df.index)
            config['cdata']=pd.DataFrame(data=np.concatenate([Xc_num,Xc_cat],axis=1),columns=config['num_pvars']+config['cat_pvars'],index=eval_db.index)
    
                
            env_dataset.append(config)

    return env_dataset            



env_dataset=generate_datasets()

########################################################################################################
'''
                Prepare occurrence data using drilobase taxo
'''      
########################################################################################################        
coord=pd.read_csv(file_coord,sep=";")
stations=env_df.index.tolist()
occur=pd.read_csv(file_obs_bouche,sep=";").query('Id_station in @stations')
taxo=pd.read_csv(file_taxo,sep=",")

########################################################################################################

def generate_occur(save=False):
    taxa_names=np.sort(list(set(taxo['scientificNameDrilobase']))).tolist()
    
    taxa_codes={tn:i for i, tn in enumerate(taxa_names)}
    taxo['drilo_code']=[taxa_codes[x] for x in taxo['scientificNameDrilobase'].tolist()]
    
    ### Generate species-level datasets ###
    data=occur.merge(taxo,left_on='Id_taxon',right_on='Id_taxon')[
        ['Id_station', 'Id_taxon', 'Total','scientificNameDrilobase','drilo_code']]
    
    
    
    ### Generate species-level occurrence dataset
    sp_data=data.groupby(
                     ['Id_station','scientificNameDrilobase','drilo_code']).agg(func={'Total':['sum'],'Id_taxon':[lambda x: set(x)]}).reset_index()
    
    sp_data.columns=['Id_station','tname','tcode','count','Id_taxa']                
    
    sp_df=sp_data[['Id_station','tcode','count']].pivot_table('count', aggfunc=np.sum, columns='tcode', index='Id_station',fill_value=0)                 
    splist=[taxa_names[i] for i in sp_df.columns.tolist()]   
    
    ### old_new id correspondence file
    oldnew=[]
    for j, s in enumerate(splist):
        for t in taxo.query('scientificNameDrilobase==@s')['Id_taxon'].tolist():
            oldnew.append((j,t))
            
    match=pd.DataFrame(data=np.array(oldnew),columns=['sp_id','taxa_id'])
    
    if save:
        match.to_csv('data/past/species_taxa_id.csv',sep=",",decimal='.')
        sp_df.to_csv('data/past/species_count.csv',sep=",",decimal='.')  
    
    return sp_df, match, splist, taxa_names

sp_df, match, splist, taxa_names = generate_occur()
richness=(sp_df>0).sum(axis=1)
prevalence=(sp_df>0).sum(axis=0)


eval_db['sp_id']=eval_db['taxa_id'].apply(lambda x: int(match.query('taxa_id==@x')['sp_id'])).tolist()

########################################################################################################
'''
                                    Generate cross-validation datasets

## generate and save indices                                
'''      
######################################################################################################## 
# rare_curv=[]
# for mincount in np.arange(1,20,1):
#     ret_taxa=np.where(prevalence>=mincount)[0].tolist()
#     names=[taxa_names[j] for j in ret_taxa]

#     occur=(sp_df>0).astype(int).iloc[:,ret_taxa]
#     sel_stations=np.sort(np.array(stations)[np.where(occur.sum(axis=1)>=1)[0].tolist()]).tolist()    
#     rare_curv.append((mincount,len(ret_taxa),len(sel_stations)))

# data=pd.DataFrame(np.array(rare_curv),columns=['threshold','poolsize','sites'])
# fig, ax=plt.subplots()
# sns.lineplot(ax=ax,data=data,x='threshold',y='poolsize')    
# ax2=ax.twinx()
# sns.lineplot(ax=ax,data=data,x='threshold',y='sites')
# fig.set(title='Rarity curve')

#plt.show()

mincount=5

ret_taxa=np.where(prevalence>=mincount)[0].tolist()
names=[taxa_names[j] for j in ret_taxa]

occur=(sp_df>0).astype(int).iloc[:,ret_taxa]
sel_stations=np.sort(np.array(stations)[np.where(occur.sum(axis=1)>=1)[0].tolist()]).tolist()

Y=occur.loc[sel_stations,ret_taxa]

dropsource=['Bouché (1972)','Canal & Rigole (1978)']
curr=eval_db.query('sp_id in @ret_taxa & source not in @dropsource')
sel_eval=curr.index.tolist()

subid_col={s:i for i,s in enumerate(ret_taxa)}
curr['tnum']=curr['sp_id'].apply(lambda x: subid_col.get(x)).tolist()

########################################################################################################  
def ml_cv(Y,minpos=5,rs=0,p=0.8,adjust=0):
    n,m=Y.shape
    np.random.seed(rs)
    
    size_train=int(n*p)
    size_test=n-size_train
    req=(np.ones(m)*minpos).astype(int)
    
    train=[]
    test=np.arange(n).tolist()
    unsatisf=np.arange(m).tolist()
    
    while(len(unsatisf)>0):
        y=Y[test,:][:,unsatisf]
        ### Compute available potential
        pot=y.sum(axis=0)
        min_pot=np.min(pot)
        cand=np.where(pot==min_pot)[0]
        
        choice=np.random.choice(a=cand,size=1)[0]
        rel_idx=np.random.choice(a=np.where(y[:,choice]==1)[0],size=int(req[unsatisf][choice]),replace=False).tolist()
        new_idx=[test[i] for i in rel_idx]
        
        for x in new_idx:
            train.append(x)
            test.remove(x)
        
        ### Update requirements ##
        prevs=Y[train,:].sum(axis=0)
        req=(np.ones(m)*minpos).astype(int)-prevs
        unsatisf=np.where(req>0)[0]
        
    diff=size_train-len(train)
    if diff>0:
        compl=np.random.choice(a=test,size=diff,replace=False)
        for x in compl:
            train.append(x)
            test.remove(x)
    
    if adjust>0:
        prevs_test=Y[test,:].sum(axis=0)
        correct=np.where(prevs_test<minpos)[0]
        for c in correct:
            tdiff=minpos-Y[test,c].sum(axis=0)
            
            if tdiff>0:
                new_idx=np.random.choice(a=np.where(Y[train,c]>0)[0],size=tdiff,replace=False)
                test+=new_idx.tolist()
    
    return (train,test)
       
nrep=5
mlcv=[ml_cv(Y.values,5,rs,0.5,5) for rs in range(nrep)]    

########################################################################################################
'''
                            Prepare functional group data
'''      
########################################################################################################        
if prep_fct:
    fct_groups=taxo[['drilo_code','p_epigeic','p_endogeic','p_anecic']].groupby('drilo_code').agg(np.mean)
    fct_dist=cosine_similarity(fct_groups)

########################################################################################################
'''
                            Prepare phylogenetic
'''      
########################################################################################################        
if prep_phylo:
    taxa_codes={tn:i for i, tn in enumerate(taxa_names)}
    taxa_codes_r={i:tn for i, tn in enumerate(taxa_names)}
    phylo_codes=pd.read_csv(file_phylocodes,sep=",",encoding='latin-1')
    phylo_codes['drilo_code']=[taxa_codes[x] for x in phylo_codes['scientificNameDrilobase']]
    
    # #####     
    
    ## Load distance matrix
    phylodist= pd.read_csv(file_phylodist,sep=',',decimal='.',index_col=0)
    
    ## Imputate phylogenetic information for all considered taxa
    ## Use closest relative within genera => approved by experts
    taxo_data=taxo[['Id_taxon', 'scientificNameDrilobase','drilo_code',
                    'p_epigeic','p_endogeic','p_anecic',
                    'Family','Genus','Species','Sub_species','Variety','Level']].drop_duplicates()
    
    phylo_info=[int(x in phylo_codes['drilo_code'].tolist()) for x in taxa_codes.values()]
    taxo_data['has_phylo']=taxo_data['drilo_code'].apply(lambda x: phylo_info[x])
    
    pcodes=[]
    for x in range(len(taxa_codes)):
        code=phylo_codes.query('drilo_code==@x')['Code']
        if len(code)>0:
            pcodes.append(code.values[0])
        else:
            genus=taxo_data.query('drilo_code==@x')['Genus'].values[0]
            species=taxo_data.query('drilo_code==@x')['Species'].values[0]
            subspecies=taxo_data.query('drilo_code==@x')['Sub_species'].values[0]
            variety=taxo_data.query('drilo_code==@x')['Variety'].values[0]
            
            res=genus[0:2].title()+species[0:4].title()
            if subspecies==subspecies:
                res+=('_'+subspecies[0:2].title())
            if variety==variety:
                varisplit=variety.split()
                for k in varisplit:
                    res+=k[0:2].title()
                
            pcodes.append(res)
    
    map_pcodes={c.strip():i for i,c in enumerate(pcodes)}
    map_pcodes_r={i:c.strip() for i,c in enumerate(pcodes)}
    
    
    codes_df=pd.concat([pd.Series(map_pcodes_r),pd.Series(taxa_codes_r)],axis=1).reset_index()
    codes_df.columns=['drilo_code','short_name','full_name']     
       
    #taxo_data['Code']=pcodes
    
    levels=['Variety','Sub_species','Species','Genus','Family']
    num_level={l:i for i,l in enumerate(levels)}
    
    def get_sibling(t,maxlev=1):
        tpath=taxo_data.query('drilo_code==@t')
        if tpath['has_phylo'].values[0]==1:
            res=(-1,{t})
        else:
            res=(maxlev,[])
            lev=num_level.get(tpath['Level'].values[0])
                
            while(lev<maxlev):
                spec=tpath[levels[lev+1]].values[0]
                siblings=taxo_data.query(levels[lev+1]+'==@spec & has_phylo==1')
                if len(siblings)>0:
                    res=(lev,set(siblings['drilo_code'].tolist()))
                    lev=maxlev
                
                else:
                    lev+=1
        
        return taxa_names[t], res[0], res[1] 
    
    phyl_nn=[]
    check=[]
    for t in range(len(taxa_codes)):
        sib=get_sibling(t,3)
        if len(sib[2])==0:
            check.append(t)
        phyl_nn+=[(sib[0],t,int(s),sib[1],pcodes[t],pcodes[s]) for s in sib[2]]
        
    phyl_nn_df=pd.DataFrame(data=np.array(phyl_nn),columns=['name','sp_id','nn','level_up','code_t','code_nn'])
    phyl_nn_df['nn']=phyl_nn_df['nn'].astype(np.int64)
    
    
    ## Assign mean distance of nearest neighbors for each taxa
    pdist_long=phylodist.melt()
    pdist_long.columns=['X','dist']
    pdist_long['Y']=phylodist.columns.tolist()*phylodist.shape[0]
    
    nndf=phyl_nn_df[['sp_id','nn','code_t','code_nn']]
    
    nndf['code_t']=nndf['code_t'].astype(str).str.strip()
    nndf['code_nn']=nndf['code_nn'].astype(str).str.strip()
    nndf['sp_id']=nndf['sp_id'].astype(int)
    nndf['nn']=nndf['nn'].astype(int)
    
    all_dist=nndf.merge(nndf.merge(pdist_long,right_on='X',left_on='code_nn'),right_on='Y',left_on='code_nn',suffixes=('_y','_x'))
    
    phyl_mat=all_dist[['sp_id_x','code_t_x','sp_id_y','code_t_y','dist']].groupby(['sp_id_x','code_t_x','sp_id_y','code_t_y']).agg('mean').reset_index()        
    