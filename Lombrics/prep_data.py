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

from deep_mvp import *
from eval_functions import *
import tensorflow_addons as tfa

##########Y Reproducibility #############
rs=0
np.random.seed(rs)


##### Disable script portions ######
prep_phylo=False
prep_funct=False
test_conserv=False

use_spat=False

########################################################################################################
'''
                                        Data paths 
'''
########################################################################################################
folder_data="application/data/"
folder_taxo="application/data/taxo/"
folder_phylo="application/data/phylogeny/"
folder_traits="application/data/traits/"
folder_model="application/train/model/"

folder_log="application/train/log/"

### Geographic data
file_coord=folder_data+"past/coords_clean.csv"

### Observations data
file_obs_bouche=folder_data+"past/lien_taxa_station.csv"
file_eval="application/vers22_out/bdd_prep.csv"
file_eval_sp="application/vers22_out/bdd_splines.csv"

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
        match.to_csv('application/data/past/species_taxa_id.csv',sep=",",decimal='.')
        sp_df.to_csv('application/data/past/species_count.csv',sep=",",decimal='.')  
    
    return sp_df, match, splist, taxa_names

sp_df, match, splist, taxa_names = generate_occur()
richness=(sp_df>0).sum(axis=1)
prevalence=(sp_df>0).sum(axis=0)

########################################################################################################
'''
                                    Generate cross-validation datasets

## generate and save indices                                
'''      
######################################################################################################## 
mincount=5

ret_taxa=np.where(prevalence>=mincount)[0].tolist()
names=[taxa_names[j] for j in ret_taxa]

occur=(sp_df>0).astype(int).iloc[:,ret_taxa]
sel_stations=np.sort(np.array(stations)[np.where(occur.sum(axis=1)>=1)[0].tolist()]).tolist()

Y=occur.loc[sel_stations,ret_taxa]

curr=eval_db.query('sp_id in @ret_taxa')
sel_eval=curr.index.tolist()

subid_col={s:i for i,s in enumerate(ret_taxa)}
curr['tnum']=curr['sp_id'].apply(lambda x: subid_col.get(x))

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
                                    Generate architectures
''' 
########################################################################################################
 
ep=200  
opt='adamax'
bs=32
ns=100
gpus=1
gamma=2

nbmods={'crusting':6,'erodi':6,'wr':5,'parmado':9,'clc':16}    




########################################################################################################
'''
                                    Preprocess phylogenetic data
'''      
######################################################################################################## 


if prep_phylo:
    taxa_codes={tn:i for i, tn in enumerate(taxa_names)}
    phylo_codes=pd.read_csv(file_phylocodes,sep=",",encoding='latin-1')
    phylo_codes['drilo_code']=[taxa_codes[x] for x in phylo_codes['scientificNameDrilobase']]
    
    # #####     
    # ## Read phylo tree
    # tree= Phylo.read(file_phylotree, 'newick')
    
    # fig, ax=plt.subplots(1,1,figsize=(10,15))
    # Phylo.draw(tree,axes=ax)
    
    ## Load distance matrix
    phylodist= pd.read_csv(file_phylodist,sep=',',decimal='.',index_col=0)
    
    ## Imputate phylogenetic information for all considered taxa
    ## Use closest relative within genera => approved by experts
    taxo_data=taxo[['Id_taxon', 'scientificNameDrilobase','drilo_code','categ_eco','Family','Genus','Species','Sub_species','Variety','Level']].drop_duplicates()
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
    all_dist=nndf.merge(nndf.merge(pdist_long,right_on='X',left_on='code_nn'),right_on='Y',left_on='code_nn',suffixes=('_y','_x'))
    
    phyl_mat=all_dist[['sp_id_x','code_t_x','sp_id_y','code_t_y','dist']].groupby(['sp_id_x','code_t_x','sp_id_y','code_t_y']).agg('mean').reset_index()        
    
    #len(out_mat['sp_id_x'].unique().astype(int))
    #pb=[t for t in range(len(taxa_codes)) if t not in set(out_mat['sp_id_x'].astype(int))]


########################################################################################################
'''
                                    Preprocess traits data
'''      
######################################################################################################## 
    
if prep_funct:
    ### Ecological categories
    ecocat_sp={}
    for s,sn in zip(sp_df.columns.tolist(),splist):
        ecocat_sp[s]=taxo.query('drilo_code==@s')['categ_eco'].unique().tolist()
    
    ### Other traits
    traits=pd.read_csv(file_traits,sep=",",index_col=0)
    traits['drilo_code']=traits['Id_taxon'].apply(lambda x: int(match.query('taxa_id==@x')['sp_id']))
    

if test_conserv:
    sel1h='climall_raw_onehot_shared16_nn1_8_wd_True_wnll_0.001_0.001'
    sel0='selected_archi'
    sel='climall_raw_embed_shared16_nn1_8_wd_True_wnll_0.005_0.010'
    import h5py
    f = h5py.File('application/train/model/%s.h5'%sel,'r')
    sw=f.get('mu_'+sel)
    
    noise=f.get('noise_chol_11').get('noise_chol_11')
    chol_prec=noise.get(sel+'_noisechol_uomega:0')[:]+np.diag(np.exp(noise.get(sel+'_noisechol_domega:0')[:]))
    prec=np.dot(chol_prec.T,chol_prec)
    
    pcor=cov2corr(prec)
    sel_weights=[np.array(sw.get(k).get('kernel:0')[:]) for k in sw.keys()]
    
    off=1
    nl=2
    flat_weights=[]
    for i in range(m):
        fw=[]
        for j in range(nl):
            fw+=sel_weights[off+i*nl+j].flatten().tolist()
        
        flat_weights.append(np.array(fw))
        
    flat_weights=np.stack(flat_weights,axis=1)
    
    
    
    from sklearn.metrics.pairwise import cosine_distances
    import gower
    
    resp_diss=pd.DataFrame(cosine_distances(flat_weights.T),columns=ret_taxa,index=ret_taxa)
    resp_diss.to_csv('resp_diss.csv',sep=',',decimal='.')
    
    ## get phylo distance between modeled taxa
    def get_phylo_dist(a,b):
        ## get phylo codes
        code_a=pcodes[a]
        code_b=pcodes[b]
        
        d=out_mat.query('code_t_x==@code_a & code_t_y==@code_b')['dist'].values
        
        return -1 if len(d)==0 else float(d[0])
    
    
    def get_fct_dist(a,b):
        idx=traits.index.tolist()
        idt_a=[x for x in taxo_data.query('drilo_code==@a')['Id_taxon'] if x in idx]
        idt_b=[x for x in taxo_data.query('drilo_code==@b')['Id_taxon'] if x in idx]
        
        if (len(idt_a)==0) | (len(idt_b)==0):
            return -1
        else:
            print(a , b)
            tra=traits.loc[idt_a].values
            trb=traits.loc[idt_b].values
            d=cosine_distances(tra,trb)
            
            return d.mean()
    
    ret_taxa_fct=[(a,b,get_fct_dist(a,b)) for a,b in itertools.combinations(ret_taxa,2) ]
    ret_fdist=pd.DataFrame(np.array(ret_taxa_fct),columns=['x','y','fct_dist']).query('fct_dist!=-1') 
    ret_fdist['resp_diss']=[resp_diss.loc[x,y] for x,y in ret_fdist[['x','y']].values.tolist()] 
    ret_fdist[['resp_diss','fct_dist']].corr()
    
    sns.scatterplot(data=ret_fdist,x='fct_dist',y='resp_diss') 
    
    ret_taxa_phylo=[(a,b,get_phylo_dist(a,b)) for a,b in itertools.combinations(ret_taxa,2) ]
    ret_pdist=pd.DataFrame(np.array(ret_taxa_phylo),columns=['x','y','phylo_dist']).query('phylo_dist!=-1') 
    ret_pdist['resp_diss']=[resp_diss.loc[x,y] for x,y in ret_pdist[['x','y']].values.tolist()] 
    
    ret_pdist[['resp_diss','phylo_dist']].corr()
    
    sns.scatterplot(data=ret_pdist,x='phylo_dist',y='resp_diss')     