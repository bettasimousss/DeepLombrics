# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 20:22:41 2020

@author: saras
"""


from load_data import *
import sklearn.preprocessing as skprep
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt


plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "figure.titleweight" : 'bold',
    "pgf.preamble": [
         r"\usepackage[utf8x]{inputenc}",
         r"\usepackage[T1]{fontenc}",
         r"\usepackage{cmbright}",
         ]
})


################## ######### ######### ######### ######### ######### ######### ######### ######### #########  
'''
                                            Feature description
'''
######### ########## ######### ######### ######### ######## ######### ######### ######### ######### ######### 
clim=['bio_%d'%i for i in range(1,20)]
num=clim+['pH','Carbone','CN','clay','silt']

cat=['crusting','erodi','wr','parmado','clc']
ordi=['awc_top', 'bs_top', 'cec_top', 'dgh', 'dimp','pd_top']
bina=['proxi_eau_fast']

num_vars=num+bina+ordi
cat_vars=cat



'''
1. We group features per thematic
'''
feature_groups={
    'structural':['crusting','erodi','silt','clay','dgh','dimp','parmado','pd_top'],
    'hydro':['awc_top','bs_top','cec_top','wr','proxi_eau_fast'],
    'temperature':clim[0:11],
    'precipitation':clim[11:19],
    'landcover':['clc'],
    'physchem':['pH','Carbone','CN']
}

vargroups=pd.DataFrame(np.concatenate(
    [np.array([feature_groups.get(k),[k]*len(feature_groups.get(k))]) for k in feature_groups.keys()],
    axis=1).T,columns=['feature','group']).set_index('feature')
    

colors={'temperature':'red',
        'precipitation':'blue',
        'hydro':'purple',
        'physchem':'orange',
        'landcover':'brown',
        'structural':'black'}

'''
2. We define evaluation functions for 
   * each metric    
   * overall vs taskwise 
   
3. We also define lower limits for each metric
'''

lwlimit={'balanced_accuracy':('Balanced accuracy',0.8),
         'auc':('Area Under the Curve',0.6),
         'precision':('Precision',0.5),
         'recall':('Recall',0.5),
         'f1':('F1-score',0.5),
         'tss':('True Skill Statistic',0.3)}


######### ########## ######### ######### ######### ######## ######### ######### ######### ######### ######### 
'''
WR: Dominant annual average soil water regime class of the soil profile of the STU
----------------
0   No information
1   Not wet within 80 cm for over 3 months, nor wet
    within 40 cm for over 1 month
2   Wet within 80 cm for 3 to 6 months, but not wet
    within 40 cm for over 1 month
3   Wet within 80 cm for over 6 months, but not wet
    within 40 cm for over 11 months
4   Wet within 40 cm depth for over 11 months


PAR-MAT-DOM1: Major group code for the dominant parent material of the STU
----------------
0   No information
1   consolidated-clastic-sedimentary rocks
2   sedimentary rocks (chemically precipitated,
    evaporated, or organogenic or biogenic in origin)
3   igneous rocks
4   metamorphic rocks
5   unconsolidated deposits (alluvium, weathering
    residuum and slope deposits)
6   unconsolidated glacial deposits/glacial drift
7   eolian deposits
8   organic materials
9   anthropogenic deposits
'''

'''
11	Zones urbanisées	Urban fabric
12	Zones industrielles ou commerciales et réseaux de communication	Industrial, commercial and transport units
13	Mines, décharges et chantiers	Mine, dump and construction sites
14	Espaces verts artificialisés, non agricoles	Artificial, non-agricultural vegetated areas
21	Terres arables	Arable land
22	Cultures permanentes	Permanent crops
23	Prairies	Pastures
24	Zones agricoles hétérogènes	Heterogeneous agricultural areas
31	Forêts	Forests
32	Milieux à végétation arbustive et/ou herbacée	Scrub and/or herbaceous vegetation associations
33	Espaces ouverts, sans ou avec peu de végétation	Open spaces with little or no vegetation
41	Zones humides intérieures	Inland wetlands
42	Zones humides côtières	Coastal wetlands
51	Eaux continentales	Inland waters
52	Eaux maritimes	Marine waters
'''


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



num_names={
'bio_1':'TMeanY',
'bio_2':'TMeanRngD',
'bio_3':'TIso',
'bio_4':'TSeason',
'bio_5':'TMaxWarmM',
'bio_6':'TMinColdM',
'bio_7':'TRngY',
'bio_8':'TMeanWetQ',
'bio_9':'TMeanDryQ',
'bio_10':'TMeanWarmQ',
'bio_11':'TMeanColdQ',
'bio_12':'PTotY',
'bio_13':'PWetM',
'bio_14':'PDryM',
'bio_15':'PSeason',
'bio_16':'PWetQ',
'bio_17':'PDryQ',
'bio_18':'PWarmQ',
'bio_19':'PColdQ',
'pH':'pH',
'Carbone':'C',
'CN':'CN',
'clay':'clay',
'silt':'silt',
'proxi_eau_fast':'proxi_eau_fast',
#'sand',
'awc_top':'awc_top',
'bs_top':'bs_top',
'cec_top':'cec_top',
'dgh':'dgh',
'dimp':'dimp',
'pd_top':'pd_top'
 }


lvars={**num_names,**{v:v for v in cat_vars}}


prep_config={
        'ctrt':'onehot',
        'categories':[cat_names.get(k)[1] for k in cat],
        'numeric':num,
        'categorical':cat,
        'ordinal':ordi,
        'binary':bina
        }

################## ######### ######### ######### ######### ######### ######### ######### ######### #########  

#################################
'''
                            Wrapper functions for external calls
'''
#################################

'''
Preprocess raw input
'''

def preprocess_df(prep_config):
    numeric_features = prep_config.get('numeric')
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', skprep.StandardScaler())])
    
    categorical_features = prep_config.get('categorical')
    steps=[
        ('imputer', SimpleImputer(missing_values=-1,strategy='most_frequent'))]
    
    if prep_config['ctrt']=='onehot':
        steps.append(('onehot', skprep.OneHotEncoder(categories=prep_config['categories'])))
        
    else:
        steps.append(('embed', skprep.OrdinalEncoder(categories=prep_config['categories'])))
    
    categorical_transformer = Pipeline(steps=steps)
    
    ordinal_features = prep_config.get('ordinal')
    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=-1,strategy='median')),
        ('scaler', skprep.StandardScaler())])
    
    binary_features = prep_config.get('binary')
    binary_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=-1,strategy='median'))])
            
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('bin', binary_transformer, binary_features),
            ('ord', ordinal_transformer, ordinal_features),
            ('cat', categorical_transformer, categorical_features)]
        )
    return  preprocessor

prep=preprocess_df(prep_config).fit(env_raw[num_vars+cat_vars])
varnames=num_vars+[c+'_%d'%k for c in cat_vars for k,_ in enumerate(cat_names[c][1])]
#env_dataset[0]['num_pvars']+env_dataset[0]['cat_pvars']

def normalize_env(prep,X_raw):
    ## prep: data prep object 
    ## X_raw: raw environmental dataframe
    df=pd.DataFrame(X_raw,columns=num_vars+cat_vars)
    df[cat_vars]=df[cat_vars].astype(int)
    data=prep.transform(df)
    X_norm=pd.DataFrame(data,columns=varnames)    
    return X_norm


'''
                        Wrapper for predict function
'''
def wrap_predict(model,t):  ### t could be a single value or multiple values
    def predict(X):  ### X should be in data frame format
        X_in=normalize_env(prep, X)
        y_pred=model.predict_community(X_in,vb=0)[1]
        return y_pred[:,t]
        
    return predict
