# -*- coding: utf-8 -*-
"""
Created on Sun May  3 19:18:53 2020

@author: saras
"""

import argparse
from ew_interp_utils import *
from scipy import stats
from ast import literal_eval as make_tuple

#from skbio.math.stats.distance import mantel

from skbio.stats.distance._mantel import mantel

from skmultilearn.model_selection import IterativeStratification
import sklearn.metrics as skm

import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.colors import rgb2hex, rgb_to_hsv
import scipy.spatial as sp, scipy.cluster.hierarchy as hc
import pingouin as pg

from scipy import cluster

from scipy.cluster.hierarchy import dendrogram, leaves_list
from eli5.permutation_importance import get_score_importances


### Architecture grid
archis=[('sh0',[]),
        ('sh8',[8]),
        ('sh16',[16]),
        ('sh16-8',[16,8]),
        ('sh32-16',[32,16]),
        #('sh24-16',[24,16]),
        ('sh32-16-8',[32,16,8])]  #('sh16-8',[16,8]),
regs=[(l1,l2) for l1,l2 in itertools.product([0.,0.0001,0.001],[0.,0.0001,0.001])]

 
### Losses
losses=[('BCE',lambda w: 'binary_crossentropy'),
        ('WBCE',lambda w: bce(bw=w,lw=np.ones_like(w),avg=True)),
        ('FC2',lambda w: stable_focal_loss(gamma=2,alpha=w)),
        #('FC5',lambda w: stable_focal_loss(gamma=5,alpha=w))
        ]

optims={'adam':'adam',
        #'adamax':'adamax',
        #'amsgrad':'amsgrad',
        'adamW3':tfa.optimizers.AdamW(weight_decay=0.0001),
        'adamW2':tfa.optimizers.AdamW(weight_decay=0.001),
        'adamW1':tfa.optimizers.AdamW(weight_decay=0.01)
        }

acts=['sigmoid','probit']
r_list=[5]

bias_list=[True]

### Fixed parameters
di=0 
th=0.5  
ep=200
bs=16
vb=0
tb=False
es=True  
logdir='empirical/log/' 

cpt=0
params=[]
for fi, r, reg, li, optim, act, use_bias in itertools.product(np.arange(len(archis)),r_list,regs,np.arange(len(losses)),optims.keys(),acts,bias_list):
    
    logreg=abs(np.log10(reg))
    logreg[logreg==np.inf]=0
    name=archis[fi][0]+'_%d'%r+'_L%dR%d'%tuple(logreg)+'_'+losses[li][0]+'_'+optim+'_'+act+'_'+('bias' if use_bias else 'nobias')
    
    params.append({'fi':fi,
                   'r':r,
                   'reg':reg,
                   'li':li,
                   'optim':optim,
                   'act':act,
                   'use_bias':use_bias,
                   'ep':ep,
                   'bs':bs,
                   'vb':vb,
                   'tb':tb,'es':es,'logdir':logdir,'di':di,'name':name
                   })
    cpt+=1

class EvalCbk(tfk.callbacks.Callback):
    def __init__(self,dvjsdm=None,evalset=None):
        self.dvjsdm=dvjsdm
        self.evalset=evalset
        self.scores=[]
        self.max_score=0
        self.weights=[]
        
    def on_epoch_end(self,epoch,logs=None):
        y_pred=self.dvjsdm.predict_community(self.evalset.get('X'),bs=16,vb=1)[1]
        #print(y_pred)
        ## Evaluate
        probs=y_pred[self.evalset.get('y')]  ##Extract only probabilities for observed species
        score=np.mean((np.array(probs)>th).astype(int))
        self.scores.append(score)
        
        if score>self.max_score:
            self.max_score=score
        
        self.weights.append(self.dvjsdm.mtlvae.mt_vae.get_weights())
        
        
def fit_config(name, ##name of the configuration
               ##Dataset, fold
               di=0,k=-1,f=-1,evsplit=None,
               
               ##Architecture of HSM, embedding dimension (r = max 5 see Clark paper and Warton 2015)
               act='sigmoid',
               fi=0,diag=False,beta=1.0,
               r=3,  
               reg=((0.,0.),(0.,0.)),
               
               ##Optimization options: optimizer, loss, initialization
               optim='adam',
               li=0,use_bias=True,
               ## Training configuration, callbacks
               ep=200,bs=16,vb=1,
               tb=False,es=False,logdir='log/'):
    
    os.makedirs(logdir+name+'/',exist_ok=True)
    
    '''
    Prepare dataset
    '''
    d=env_dataset[di]
    env=d['data']
    envc=d['cdata']
    num_vars=d['num_pvars']
    cat_vars=d['cat_pvars'] 
    
    if k==0:
        train,test=mlcv[f]
        idx_train=[sel_stations[i] for i in train]
        idx_test=[sel_stations[i] for i in test]
        
    elif k==1:
        test,train=mlcv[f]
        
        idx_train=[sel_stations[i] for i in train]
        idx_test=[sel_stations[i] for i in test]
    
    else:
        ### Train on full dataset ###
        idx_train=sel_stations
        idx_test=sel_stations
    
    Y_train=Y.loc[idx_train]
    Y_test=Y.loc[idx_test]
    
    X_train=env.loc[idx_train,num_vars+cat_vars]
    X_test=env.loc[idx_test,num_vars+cat_vars]     

    ######### Datasets ####
    train_dataset=(X_train,Y_train)
    test_dataset=(X_test,Y_test)
    
    if k==-1:
        eval_dataset={'X':envc.loc[sel_eval,num_vars],#]+[envc[[c]].astype(int) for c in cat_vars],
                  'y':(np.arange(len(curr)),curr['tnum'].values.astype(int))}
        
    ### dimensions
    m=Y.shape[1]
    ntr=len(idx_train)
    nte=len(idx_test)
    p=len(num_vars) #+cat_vars


    '''
    Architecture
    '''
    archi_id, nn=archis[fi]
    dvjsdm=DVJsdm(name=name,dims=(m,p,r),nn=nn,act=act,reg=reg,diag=diag,beta=beta)
    
    '''
    Training
    '''
    cbks=[]
    if tb:
        cbks.append(tfk.callbacks.TensorBoard(
                                    log_dir=logdir+name,
                                    write_graph=False,
                                    #embeddings_layer_names=[name'_noise/sqsigma'],  ##put here categ embedding as well as if rank>0 sigma_sqrt
                                    update_freq='epoch',
                                    write_images=True,
                                    #write_grads=True,
                                    profile_batch=0,
                                    histogram_freq=5))
        
    if es:
        cbks.append(tfk.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=20,restore_best_weights=True)) 
    
    ## eval callback
    evcbk=EvalCbk(dvjsdm,eval_dataset)
    cbks.append(evcbk)
    hist=dvjsdm.fit_model(train_dataset,evalset=test_dataset if evsplit is None else None,opt=optims.get(optim),
                     epoch=ep,bs=bs,init_bias=use_bias,loss=losses[li][1],
                     prevs=Y.mean(axis=0),use_cw=(li!=0),evsplit=evsplit,
                     vb=vb,cbks=cbks)
    
    scores=dvjsdm.evaluate_model(test_dataset)
    
    ## Predict current data
    if k==-1: ##if model trained on full data otherwise skip projection
        x_pred,y_pred=dvjsdm.predict_community(eval_dataset.get('X'),bs=16,vb=1)
    
        ## Evaluate
        probs=y_pred[eval_dataset.get('y')]  ##Extract only probabilities for observed species
        scores['eval_recall']=np.mean((np.array(probs)>th).astype(int))
        
    '''
    Summarize configuration 
    '''
    conf={}
    conf['name']=name
    conf['fold']=f
    conf['k']=k
    conf['archi']=archi_id
    conf['act']=act
    conf['r']=r
    conf['usebias']=use_bias
    conf['opt']=optim 
    conf['es']=str(es) ##whether early stopping was used 
    conf['loss']=losses[li][0]
    
    ## Update conf with scores
    return {**conf,**scores}, dvjsdm, evcbk, hist

alpha=0.05
def model_selection():
    l_res=[]
    for i, conf in enumerate(params):
        resi=pd.read_csv(logdir+'results_%s.csv'%conf['name'],index_col=0)
        resi['reg']=str(params[i]['reg'])
        resi['loss_fct']=losses[conf['li']][0]
        l_res.append(resi)
        
    full_res=pd.concat(l_res,axis=0)
    
    comp=[]
    metrics=['binary_accuracy', 'auc', 'precision', 'recall','precision_at_recall', 'sensitivity_at_specificity']
    models=full_res['name'].unique().tolist()
    for a,b in itertools.combinations(models, 2):
        for met in metrics:
            comp.append((a,b,met,paired5x2cv(a, b, met)))
            
    ## Best model ##
    comp_df=pd.DataFrame(data=np.array(comp),
                         columns=['a','b','met','ttest'])
    
    
    comp_df['signif']=comp_df['ttest'].apply(lambda x: x[1]<alpha)
    
    for i in range(len(comp)):
        comp_df.loc[i,'a_score']=full_res.query('name=="%s"'%comp[i][0])[comp[i][2]].mean()
        comp_df.loc[i,'b_score']=full_res.query('name=="%s"'%comp[i][1])[comp[i][2]].mean()
    
    #met='precision_at_recall'
    full_res['l1_l2']=full_res['reg'].apply(lambda x: make_tuple(x))
    full_res['l1']=full_res['reg'].apply(lambda x: make_tuple(x)[0])
    full_res['l2']=full_res['reg'].apply(lambda x: make_tuple(x)[1])
    
    sel_regs=[(0.0,0.0),(0.0,0.0001),(0.0001,0.0),(0.0001,0.0001)]
    subdata=full_res.query('loss_fct=="BCE" & act=="probit" & l1_l2 in @sel_regs')
    for met in metrics:
        fig, ax=plt.subplots(nrows=3,ncols=1,figsize=(20,15))
        sns.boxplot(data=subdata,x='archi',y=met,hue='loss_fct',ax=ax[0])
        sns.boxplot(data=subdata,x='archi',y=met,hue='act',ax=ax[1])
        sns.boxplot(data=subdata,x='archi',y=met,hue='reg',ax=ax[2])
        
        fig.savefig('BCE_reg_probit_%s.png'%met)
    
    ### Subselection ###
    ## loss_fct => BCE
    ## Activation choice cannot be done at this level (differences not signif)
    ## Architecture => indifferent
    
    
    ##1) Grid over model complexity 
    ##2) Grid over beta-regularization 
        
    cpt=0
    sel_params=[]
    for fi, r, reg, li, optim, act, use_bias, diag, beta in itertools.product(
            np.arange(len(archis)),
            [2,3,4,5],
            [(0.0001,0.0001)],
            [0,1],
            ['adam'],
            ['probit'],
            [True],
            [True,False],
            [1.0]):
        
        logreg=abs(np.log10(reg))
        logreg[logreg==np.inf]=0
        name=archis[fi][0]+'_%d'%r+'_L%dR%d'%tuple(logreg)+'_'+losses[li][0]+'_'+optim+'_'+act+'_'+('bias' if use_bias else 'nobias')
        
        sel_params.append({'fi':fi,
                       'r':r,
                       'reg':reg,
                       'li':li,
                       'optim':optim,
                       'act':act,
                       'use_bias':use_bias,
                       'ep':150,
                       'bs':bs,
                       'vb':vb,
                       'beta':beta,
                       'diag':diag,
                       'tb':tb,'es':es,'logdir':logdir,'di':di,'name':name
                       })
        cpt+=1
        
    results=[]
    for conf in sel_params[73:]:
        conf['f']=-1
        conf['k']=-1
        conf['vb']=1
        conf['tb']=False
        conf['es']=True
        conf['evsplit']=0.2
        res=fit_config(**conf)
        results.append({**conf,**res})

    
    results_df=pd.DataFrame.from_dict(results)
    
    ### Select best architecture , diag, dimension, loss function
    ##loss=WBCE
    ## diag ?   => False
    ## dimension ? =========> 3
    ## architecture ?  => sh16, sh168, sh3216
    subresults=results_df.query('li==1')
    sns.boxplot(data=subresults,x='r',y='eval_recall',hue='diag')
    sns.boxplot(data=subresults,x='r',y='auc',hue='diag')
    sns.boxplot(data=subresults,x='r',y='recall',hue='diag')
    
    ### Run selected architecture for various values of beta and wregularization and optimizer
    ## And activation
    
    cpt=0
    sel_params2=[]
    for fi, r, li, (optim,reg), act, use_bias, diag, beta in itertools.product(
            [2,3,4],
            [3],
            #[(0.0001,0.0001)],
            [1],
            [('adam',(0.0001,0.0)),('adam',(0.0001,0.0001)),
             ('adamW3',(0.0,0.0)),('adamW2',(0.0,0.0))],
            ['probit','sigmoid'],
            [True],
            [False],
            [1.0,2.0,3.0]):
        
        logreg=abs(np.log10(reg))
        logreg[logreg==np.inf]=0
        name=archis[fi][0]+'_%d'%r+'_L%dR%d'%tuple(logreg)+'_'+losses[li][0]+'_'+optim+'_'+act+'_'+('bias' if use_bias else 'nobias')
        
        sel_params2.append({'fi':fi,
                       'r':r,
                       'reg':reg,
                       'li':li,
                       'optim':optim,
                       'act':act,
                       'use_bias':use_bias,
                       'ep':150,
                       'bs':bs,
                       'vb':vb,
                       'beta':beta,
                       'diag':diag,
                       'tb':tb,'es':es,'logdir':logdir,'di':di,'name':name
                       })
        cpt+=1   
        
    
    ##Architecture (fe+activation) x regularization (wdecay vs elasticnet + betaVAE)   
    results2=[]
    for conf in sel_params2[33:]:
        conf['f']=-1
        conf['k']=-1
        conf['vb']=1
        conf['tb']=False
        conf['es']=True
        conf['evsplit']=None
        res=fit_config(**conf)
        results2.append({**conf,**res})

    results_df2=pd.DataFrame.from_dict(results2)    
    
    subresults2=results_df2.query('optim=="adam"')
    sns.boxplot(data=subresults2,x='archi',y='eval_recall',hue='beta')
    sns.boxplot(data=subresults2,x='archi',y='auc',hue='beta')
    sns.boxplot(data=subresults2,x='archi',y='recall',hue='beta')
    
    sns.boxplot(data=subresults2,x='archi',y='auc',hue='reg')
    sns.boxplot(data=subresults2,x='archi',y='recall',hue='reg')
    sns.boxplot(data=subresults2,x='archi',y='eval_recall',hue='reg')
    
    ##index of selected config in sel_params2
    ## I should try other activations for the fe component 
    
    return full_res, comp_df, results_df, results_df2

        
##paired t-tests to select best architecture
def paired5x2cv(full_res,a,b,met):
    x=full_res.query('name==@a')[['k','fold',met]].pivot_table(index='fold',columns='k',values=met)
    y=full_res.query('name==@b')[['k','fold',met]].pivot_table(index='fold',columns='k',values=met)
    
    diff=abs(x-y)
    
    meandiff=diff.mean(axis=1).values
    vardiff=diff.var(axis=1)#apply(lambda x: (x-meandiff)**2, axis=0).sum(axis=1)
    
    tt=diff.loc[0,0]/np.sqrt(float(vardiff.mean()))
    
    pval = stats.t.sf(np.abs(tt), 4)*2  # two-sided pvalue = Prob(abs(t)>tt)
    
    return(tt,pval)

def select_analyze_models():
    configs=[]
    cpt=0
    for di,li, fi, beta, diag, r, lmd in itertools.product([3,5],##0 done, remaining: 3(pca), 5(vif)
                                                           [0,1],[0,2,4],[1.0,2.0,3.0],[True],[3],[1,2,5,10]):
        os.makedirs(logdir+'%d'%di,exist_ok=True)
        code='%s_LV%d_%s_%d_%d_%s'%(archis[fi][0],r,str(diag),int(beta),lmd,losses[li][0])
        cpt+=1
        sel_config={'fi':fi,
                    'f':-1,
                    'k':-1,
                    'evsplit':None,
                    'r':r,
                    'reg':(lmd*1E-4,0.0001),
                    'li':li,
                    'optim':'adam',
                    'act':'probit',
                    'use_bias':True,
                    'ep':150,
                    'bs':16,
                    'vb':1,
                    'beta':beta,
                    'diag':diag,
                    'tb':False,'es':True,'logdir':logdir+'%d/'%di,'di':di,'name':code
                  }
        configs.append(sel_config)
    
    perfs=[]
    models=[]  
    hists=[]
    ##train on full dataset, visualize hsm and associations 
    for sel_config in configs:
        res, model, evcbk, hist=fit_config(**sel_config)
        _=model.visualize_embeddings(file=sel_config['logdir']+'%s/'%sel_config['name'])
        perfs.append(res)
        models.append(evcbk)
        hists.append(hist)

    perf_df=pd.DataFrame.from_dict(perfs)
    gap={perf_df.loc[x,'name']:x+72 for x in range(72)}        
    
    allhist=[]
    for i, dv in enumerate(models): 
        di=3 if i <72 else 5
        ### Evaluate on all current dataset 
        envc=env_dataset[di]['cdata']
        num_vars=env_dataset[di]['num_pvars']
        cat_vars=env_dataset[di]['cat_pvars']
        eval_dataset={'X':envc.loc[:,num_vars],#]+[envc[[c]].astype(int) for c in cat_vars],
                  'y':(np.arange(len(curr)),curr['tnum'].values.astype(int))}        
        y_pred=dv.dvjsdm.predict_community(E=eval_dataset.get('X'))[1]
        probs=y_pred[eval_dataset.get('y')]  ##Extract only probabilities for observed species
        perfs[i]={**perfs[i],'eval_curr':np.mean((np.array(probs)>th).astype(int))}          
        
        ##History data
        eval_hist=dv.scores
        train_hist=pd.DataFrame(hists[i].history)
        train_hist['epoch']=train_hist.index.tolist()
        train_hist['model']=res['name']
        train_hist['dataset']=train_hist['model'].apply(lambda x: gap.get(x))
        train_hist['eval']=eval_hist
        
        allhist.append(train_hist)
        
    histdf=pd.concat(allhist,axis=0,ignore_index=True)
    histdf['dataset']=histdf['epoch'].apply(lambda i: 3 if i<72 else 5)
    
    perf_df=pd.DataFrame.from_dict(perfs)
    perf_df['dataset']=[3 if x<72 else 5 for x in range(len(perf_df))]

    histdf.to_csv('empirical/log/history_35.csv')
    perf_df.to_csv('perf_35.csv')
    
    ## Combine to selected model
    sel_df=pd.read_csv('empirical/select.csv',sep=',',decimal='.',index_col=0)
    sel_df['dataset']=0 

    all_select=pd.concat([sel_df,perf_df],axis=0)       
    
    all_select.to_csv('empirical/log/selection.csv',sep=',',decimal='.')
    
    ### Visualize parameters
    for i, dv in enumerate(models):
        code=perfs[i]['name']
        feparams=dv.mtlvae.abio_fe.get_weights()
        for j, f in enumerate(feparams):
            g=sns.heatmap(data=f, cmap='seismic',center=0)
            g.set_title(code+' %d'%j)
            g.get_figure().savefig(logdir+code+'/shared_%d.png'%j)
            plt.close()
            
        params=dv.visualize_embeddings()['loadings']
        g=sns.clustermap(data=params.T, cmap='seismic',center=0,col_cluster=False)
        g.fig.suptitle(code+' embeddings')
        g.fig.savefig(logdir+code+'/embeddings.png')
        plt.close()
        
        context=dv.mtlvae.encoder_net.get_weights()[0]
        
        g=sns.clustermap(data=context, cmap='seismic',center=0,col_cluster=False)
        g.fig.suptitle(code+' context')
        g.fig.savefig(logdir+code+'/context.png')
        plt.close()        

    sel_config={'fi':2,
            'f':-1,
            'k':-1,
            'evsplit':0.2,
            'r':r,
            'reg':(0.0005,0.0001),
            'li':1,
            'optim':'adam',
            'act':'probit',
            'use_bias':True,
            'ep':150,
            'bs':16,
            'vb':1,
            'beta':2.0,
            'diag':False,
            'tb':False,'es':True,'logdir':logdir,'di':0,'name':'sh16_LV3_False_2_5_WBCE'
          }
    
    id_model=19
    
    ### Predictions ###
    dv=models[id_model]
    pred=dv.predict_community(env_dataset[0]['data'].loc[sel_stations])
    x_pred=pred[0]
    y_pred=pred[1]
    y_true=Y
    ### Species-wise performances ###
    task_perfs=pd.DataFrame.from_dict(eval_task(y_true.values, y_pred,taxa=None,th=0.5))
    task_perfs['names']=task_perfs['taxa'].apply(lambda x: names[x])
    return models[id_model]
    
final={'fi':2,
        'f':-1,
        'k':-1,
        'evsplit':None,
        'r':3,
        'reg':(0.001,0.0001),
        'li':1,
        'optim':'adam',
        'act':'probit',
        'use_bias':True,
        'ep':200,
        'bs':16,
        'vb':1,
        'beta':3.0,
        'diag':True,
        'tb':False,'es':True,'logdir':logdir,'di':0,'name':'lombrics'
      }  


di=0
eval_dataset={'X':env_dataset[di]['cdata'].loc[sel_eval,env_dataset[di]['num_pvars']],#]+[envc[[c]].astype(int) for c in cat_vars],
          'y':(np.arange(len(curr)),curr['tnum'].values.astype(int))}

th=0.5
ep=200
def select_model_evaluation(final,save):
    ### Fit selected architecture ### 
    res, dv, evcbk, hist=fit_config(**final)
    
    scores=pd.DataFrame(np.array([
        hist.history['auc'],
        hist.history['recall'],
        evcbk.scores
        ]).T,columns=['auc','recall','eval'])
    
    scores['epoch']=scores.index.tolist()   
    
    fig, axs=plt.subplots(3,1,figsize=(7,10),sharex=True)
    sns.scatterplot(data=scores,y='auc',x='epoch',ax=axs[0])
    sns.scatterplot(data=scores,y='recall',x='epoch',ax=axs[1])
    sns.scatterplot(data=scores,y='eval',x='epoch',ax=axs[2])
    
    ep=np.argmax(evcbk.scores)
    
    sel_weights=evcbk.weights[ep]
    dv.mtlvae.mt_vae.set_weights(sel_weights)
    
    dv.mtlvae.mt_vae.save_weights('empirical/'+save+'.h5')
    
    ### Past predictions ###
    pred=dv.predict_community(env_dataset[0]['data'].loc[sel_stations])
    x_pred=pred[0]
    y_pred=pred[1]
    y_true=Y
    ### Species-wise performances ###
    task_perfs=pd.DataFrame.from_dict(eval_task(y_true.values, y_pred,taxa=None,th=0.5))
    task_perfs['names']=task_perfs['taxa'].apply(lambda x: names[x])
    
    task_perfs.to_csv('empirical/final/taskwise.csv',sep=',',decimal='.')    

f_model='empirical/best/selmodel_87_77_75_full.h5'
l_model='empirical/linear.h5'
def analyze_final_model(f_model):
    ### Instantiate model object ###
    res, dvjsdm, evcbk, hist=fit_config(**final)
    
    ### Load pretrained weights ###
    dvjsdm.mtlvae.mt_vae.load_weights(f_model)
    #dvjsdm.compiled=True
    #dvjsdm.fitted=True
    
    ### Visualize
    params=dvjsdm.visualize_embeddings()#◘logdir='empirical/best/')
    ### Visualize parameters
    
    ### Posterior covariance matrix
    
    code='best/'#final['name']
    feparams=dvjsdm.mtlvae.abio_fe.get_weights()
    for j, f in enumerate(feparams):
        g=sns.heatmap(data=f, cmap='seismic',center=0)
        g.set_title('Shared feature extraction %d'%j)
        g.get_figure().savefig(logdir+code+'/shared_%d.png'%j)
        plt.close()
        
    g=sns.clustermap(data=params['loadings'].T, cmap='seismic',center=0,col_cluster=False)
    g.fig.suptitle('Embeddings')
    g.fig.savefig(logdir+code+'/embeddings.png')
    plt.close()
    
    context=dvjsdm.mtlvae.encoder_net.get_weights()[0]
    g=sns.clustermap(data=context, cmap='seismic',center=0,col_cluster=False)
    g.fig.suptitle('Context')
    g.fig.savefig(logdir+code+'/context.png')
    #plt.close()  

    ## Analyze w.r.t phylogeny and functional groups ##
    shared=feparams[0]   ### 73 x 16
    beta=params['beta']  ### 16 x 77
    loadings=params['loadings'] ### 3 x 77
    ctx_emb=context.T   ### 6 x 77 [3 for mean and 3 for variance]
        
    ## Phylogenetic distance
    ## Taxa with no phylogenetic information
    phyl_mat['sp_id_x']=phyl_mat['sp_id_x'].astype(int)
    phyl_mat['sp_id_y']=phyl_mat['sp_id_y'].astype(int)
    print('Taxa with no phylo information \n' , [taxa_names[t] for t in ret_taxa if t not in phyl_mat['sp_id_x'].unique()])
    ret_taxa_=[t for t in ret_taxa if t in phyl_mat['sp_id_x'].unique()]
    
    ## Response similarity
    abio_sim=pd.DataFrame(cosine_similarity(beta.T),columns=ret_taxa,index=ret_taxa).loc[ret_taxa_,ret_taxa_]
    bio_sim=pd.DataFrame(cosine_similarity(loadings.T),columns=ret_taxa,index=ret_taxa).loc[ret_taxa_,ret_taxa_]
    
    ## Functional similarity
    fct_sim=pd.DataFrame(data=fct_dist[ret_taxa_,:][:,ret_taxa_],columns=ret_taxa_,index=ret_taxa_)
    
    ## Pairwise associations
    pairwise=phyl_mat.query('sp_id_x!=sp_id_y & sp_id_x in @ret_taxa_ & sp_id_y in @ret_taxa_').reset_index()[['sp_id_x', 'code_t_x', 'sp_id_y', 'code_t_y', 'dist']]
    pairwise['fct_dist']=[1-fct_sim.loc[pairwise.loc[i,'sp_id_x'],pairwise.loc[i,'sp_id_y']] for i in range(len(pairwise))]
    pairwise['abio_dist']=[1-abio_sim.loc[pairwise.loc[i,'sp_id_x'],pairwise.loc[i,'sp_id_y']] for i in range(len(pairwise))]    
    pairwise['bio_dist']=[1-bio_sim.loc[pairwise.loc[i,'sp_id_x'],pairwise.loc[i,'sp_id_y']] for i in range(len(pairwise))]    
    pairwise['bio_sim']=[bio_sim.loc[pairwise.loc[i,'sp_id_x'],pairwise.loc[i,'sp_id_y']] for i in range(len(pairwise))]    
    
    
    cols=['phylo','functional','abiotic','biotic','association']
    pairwise.columns=['x','x_name','y','y_name']+cols
    
    assocs=['phylo','functional','abiotic','biotic']
    
    '''
    Do functionally-phylogenetically similar species respond
    similarly to the environment ? to latent factors ? 
    M1) Dissimilarity analysis 
    1.1/ In general ?
    1.2/ Is it marked for certain abiotic conditions in particular ?
    
    M2) Analysis on raw parameters 
    2.1/ Relate functional coordinates to embeddings
    2.2/ Is the pattern consistent amongst certain genera/species ? 
    
    M3) Cluster analysis on species parameters (clustermap)
    '''
    ltaxa=[pcodes[t].strip() for t in ret_taxa_]
    
    # Overall
    plot_correlation(pairwise[assocs], 'spearman')
    plot_correlation(pairwise[assocs], 'kendall')
    plot_correlation(pairwise[assocs], 'pearson')
    
    DF_dism = pairwise[['x','y','phylo']].pivot_table(values='phylo',index='x',columns='y',fill_value=0).loc[ret_taxa_,ret_taxa_]   # distance matrix
    
    '''
                        Mantel test
    '''
    dphylo=pd.DataFrame(np.triu(DF_dism)+np.triu(DF_dism).T,columns=ltaxa,index=ltaxa)
    dfunct=1-fct_sim
    dfunct.columns=dfunct.index=ltaxa
    dabio=1-abio_sim
    dabio.columns=dabio.index=ltaxa
    dbio=1-bio_sim
    dbio.columns=dbio.index=ltaxa
    
    ## Fct-abio
    test_correlation_mantel(x=dfunct.values,y=dabio.values,meth='pearson')
    test_correlation_mantel(x=dfunct.values,y=dabio.values,meth='spearman')
    
    ## Fct-bio
    test_correlation_mantel(x=dfunct.values,y=dbio.values,meth='pearson')
    test_correlation_mantel(x=dfunct.values,y=dbio.values,meth='spearman')
    
    ## Phylo-abio
    test_correlation_mantel(x=dphylo.values,y=dabio.values,meth='pearson')
    test_correlation_mantel(x=dphylo.values,y=dabio.values,meth='spearman')
    
    ## Phylo-bio
    test_correlation_mantel(x=dphylo.values,y=dbio.values,meth='pearson')
    test_correlation_mantel(x=dphylo.values,y=dbio.values,meth='spearman')
    
    ## Fct-Phylo
    test_correlation_mantel(x=dphylo.values,y=dfunct.values,meth='pearson')
    test_correlation_mantel(x=dphylo.values,y=dfunct.values,meth='spearman')
    
    '''
                    Visualize raw parameters
    '''
    # Assign colors to each taxa based on functional coordinates
    fg_raw=fct_groups.loc[ret_taxa_,:]/fct_groups.loc[ret_taxa_].sum(axis=1).values.reshape(-1,1)
    fg_raw['color_rgb']=[rgb2hex(x) for x in fg_raw.values[:,0:3]]
    fg_raw['taxa']=ltaxa
    #fg_raw['color_hsv']=[rgb_to_hsv(x) for x in fg_raw.values[:,0:3]]
    
    # Visualize raw parameters with colors of FG
    palette={code:color for code,color in fg_raw[['taxa','color_rgb']].values}
    
    # Structure of abiotic response
    env_resp=pd.DataFrame(beta.T,index=ret_taxa).loc[ret_taxa_]
    env_resp.index=ltaxa
    
    bio_resp=pd.DataFrame(loadings.T,index=ret_taxa).loc[ret_taxa_]
    bio_resp.index=ltaxa
   
    #sns.scatterplot(data=abio_,x=0,y=1,hue='taxa',palette=palette)
    
    # Visualize as a scatterplot with colors of FG
    sns.set(font="monospace",font_scale=2)
    
    '''
    RGB: Red (epigeic), Green(endogeic), Blue (Anecic)
    
    Abiotic
    '''
    clustermap_linkage(df=env_resp,dist=None,
                       title='Functional vs Response groups - Abiotic',palette=palette)
    
    clustermap_linkage(df=env_resp,
                       dist=dphylo,
                       meth='complete',
                       title='Phylogenetic structure - Abiotic',
                       col=False,
                       palette=palette)
    
    '''
    Biotic
    '''
    clustermap_linkage(df=bio_resp,dist=None,col=False,
                       title='Functional vs Latent groups - Biotic',palette=palette)
    
    clustermap_linkage(df=bio_resp,
                       dist=dphylo,
                       meth='complete',
                       title='Phylogenetic structure - Biotic',
                       col=False,
                       palette=palette) 
    
    bio_sim.columns=bio_sim.index=ltaxa
    g=sns.clustermap(data=bio_sim,cmap='seismic',yticklabels=1,xticklabels=1,figsize=(25,25))    
    for tick_label in g.ax_heatmap.axes.get_yticklabels():
        tick_text = tick_label.get_text()
        tick_label.set_color(palette.get(tick_text)) 
    
    for tick_label in g.ax_heatmap.axes.get_xticklabels():
        tick_text = tick_label.get_text()
        tick_label.set_color(palette.get(tick_text))     
    
        
    '''
    Contribution of environmental variables to shared env space dims
    '''
    envars=env_dataset[di]['num_pvars']
    shared_df=pd.DataFrame(data=shared,index=envars)
    
    ## Plot as stacked barplot 
    fig, ax=plt.subplots(1,1,figsize=(10,15))
    sns.heatmap(shared_df.T,cmap='seismic',center=0,yticklabels=1,ax=ax)
    
    

def clustermap_linkage(df,dist,meth='single',title='',palette=None,col=True,
                       vmin=-1,vmax=1,figsize=(10, 15)):
    if dist is None:
        linkage=None
    elif meth=='linkage':
        linkage = dist     
    else:
        np.fill_diagonal(dist, 0)
        linkage = hc.linkage(sp.distance.squareform(dist), 
                             method=meth, optimal_ordering=True)     
    
    g=sns.clustermap(df, row_linkage=linkage,col_linkage=linkage,
                   center=0,cmap='seismic',vmin=-1,vmax=1,
                   xticklabels=1,yticklabels=1,figsize=figsize) 
    
    if palette is not None:
        for tick_label in g.ax_heatmap.axes.get_yticklabels():
            tick_text = tick_label.get_text()
            tick_label.set_color(palette.get(tick_text))    
    if col:        
        for tick_label in g.ax_heatmap.axes.get_xticklabels():
            tick_text = tick_label.get_text()
            tick_label.set_color(palette.get(tick_text)) 
    g.fig.suptitle(title)
       
    
def plot_correlation(df,meth):
    m=df.shape[1]
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones((m,m), dtype=np.bool))
    g=sns.heatmap(df.corr(method=meth), 
                cmap='seismic',center=0,vmin=-1,vmax=1,mask=mask,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    g.set_title(meth.title()) 
    
    return g

def test_correlation_mantel(x,y,meth='pearson'):
    np.fill_diagonal(x,0)
    np.fill_diagonal(y,0)
    r,pv, _ =mantel(x,y,method=meth,alternative='two-sided')
    return r, pv 


m,p,r=(77,73,3)
n=Y.shape[0]

#####
'''
                        Threshold selection
'''
#####

### Data prep
X=normalize_env(prep, env_raw.loc[sel_stations,num_vars+cat_vars])
codes=[map_pcodes_r.get(t) for t in ret_taxa]

from sklearn.metrics import multilabel_confusion_matrix

def multi_eval(ytrue,ypred,eps=1E-4):
    ml_confusion=multilabel_confusion_matrix(ytrue,ypred)
    ml_scores=[]
    
    for j in range(m):
        tn, fp, fn, tp = ml_confusion[j,:,:].ravel()
        sens=tp/(tp+fn+eps)
        spec=tn/(tn+fp+eps)
        tss=sens+spec-1
        f1=(2*tp)/(2*tp+fp+fn+eps)
        acc=(tp+tn)/(tp+tn+fp+fn+eps)
        tpr=tp/(tp+fn+eps)
        tnr=tn/(tn+fp+eps)
        bacc=(tpr+tnr)/2
        ml_scores.append((j,sens,spec,tss,f1,acc,tpr,tnr,bacc))
        
    return pd.DataFrame(data=np.array(ml_scores),
                        columns=['tid','sensitivity','specificity','tss','f1','accuracy','tpr','tnr','bacc'])
    
    
def select_thCV():
    ## Generate CV datasets
    split=IterativeStratification(n_splits=5,random_state=1234).split(X,y=Y.values)
    
    #mod=8
    ## threshold
    th_list=np.arange(0.01,0.9,0.01)
    cv_preds=[]
    for train,test in split:  
        #train=test=range(n)
        ## trainset and testset
        trainset=[X.iloc[train],Y.iloc[train]]
        testset=[X.iloc[test],Y.iloc[test]]
        
        ## Create model
        #dvjsdm=DVJsdm(name='Lombrics_MT',dims=(m,p,r),nn=[16],act='probit',reg=(0.001,0.0001),diag=True,beta=3.0)
        dvjsdm=DVJsdm(name='Lombrics_MT',dims=(m,p,r),nn=[16],feact='relu',act='probit',reg=(0.0001,0.0001),diag=True,beta=1.0)
        #16,8 worse
        #16, sigmoid no early stopping better 
        #linear model better, but eval much lower
        #32 waaay better in prediction (all above 72)
        #common taxa only
        
        ## Fit 
        # hist=dvjsdm.fit_model(trainset,evalset=testset,opt='adam',
        #              epoch=200,bs=32,init_bias=True,loss=losses[1][1],
        #              prevs=Y.mean(axis=0),use_cw=(li!=0),evsplit=None,
        #              vb=1,cbks=[tfk.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=20,restore_best_weights=True)])
        
        
        evcbk=EvalCbk(dvjsdm,eval_dataset)
        li=1
        hist=dvjsdm.fit_model(trainset,evalset=testset,opt='adam',#tfk.optimizers.SGD(lr=0.001,nesterov=True),
                     epoch=200,bs=32,
                     init_bias=True,
                     loss=losses[li][1],
                     prevs=Y.mean(axis=0),use_cw=(li!=0),evsplit=None,
                     cbks=[evcbk],
                     vb=1#,cbks=[tfk.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=20,restore_best_weights=True)]
                     )
        
        #dvjsdm.mtlvae.mt_vae.set_weights(evcbk.weights[9])
        #dvjsdm.mtlvae.mt_vae.load_weights('checkpoints/0.h5')
        ## Predict
        pred=dvjsdm.predict_community(E=testset[0],bs=16,vb=1)
        #cv_preds.append((pred[0],pred[1],testset[1]))
        
        past_scores=pd.DataFrame([(codes[j],skm.roc_auc_score(Y.values[:,j],pred[1][:,j])) for j in range(m)])
        past_scores.index=past_scores.values[:,0]
        
        ## select optimal threshold
        yt=Y.values
        yp=pred[1]
        all_scores=[]
        for th in th_list:
            scores=multi_eval(yt, (yp>=th).astype(int))
            scores['th']=th
            #scores['fold']=f
            scores['prev']=yt.mean(axis=0).tolist()
            scores['taxa']=codes
            all_scores.append(scores)        
        
        thsel=pd.concat(all_scores,axis=0)
        
        thrs=[]
        for tax in codes:
            subdata=thsel.query('taxa==@tax')
            goal=subdata.tss.max()
            argmax=subdata.query('tss>=@goal')['th'].min()
            thrs.append(argmax)
            
        
        ## TSS
        o=multi_eval(yt,(pred[1]>0.5).astype(int))
        
        ## Evaluate
        x_pred,y_pred=dvjsdm.predict_community(eval_dataset.get('X'),bs=16,vb=1)
        evnames=[codes[j] for j in eval_dataset.get('y')[1]]
        probs=pd.DataFrame(np.stack([evnames,y_pred[eval_dataset.get('y')]]).T,columns=['code','prob'])  ##Extract only probabilities for observed species
        probs.prob=probs.prob.astype(float)
        thr=max(min(thrs),0.3)
        np.mean(np.array(probs.prob>thr).astype(int)) 
        
        eval_scores=probs.groupby('code').agg(lambda x: np.mean(x>thr)) 
        past_scores['eval']=[eval_scores.loc[c,'prob'] if c in eval_scores.index else -1 for c in past_scores.index]
        past_scores['tss']=o['tss'].values
        past_scores['opt_th']=thrs
        
        past_scores.to_csv('checkpoints/%d.csv'%mod,sep=',',decimal='.')
        dvjsdm.mtlvae.mt_vae.save_weights('checkpoints/%d.h5'%mod)
        
        
    ## evaluate
    all_scores=[]
    f=0
    for xp, yp, yt in cv_preds:
        for th in th_list:
            scores=multi_eval(yt, (yp>=th).astype(int))
            scores['th']=th
            scores['fold']=f
            scores['prev']=yt.mean(axis=0).tolist()
            scores['taxa']=codes
            all_scores.append(scores)
        
        f+=1
    
    tmax=0.75    
    
    th_score=pd.concat(all_scores,axis=0)
    avg=th_score.groupby(['tid','taxa','th']).mean()
    sd=th_score.groupby(['tid','taxa','th']).std()
    
    ## median scores
    med=th_score.query('prev>0 & th<=@tmax').groupby(['tid','taxa','th']).median().reset_index()
    
    ### Selection
    selec=[]
    for met in ['tss','f1','bacc']:
        for t in range(m):
            data=med.query('tid==@t')
            best=data[met].max()
            th_sel=data.query(met+'>=@best')['th'].min()
            
            selec.append({'metric':met,'taxa':codes[t],'tid':t,'th':th_sel,'score':best})
            
    
    sel_out=pd.DataFrame.from_dict(selec)
    
    th_score.to_csv('th_selection_raw.csv',sep=',')
    med.to_csv('th_selection_med.csv',sep=',')
    sel_out.to_csv('selected.csv',sep=',')
    
    return sel_out



#####
'''
                        Feature importance plots
'''
#####

ltaxa=[pcodes[j].strip() for j in ret_taxa]
def feature_importance(f_model,folder_interp='empirical/iml/'):
    # res, dvjsdm, evcbk, hist=fit_config(**final)
    # w1=dvjsdm.mtlvae.mt_vae.get_weights()
    # dvjsdm.mtlvae.mt_vae.load_weights(f_model)
    # w1l=dvjsdm.mtlvae.mt_vae.get_weights()
    
    # res, dvjsdm, evcbk, hist=fit_config(**final)
    # w2=dvjsdm.mtlvae.mt_vae.get_weights()
    # dvjsdm.mtlvae.mt_vae.load_weights(f_model)
    # w2l=dvjsdm.mtlvae.mt_vae.get_weights()
    
    m,p,r=(77,73,3)
    dvjsdm=DVJsdm(name='Lombrics_MT',dims=(m,p,r),nn=[16],act='probit',reg=(0.001,0.0001),diag=True,beta=3.0)
    #dvjsdm.compile_model()
    
    ### Load pretrained weights ###
    dvjsdm.mtlvae.mt_vae.load_weights(f_model)
    # dvjsdm.compiled=True
    # dvjsdm.fitted=True
    
    # x_pred,y_pred=dvjsdm.predict_community(E=eval_dataset.get('X'),bs=16,vb=1)
    # ## Evaluate
    # probs=y_pred[eval_dataset.get('y')]  ##Extract only probabilities for observed species
    # np.mean((np.array(probs)>th).astype(int))
    
    ### Call feature impportance routines
    low_color='red'
    os.makedirs(folder_interp, exist_ok=True)
    
    '''
                    Taxa-wise variable importance
    '''
    
    alltaxa=np.arange(m).tolist()
    model=dvjsdm
    
    X=env_raw.loc[sel_stations,num_vars+cat_vars]
    # X=normalize_env(prep, X_raw)
    # E=env_dataset[0]['data'].loc[sel_stations]
    
    # yw=wrap_predict(model, alltaxa)(X_raw)
    # y=dvjsdm.predict_community(E)[1]
    
    for metric in ['auc']:
        print('Feature importance for metric: %s'%metric)
        
        base_score_tasks, score_decreases_tasks = get_score_importances(
            score_func=score_j(mhsm=dvjsdm,pred_fun=wrap_predict(model,alltaxa),
                         j=alltaxa,m=len(alltaxa),metric=metric,mode=None,th=0.5), 
            X=X.values, y=Y.values,random_state=1234)
        
        met_df=pd.DataFrame((base_score_tasks).reshape((len(ltaxa),1)),columns=[lwlimit.get(metric)[0]],index=ltaxa)
        met_df.to_csv(folder_interp+'%s_scores.csv'%metric,sep=',',decimal='.')
        
        low_scores=[ltaxa[i] for i, x in enumerate(base_score_tasks) if x<lwlimit.get(metric)[1]]
        bplot=(met_df-lwlimit.get(metric)[1]).plot.bar(figsize=(15,10))
        
        
        for tick_label in bplot.axes.get_xticklabels():
            tick_text = tick_label.get_text()
            if tick_text in low_scores:
                tick_label.set_color(low_color)
        
        bplot.get_figure().savefig(folder_interp+'%s_scores.pdf' % metric,bbox_inches='tight')
    
    
        pim_scores_tasks=pd.DataFrame(data=np.stack(score_decreases_tasks).mean(axis=0).T,columns=num_vars+cat_vars,index=ltaxa)
        pim_scores_tasks.to_csv(folder_interp+'pim_tasks_%s.csv'%metric,sep=",",decimal='.')
        df_long = pd.melt(pim_scores_tasks, var_name="feature", value_name="score").merge(vargroups,on='feature').sort_values(by='group')
        df_long['score']=df_long['score'].abs()
        df_long['taxa']=ltaxa*X.shape[1]
        df_long['fname']=df_long['feature'].apply(lambda x: lvars.get(x))
        
        
        fig, ax = plt.subplots(figsize=(15,8))
        chart=sns.boxplot(ax=ax,x="fname", hue="group", y="score", data=df_long,palette=colors)
        chart.set(title='Permutation Importance Distributions Among Taxa', ylabel='Importance',xlabel='Variable')
        chart.set_xticklabels(chart.get_xticklabels(), rotation=45)                                                     
        fig.savefig(folder_interp+'pim_vargroups_%s.pdf'%metric,bbox_inches='tight')
        
        fig, ax = plt.subplots(figsize=(15,8))
        chart=sns.boxplot(ax=ax,x="group", hue="group", y="score", data=df_long,palette=colors)
        chart.set(title='Permutation Importance Distributions Among Groups',ylabel='Importance',xlabel='Variable')
        chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
        fig.savefig(folder_interp+'pim_groups_%s.pdf'%metric,bbox_inches='tight')
        
        tfold=folder_interp+'taxwise_%s/'%metric
        tfold_g=folder_interp+'taxwise_groups_%s/'%metric
        os.makedirs(tfold,exist_ok=True)
        os.makedirs(tfold_g,exist_ok=True)
        
        scores_taxa=np.stack(score_decreases_tasks)
        for t in range(len(ltaxa)):
            fig, ax = plt.subplots(figsize=(15,8))
            odata=pd.melt(pd.DataFrame(data=scores_taxa[:,:,t],columns=num_vars+cat_vars), var_name="feature", value_name="score").merge(vargroups,on='feature').sort_values(by='group')
            odata['fname']=odata['feature'].apply(lambda x: lvars.get(x))
            odata['score']=odata['score'].abs()
            chart=sns.boxplot(ax=ax,data=odata,x="fname",y="score",hue="group",palette=colors)
            chart.set(title='Average permutation Importance Distributions for taxa %s , base score %.2f' % (ltaxa[t],base_score_tasks[t]),ylabel='Importance')
            chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
            fig.savefig(tfold+ltaxa[t]+'.png')
            plt.close()
            
        for t in range(len(ltaxa)):
            fig, ax = plt.subplots(figsize=(15,8))
            odata=pd.melt(pd.DataFrame(data=scores_taxa[:,:,t],columns=num_vars+cat_vars), var_name="feature", value_name="score").merge(vargroups,on='feature').sort_values(by='group')
            odata['score']=odata['score'].abs()
            chart=sns.boxplot(ax=ax,data=odata,x="group",y="score",hue="group",palette=colors)
            chart.set(title='Average permutation Importance Distributions for taxa %s , base score %.2f' % (ltaxa[t],base_score_tasks[t]),ylabel='Importance')
            chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
            fig.savefig(tfold_g+ltaxa[t]+'.png')
            plt.close()
            
    for met, mode in itertools.product(['auc'],['micro','macro','weighted']):    
        metric=met+"_"+mode  
        base_score, score_decreases = get_score_importances(
            score_func=score_j(mhsm=model,pred_fun=wrap_predict(model,alltaxa),
                         j=alltaxa,m=len(alltaxa),metric=met,mode=mode,th=0.5), X=X.values, y=Y.values)
        
        
        #### Variable importance
        pim_scores=pd.DataFrame(data=np.stack(score_decreases),columns=num_vars+cat_vars).abs()
        pim_scores.to_csv(folder_interp+'pim_overall_%s.csv'%metric,sep=",",decimal='.')
        
        df_long_overall = pd.melt(pim_scores, var_name="feature", value_name="score").merge(vargroups,on='feature').sort_values(by='group')
        df_long_overall['fname']=df_long_overall['feature'].apply(lambda x: lvars.get(x))
        
        ### Plot ###
        fig, ax = plt.subplots(figsize=(15,8))
        chart=sns.boxplot(ax=ax,x="fname", hue="group", y="score", data=df_long_overall,palette=colors)
        chart.set(title='Permutation Importance Distributions, base_score %.2f'%base_score,ylabel='Importance',xlabel='Variable')
        chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
        fig.savefig(folder_interp+'pim_vargroups_overall_%s.pdf'%metric,bbox_inches='tight')
        
        ###
        fig, ax = plt.subplots(figsize=(15,8))
        chart=sns.boxplot(ax=ax,x="group", hue="group", y="score", data=df_long_overall,palette=colors)
        chart.set(title='Permutation Importance Distributions Overall Among Groups , base_score %.2f'%base_score,ylabel='Importance',xlabel='Variable')
        chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
        fig.savefig(folder_interp+'pim_groups_overall_%s.pdf'%metric,bbox_inches='tight')        
    
    
def save_embedding(model,X,save=False):
    env_emb=y=model.predict_community(E=X,vb=0)[0]
    sp_emb=model.visualize_embeddings().get('beta').T
    
    ### Visualize
    if save:
        all_emb=pd.DataFrame(np.concatenate([env_emb,sp_emb],axis=0))
        all_emb['id']=['S%d'%i for i in range(n)]+codes
        all_emb['type']=['site']*n+['taxa']*m
        
        all_emb[['id','type']].to_csv('metadata_emb.tsv',sep='\t',index=False)
        all_emb.iloc[:,0:16].to_csv('site_sp_emb.tsv',sep='\t',index=False)
        
    return env_emb, sp_emb


'''
                    Load final model
'''
model=DVJsdm(name='Lombrics_MT',dims=(m,p,r),nn=[16],act='probit',reg=(0.001,0.0001),diag=True,beta=3.0)
### Load pretrained weights ###
model.mtlvae.mt_vae.load_weights(f_model)

Ycond=model.mtlvae.mt_vae.predict([X.values,Y.values])
Yjoint=model.predict_community(E=X)[1]

# Yp=model.predict_community(E=X)[1]

jaucs=[skm.roc_auc_score(Y.values[:,j],Yjoint[:,j]) for j in range(m)]
caucs=[skm.roc_auc_score(Y.values[:,j],Ycond[:,j]) for j in range(m)]
#np.stack([codes,jaucs,caucs],axis=1)
# cov=skm.coverage_error(Y,Yp)
# microaurc=skm.roc_auc_score(Y,Yp,average='micro')
# perfs=[skm.roc_auc_score(Y.values[:,j],Yp[:,j]) for j in range(m)]
# sel_perfs=select_thCV()

## Save embeddings
all_emb=save_embedding(model, X)

'''
                   Functional groups color coding
'''

# Assign colors to each taxa based on functional coordinates
fg_raw=fct_groups.loc[ret_taxa,:]/fct_groups.loc[ret_taxa].sum(axis=1).values.reshape(-1,1)
fg_raw['color_rgb']=[rgb2hex(x) for x in fg_raw.values[:,0:3]]
fg_raw['taxa']=ltaxa

# Visualize raw parameters with colors of FG
palette={code:color for code,color in fg_raw[['taxa','color_rgb']].values}  

# Create species level mappings
map_sp=taxo_data[['drilo_code','Genus','Species']].query('drilo_code in @ret_taxa').drop_duplicates()
map_sp['tcode']=map_sp['drilo_code'].apply(lambda x: map_pcodes_r.get(x))
map_sp['scode']=map_sp['Genus'].apply(lambda x: x[0]) + '.' + map_sp['Species']

tax2sp={tc:sc for tc,sc in zip(map_sp.tcode,map_sp.scode)}

# Aggregate functional groups to species level
map_sp=pd.merge(map_sp,fg_raw,on='drilo_code')
fg_species=map_sp[['scode','Genus','Species','p_epigeic','p_endogeic','p_anecic']].groupby(['scode','Genus','Species']).agg('mean').reset_index()

# Create palette for species
fg_species['color']=[rgb2hex(x) for x in fg_species.values[:,3:6].astype(float)]
sppalette={code:color for code,color in fg_species[['scode','color']].values} 

clustcol={'0':'cyan',
          '1':'red',
          '2':'purple',
          '3':'goldenrod',
          '4':'green'}


def estimate_cov():
    ### Estimate of the covariance matrix
    varpar=model.mtlvae.encoder_net.predict(Y)
    varmu=varpar[:,0:r]
    varsigma=varpar[:,r:]
    varsum=np.diag(varsigma.sum(axis=0))
    
    sigmahat=(np.dot(varmu.T,varmu) + varsum)/Y.shape[0]
    
    loadings=model.visualize_embeddings()['loadings']
    sigmarhat=np.dot(np.dot(loadings.T,sigmahat),loadings)
    corhat=cov2corr(sigmarhat)
    
    ### Aggregate to species level
    tmp=pd.DataFrame(sigmarhat,columns=codes,index=codes).melt()
    tmp['variable2']=codes*len(codes)
    tmp['sp']=[tax2sp.get(c) for c in tmp['variable']]
    tmp['sp2']=[tax2sp.get(c) for c in tmp['variable2']]
    tmp=tmp.groupby(['sp','sp2']).agg('mean')
    
    spsigmarhat=tmp.pivot_table(index='sp',columns='sp2',values='value')
    spcodes=spsigmarhat.index.tolist()
    spcorhat=cov2corr(spsigmarhat)
    
    #### Export ###
    pd.DataFrame(sigmarhat,columns=codes,index=codes).to_csv('empirical/best/associations/sigmarhat.csv')
    pd.DataFrame(corhat,columns=codes,index=codes).to_csv('empirical/best/associations/corhat.csv')
    
    #### Export sp level
    pd.DataFrame(spsigmarhat,columns=spcodes,index=spcodes).to_csv('empirical/best/associations/spsigmarhat.csv')
    pd.DataFrame(spcorhat,columns=spcodes,index=spcodes).to_csv('empirical/best/associations/spcorhat.csv')
    
    
    ### Visualize ###
    g=sns.clustermap(data=pd.DataFrame(corhat,columns=codes,index=codes),
                     vmin=-1,vmax=1,cmap='seismic')
    for tick_label in g.ax_heatmap.axes.get_yticklabels():
        tick_text = tick_label.get_text()
        tick_label.set_color(palette.get(tick_text))  
        
    for tick_label in g.ax_heatmap.axes.get_xticklabels():
        tick_text = tick_label.get_text()
        tick_label.set_color(palette.get(tick_text)) 
        
    g.savefig('empirical/best/posterior_assoc_raw_fg.pdf')
    
    ### Get clusters
    hc=g.dendrogram_row.linkage
    assign=np.stack([codes,cluster.hierarchy.cut_tree(hc, n_clusters=5)[:,0],],axis=1)
    
    ### Plot dendrograms
    labels=[codes[i] for i in leaves_list(hc)]
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    dendrogram(hc,leaf_rotation=90,labels=labels,leaf_font_size=12,ax=ax)
    #ax = plt.gca()
    xlbls = ax.get_xmajorticklabels()
    for lbl in xlbls:
        lbl.set_color(palette[lbl.get_text()])
        
    fig.savefig('empirical/best/clusters_hierarchy.pdf',  bbox_inches='tight')
    
    pd.DataFrame(assign,columns=['taxa','cluster']).to_csv('empirical/best/associations/clusters.csv')
        
    #####################################################################################
    ### Visualize sp level ###
    g=sns.clustermap(data=pd.DataFrame(spcorhat,columns=spcodes,index=spcodes),
                     vmin=-1,vmax=1,cmap='seismic',xticklabels=1,yticklabels=1)
    for tick_label in g.ax_heatmap.axes.get_yticklabels():
        tick_text = tick_label.get_text()
        tick_label.set_color(sppalette.get(tick_text))  
        
    for tick_label in g.ax_heatmap.axes.get_xticklabels():
        tick_text = tick_label.get_text()
        tick_label.set_color(sppalette.get(tick_text)) 
        
    g.savefig('empirical/best/posterior_assoc_raw_fg_sp.pdf') 
    
    splabels=[spcodes[i] for i in leaves_list(sphc)]
    #splabels=[x.get_text() for x in g.ax_heatmap.axes.get_xticklabels()]
        
    ### Get row clusters
    sphc=g.dendrogram_row.linkage
    
    ### Assign clusters
    spgroups=cluster.hierarchy.cut_tree(sphc, n_clusters=5)[:,0]
    spassign=np.stack([spcodes,spgroups],axis=1)
    pd.DataFrame(spassign,columns=['species','cluster']).to_csv('empirical/best/associations/spclusters.csv')
    
    ###################### Assign colors per cluster
    spclustpalette={spassign[i,0]:clustcol.get(spassign[i,1]) for i in range(len(splabels))}
        
    ### Plot dendrograms
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    
    dendrogram(sphc,leaf_rotation=90,labels=spcodes,
               leaf_font_size=12,ax=ax)
    #ax = plt.gca()
    xlbls = ax.get_xmajorticklabels()
    for lbl in xlbls:
        lbl.set_color(spclustpalette[lbl.get_text()])
        
    fig.savefig('empirical/best/spclusters_hierarchy.pdf',  bbox_inches='tight')    
       
    
    ### cluster-colored species correlations ###
    g=sns.clustermap(data=pd.DataFrame(spcorhat,columns=spcodes,index=spcodes),
                     vmin=-1,vmax=1,cmap='seismic',xticklabels=1,yticklabels=1)
    for tick_label in g.ax_heatmap.axes.get_yticklabels():
        tick_text = tick_label.get_text()
        tick_label.set_color(spclustpalette.get(tick_text))  
        
    for tick_label in g.ax_heatmap.axes.get_xticklabels():
        tick_text = tick_label.get_text()
        tick_label.set_color(spclustpalette.get(tick_text)) 
        
    g.savefig('empirical/best/posterior_assoc_clust_sp.pdf')  
    
    ### cluster-colored species correlations filtered ###
    spcorhat[np.abs(spcorhat)<0.5]=0
    g=sns.clustermap(data=pd.DataFrame(spcorhat,columns=spcodes,index=spcodes).loc[splabels,splabels],
                     vmin=-1,vmax=1,cmap='seismic',xticklabels=1,yticklabels=1)
    for tick_label in g.ax_heatmap.axes.get_yticTaxwiseklabels():
        tick_text = tick_label.get_text()
        tick_label.set_color(spclustpalette.get(tick_text))  
        
    for tick_label in g.ax_heatmap.axes.get_xticklabels():
        tick_text = tick_label.get_text()
        tick_label.set_color(spclustpalette.get(tick_text)) 
        
    g.savefig('empirical/best/posterior_assoc_clust_sp_filt.pdf')

    ### Raw viz
    corhat[np.abs(corhat)<0.5]=0
    g=sns.clustermap(data=pd.DataFrame(corhat,columns=codes,index=codes), vmin=-1,vmax=1,cmap='seismic')
    g.savefig('empirical/best/posterior_assoc.pdf')
    
    load_sim=cosine_similarity(loadings.T)
    g=sns.clustermap(data=pd.DataFrame(load_sim,columns=codes,index=codes), vmin=-1,vmax=1,cmap='seismic')
    g.savefig('empirical/best/prior_assoc_raw.pdf')  
    
    load_sim[np.abs(load_sim)<0.5]=0
    g=sns.clustermap(data=pd.DataFrame(load_sim,columns=codes,index=codes), vmin=-1,vmax=1,cmap='seismic')
    g.savefig('empirical/best/prior_assoc.pdf')      
    

def save_prep_datasets():
    #### Save prepared datasets for fitting with BIOMOD 
    for di in [0,3,5]:
        num_vars=env_dataset[di]['num_pvars']
        X=env_dataset[di]['data'].loc[sel_stations,num_vars]
        Xc=env_dataset[di]['cdata'].loc[sel_eval,num_vars]
        Xc['Id_gtaxa']=curr['tnum'].values.astype(int)
        Xc['code']=Xc['Id_gtaxa'].apply(lambda t: map_pcodes_r.get(t))
        
        X.to_csv('biomod_eval/train_prep_%d.csv'%di,sep=',',decimal='.',index=False)
        Xc.to_csv('biomod_eval/eval_prep_%d.csv'%di,sep=',',decimal='.',index=False)
    
    Y.columns=codes
    Y.to_csv('biomod_eval/data/occur.csv',sep=',',decimal='.')
    
    coord.query('Id_Station in @sel_stations').to_csv('biomod_eval/data/coords.csv',sep=',',decimal='.')
    
        
#save_prep_datasets()