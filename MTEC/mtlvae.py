# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 11:51:57 2020

@author: saras
"""

from lego_blocks import *
from eval_functions import *
from scipy.special import logit, expit
from statsmodels.stats.moment_helpers import cov2corr
import seaborn as sns


class DVJsdm(object):
    """
    This creates a subtype of MTLVAE that models a LV-JSDM with non-linear abiotic response
    And amortized variational inference (using a VAE) for the latent variables
    """
    
    def __init__(self,name,dims,nn,act,reg,diag=False):
        self.mtlvae=MTLVAE(name)
        self.m,self.e,self.d=dims
        self.p=nn[-1] if len(nn)>0 else self.e
        
        
        ## Setting up configuration
        in_feat=[('num',{'id':'env','dim':self.e})]
        in_config=[('num',{'id':'taxa','dim':self.m})]
        
        fe_config={'name':'fe','archi':{'nbnum':self.e,'nl':len(nn),'nn':nn,'activation':'relu'},
                   'reg':{'regtype':'l1_l2','regparam':reg}}
        
        encoder_config={'name':'encoder','archi':{'nbnum':self.m,'nl':1,'nn':[self.d],'activation':'relu'},
                        'reg':{'regtype':'l1_l2','regparam':reg}}
        
        decoder_config={'name':'shared_decoder','archi':{'nbnum':self.p+self.d, 'nl':0,'nn':[],'activation':'linear'},
                        'reg':None}
        
        taxa={'name':'specific_decoder','archi':{'nbnum':self.p+self.d,'nl':1,'nn':[self.m],'fit_bias':True,'activation':'linear'},
                        'reg':{'regtype':'l1_l2','regparam':reg}}
        
        out_config=[ {'name':'pa','type':'binary','specific':taxa,'activation':act}]    
                
        ## Create MTLVAE architecture
        self.mtlvae.create_architecture(in_feat,in_config,fe_config,encoder_config,decoder_config,out_config,diag)
        
        ## Visualize components architectures
        tfkv.plot_model(self.mtlvae.abio_fe,show_shapes=True)
        tfkv.plot_model(self.mtlvae.encoder_net,show_shapes=True)
        tfkv.plot_model(self.mtlvae.decoder_net,show_shapes=True)
        tfkv.plot_model(self.mtlvae.mt_vae,show_shapes=True)
           
    def compile_model(self,opt='adam',loss='binary_crossentropy'):  ##loss=ce or fl, if fl provide also gamma 
        self.mtlvae.mt_vae.compile(loss=loss, optimizer=opt,metrics=metric_fn.get('classification'))
        #self.mtlvae.mt_pred.compile(loss=loss_fn.get('binomial')(cw), optimizer=opt,metrics=metric_fn.get('classification'))
    
    def fit_model(self,trainset,evalset=None,evsplit=0.1,
                  loss=lambda: 'binary_crossentropy',opt='adam',
                  init_bias=False,use_cw=False,prevs=None,
                  epoch=100,bs=16,vb=1,cbks=None):
        
        X, Y=trainset
        prevs=Y.mean(axis=0) if prevs is None else prevs  
        
        if evalset is not None:
            Xt, Yt=evalset
        
        ## Class weights
        cw=np.expand_dims(np.exp(-logit(prevs)),axis=0) if use_cw else np.ones((1,self.m))  
        cw=np.maximum(cw,1)
        
        ## Compile model
        self.compile_model(opt,loss(cw))
        self.compiled=True
        
        ### Init bias with prevalence to prevent the model from learning the imbalance ratio
        if init_bias:
            bias=logit(prevs)
            w_dec=self.mtlvae.decoder_net.get_weights() 
            w_dec[1]=bias
            self.mtlvae.decoder_net.set_weights(w_dec)
         
        ## Fit to data   
        self.mtlvae.mt_vae.fit(x=[X,Y],y=Y,batch_size=bs,epochs=epoch,verbose=vb,callbacks=cbks,
                               validation_split=0.0 if evalset is not None else evsplit,
                               validation_data=([Xt,Yt],Yt))
        
        self.fitted=True
        
    def evaluate_model(self,testset,vb=1):
        X, Y=testset
        if self.compiled:
            ## Evaluate
            scores=self.mtlvae.mt_vae.evaluate(x=[X,Y],y=Y,verbose=vb)
            
            named_scores={k:scores[i] for i,k in enumerate(self.mtlvae.mt_vae.metrics_names)}
            
            return named_scores
        
        else:
            raise('Model is not fitted, call fit_model with a training set.')
            
    
    def predict_community(self,E,bs=16,vb=1):
        if self.compiled:
            ##Predict 
            x_feat=self.mtlvae.abio_fe.predict(x=E,batch_size=bs,verbose=vb)
            noise=np.random.normal(loc=0,scale=1,size=(E.shape[0],self.d))
            y_pred=self.mtlvae.decoder_net.predict(x=np.concatenate([x_feat,noise],axis=1),batch_size=bs,verbose=vb)
            return x_feat, y_pred
        else:
            raise('Model is not fitted, call fit_model with a training set.')
            
    def visualize_embeddings(self,xlabels,ylabels):
        ## Extract embeddings
        emb=self.mtlvae.decoder_net.get_weights()[0]
        beta=emb[0:self.p,:]
        loadings=emb[self.p:,:]
        
        cov=np.dot(loadings.T,loadings)
        corr=cov2corr(cov)
        
        sns.clustermap(pd.DataFrame(beta.T,columns=xlabels,index=ylabels),cmap='seismic_r',col_cluster=False).fig.suptitle('Abiotic response')
        
        sns.clustermap(pd.DataFrame(corr,columns=ylabels,index=ylabels),cmap='seismic_r').fig.suptitle('Residual correlation')
        
        
        return {'beta':beta,'loadings':loadings,'cov':cov,'corr':corr}        
        
        

class MTLVAE(object):
    def __init__(self,name):
        '''
        Parameters
        ----------
        name : String
            Model name.
        '''
        
        self.model_name=name
        self.fitted=False
        self.compiled=False
    
    def create_architecture(self,in_feat,in_config,fe_config,encoder_config,decoder_config,out_config,diag=False):
        '''
        Env network
        '''
        ###Input features ###
        l_in_E,l_E=custom_input(in_feat,concat_feat=True,model_name='in_fe')
        
        ###Feature extraction network
        fe_net=fc_nn(name='fe',archi=fe_config['archi'],reg=fe_config['reg'])
        X=fe_net(l_E)
        
        self.abio_fe=tfk.Model(l_in_E,X)
        
        '''
        VAE
        '''
        ###Setting up the dimensions
        encoded_size=encoder_config['archi'].get('nn')[-1]
        
        if diag:
            paramsize=tfpl.MultivariateNormalDiag.params_size(encoded_size)
        else:
            paramsize=tfpl.MultivariateNormalTriL.params_size(encoded_size)
        
        ### Inputs
        l_in,l_feat=custom_input(in_config,concat_feat=True,model_name='in_vae')
        
        ### Updating the encoder output's dimension 
        encoder_config['archi'].get('nn')[-1]=paramsize
        
        ### Recognition
        self.encoder_net=fc_nn(name='encoder',archi=encoder_config['archi'],reg=encoder_config['reg'])
        latent_params=self.encoder_net(l_feat)
        
        ### Prior over latent variables
        prior=tfd.Independent(tfd.Normal(loc=tf.zeros(encoded_size), scale=1),reinterpreted_batch_ndims=1)
        
        ### Generate code with regularization
        latent_code=code_generator(encoded_size,prior,latent_params,diag)
        
        ### Decoder network
        self.decoder_net=mtl_output(decoder_config,out_config,model_name='decoder')
        
        vae_out=self.decoder_net(tfkl.Concatenate()([X,latent_code]))
        
        ### Multi-task VAE model
        self.mt_vae=tfk.Model(l_in_E+l_in,vae_out)
        
        ### Multi-task predictive model 
        #pred_out=self.decoder_net(tfkl.Concatenate()([X,prior]))
        #self.mt_pred=tfk.Model(l_in_E,pred_out)