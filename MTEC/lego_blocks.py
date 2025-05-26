# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:22:15 2020

@author: saras

This script contains:
    A set of useful parameterizable components to plug and play in your neural architecture.
    
Each function has the following parameters:
    - Layers names - needs to be specified
    - Layers dimensions - needs to be specified
    - Regularization - can use default (no regularization)
    - Initialization - can use default (uniform glorot)
    
"""

"""
Dependencies: tensorflow 2.1, Keras (tf.keras), Tensorflow probability
"""
import os

import tensorflow as tf
import tensorflow_probability as tfp


tfm=tf.math

tfpl = tfp.layers
tfd = tfp.distributions
tfb=tfp.bijectors

tfk = tf.keras
tfkl = tf.keras.layers
tfkv= tfk.utils
tfkr=tfk.regularizers
tfki=tfk.initializers
tfkm=tf.keras.metrics

"""
Utility functions
"""
def get_regularizer(regtype,regparams):
    if(regtype=="l1"):
        kr=tfkr.l1(l=regparams[0])
    elif(regtype=="l2"):
        kr=tfkr.l2(l=regparams[0])
    elif(regtype=="l1_l2"):
        kr=tfkr.l1_l2(l1=regparams[0],l2=regparams[1])
    else:
        kr=None 
    return kr


def probit(x):
    normal = tfd.Normal(loc=0.,
                    scale=1.)
    return normal.cdf(x)

#### Activation functions selector ####
act_fn={'normal':tfk.activations.linear,
        'poisson':tfk.activations.exponential,
        'binomial':tfk.activations.sigmoid,
        'binomial2':probit,
        'categorical':tfk.activations.softmax,
        'negbin':tfk.activations.exponential
        }

"""
Basic building blocks
- Heterogeneous inputs
- Fully connected architecture
- Heterogeneous outputs
"""

'''
Heterogeneous inputs
'''

def custom_input(in_config,concat_feat=True,model_name="model"):
    '''
    Collect inputs, embed categorical inputs and concatenate all to get features
    
    '''
    l_in=[]
    l_feat=[]
    
    for v in in_config:
        k=v[0]
        if k=='num':
            in_num=tfk.Input(shape=(v[1].get('dim'),),dtype=tf.float32,name=v[1].get('id'))
            l_in.append(in_num)
            l_feat.append(in_num)
        
        if k=='cat':
            for cv in v[1]:
                in_cat=tfk.Input(shape=(1,),dtype=tf.int32,name=cv.get('id'))
                emb_cat=tfkl.Flatten()(
                    tfkl.Embedding(cv.get('mod'),cv.get('emb'),name=cv.get('id')+"_emb",embeddings_regularizer=tfk.regularizers.l2(0.001))(in_cat))
                
                l_in.append(in_cat)
                l_feat.append(emb_cat)
                
    
    if (len(l_feat)>1):
        if concat_feat:
            in_feat=tfkl.Concatenate(name=model_name+"_featVector")(l_feat)
        else:
            in_feat=l_feat
        
    elif len(l_feat)==1:
        in_feat=l_feat[0]
        
    else:
        raise BaseException('ModelSpecificationError: check input data configuration')
        
    
    return l_in, in_feat 


'''
Fully connected architecture with single input, single output
'''
def fc_nn(name,archi,reg=None):
    '''
    Syntax
    ********
    name="model_name"
    archi={'nbnum':1,'nl':2,'nn':[10,23]}
    reg={'regtype':'l1','regparam':(0.01,0.01),'dropout':[1,0.8]}
    '''  
    
    in_num=tfk.Input(shape=(archi.get('nbnum'),),name=name+"_input",dtype=tf.float32)
    if reg is None:
        reg={'regtype':None,'regparam':None}
        
    if type(reg)==tuple:
        reg={'regtype':'l1_l2','regparam':reg}
        
    kreg=get_regularizer(reg.get("regtype"),reg.get("regparam"))

    prev=in_num
    
    activs=archi.get("activation")
    if type(activs)!=list:
        activs=[activs]*archi.get("nl") ##same activation everywhere
        Warning('Using the same activation for all layers.')
    
    ### Whether to dropout inputs: useful in case of images ###    
    if reg.get('dropout') is not None:
            rate=reg.get("dropout")[0]
            if rate<1:
                prev=tfkl.Dropout(rate)(prev)
    
    for i in range(archi.get("nl")):
        prev=tfkl.Dense(archi.get("nn")[i], use_bias=archi.get("fit_bias"),activation=activs[i],name=name+"_"+str(i),kernel_regularizer=kreg)(prev)
        if reg.get('dropout') is not None:
            rate=reg.get("dropout")[i+1]
            if rate<1:
                prev=tfkl.Dropout(rate)(prev)
    
    #out=tfkl.Activation(activation=archi.get("o_activation") if archi.get("activation") is not None else None)
    m=tfk.Model(in_num,prev,name=name)
    
    return(m)
 

'''
Heterogeneous outputs
'''
def mtl_output(shared_config,out_configs,model_name="mtl",path='',plot=False,concat=False):
    
    ### Shared feature transformation component
    shared=fc_nn(name=shared_config['name'],archi=shared_config['archi'],reg=shared_config['reg'])
    l_outputs=[]
    
    if plot:
        os.makedirs(path+model_name, exist_ok=True)
        tfkv.plot_model(shared,path+model_name+'/shared.png',show_shapes=True)
    
    ### Task-specific components
    for tc in out_configs:  ##out_config is a list of configs for each output {'type':'cat','specific':{'archi','reg'}}
        tnm=tc['name']
        specific_config=tc['specific']
        tc_fc=fc_nn(tnm,specific_config['archi'],reg=specific_config['reg'])
        
        if plot:
            tfkv.plot_model(tc_fc,path+model_name+'/%s.png'%tnm,show_shapes=True)
        
        act=tc['activation']
        if act is None: ##specify default value
            if tc['type']=='binary':
                act='sigmoid'
                act_=act
                
            elif tc['type']=='categorical':
                act='softmax'
                act_=act
                
            else:
                act='linear'
                act_=act
        
        if act=='probit':
            act=probit
            act_='probit'
            
        else:
            act_=act
            
        t_pred=tfkl.Activation(activation=act,name=tnm+"_"+act_)(tc_fc(shared.output))
        
        l_outputs.append(t_pred)
        
    
    if concat:
        if len(l_outputs)>1:
            out=tfkl.Concatenate()(l_outputs)
        
        else:
            out=l_outputs[0]
    
    else:
        out=l_outputs
    
    return tfk.Model(shared.inputs,out,name=model_name)
   

"""
Composition of architectures

1. Autoencoders (generic case with multiple types of inputs/outputs)
2. Variational autoencoders
3. Conditional variational autoencoders
4. Multi-task hard sharing (without the encoder part)
5. Multi-task soft sharing (without the encoder part but soft sharing)
"""

def autoencoder(in_config,encoder_config,decoder_config,out_config):
    l_in,l_feat=custom_input(in_config,concat_feat=True,model_name='in_autoenc')
    encoder_net=fc_nn(name='encoder',archi=encoder_config['archi'],reg=encoder_config['reg'])
    decoder_net=mtl_output(decoder_config,out_config,model_name='decoder')

    autoencoder=tfk.Model(l_in,decoder_net(encoder_net(l_feat)))
    
    return encoder_net, decoder_net, autoencoder



def code_generator(encoded_size,prior,latent_params,beta=1.0):  
    
    out_code=tfpl.MultivariateNormalTriL(
        encoded_size,
        activity_regularizer=tfpl.KLDivergenceRegularizer(prior,weight=beta))(latent_params)
    return(out_code)



def variational_autoencoder(in_config,encoder_config,decoder_config,out_config,prior=None,beta=1.0):
    ### Inputs
    l_in,l_feat=custom_input(in_config,concat_feat=True,model_name='in_vae')
    
    ### Getting the dimensions right
    encoded_size=encoder_config['archi'].get('nn')[-1]
    paramsize=tfpl.MultivariateNormalTriL.params_size(encoded_size)
    
    ### Updating the encoder output's parameter 
    encoder_config['archi'].get('nn')[-1]=paramsize
    
    ### Recognition
    encoder_net=fc_nn(name='encoder',archi=encoder_config['archi'],reg=encoder_config['reg'])
    latent_params=encoder_net(l_feat)
    
    if prior is None:
        ### Prior
        prior = tfd.Independent(tfd.Normal(loc=tf.zeros(encoded_size), scale=1),
                        reinterpreted_batch_ndims=1)
    
    ### Generate code
    latent_code=code_generator(encoded_size,prior,latent_params,beta)
    
    ### Decoder network
    decoder_net=mtl_output(decoder_config,out_config,model_name='decoder')
    l_out=decoder_net(latent_code)
    
    var_autoencoder=tfk.Model(l_in,l_out)
    
    return encoder_net, decoder_net, prior, var_autoencoder


def conditional_variational_autoencoder(in_feat,in_config,fe_config,encoder_config,decoder_config,out_config):
    '''
    Prior network
    '''
    
    ###Input features ###
    l_in_E,l_E=custom_input(in_feat,concat_feat=True,model_name='in_fe')
    
    ###Setting up the dimensions
    
    encoded_size=encoder_config['archi'].get('nn')[-1]
    paramsize=tfpl.MultivariateNormalTriL.params_size(encoded_size)
    
    ### Updating the encoder output's parameter 
    fe_config['archi'].get('nn')[-1]=paramsize
    
    ###Feature extraction network
    fe_net=fc_nn(name='fe',archi=fe_config['archi'],reg=fe_config['reg'])
    X=fe_net(l_E)
    
    prior=tfpl.MultivariateNormalTriL(encoded_size)(X)
    
    '''
    VAE
    '''
    ### Inputs
    l_in,l_feat=custom_input(in_config,concat_feat=True,model_name='in_vae')
    
    
    ### Updating the encoder output's parameter 
    encoder_config['archi'].get('nn')[-1]=paramsize
    
    ### Recognition
    encoder_net=fc_nn(name='encoder',archi=encoder_config['archi'],reg=encoder_config['reg'])
    latent_params=encoder_net(l_feat)
    
    ### Generate code with regularization
    latent_code=code_generator(encoded_size,prior,latent_params)
    
    ### Decoder network
    decoder_net=mtl_output(decoder_config,out_config,model_name='decoder')
    
    in_decod=tfkl.Concatenate()([l_E,latent_code])
    l_out=decoder_net(in_decod)
    
    cvar_autoencoder=tfk.Model(l_in+l_in_E,l_out)
    
    return encoder_net, decoder_net, tfk.Model(l_in_E,prior), cvar_autoencoder


class GlobalParam(tfkl.Layer):

    def __init__(self, output_dim,vname, **kwargs):
       self.output_dim = output_dim
       self.vname=vname
       super(GlobalParam, self).__init__(**kwargs)

    def build(self, input_shapes):
       self.kernel = tf.Variable(tf.random_uniform_initializer(minval=-1,maxval=1)(shape=[1,self.output_dim],dtype=tf.float32),
                                 name=self.vname, 
                                 trainable=True)
       
       super(GlobalParam, self).build(input_shapes)  

    def call(self, inputs):
       return inputs


class BiasLayer(tfkl.Layer):
    def __init__(self, *args, **kwargs):
        super(BiasLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight('bias',
                                    shape=input_shape[1:],
                                    initializer='uniform',
                                    trainable=True)
    def call(self, x):
        return x + self.bias    