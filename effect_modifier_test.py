import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy
from scipy.sparse import csr_matrix
sns.set(font_scale=1.5)
import warnings
warnings.filterwarnings("ignore")

import sklearn.linear_model as lm
import sklearn.tree as tree
import sklearn.ensemble as ensemble
import sklearn.svm as svm
import sklearn.gaussian_process as gp
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

import networkx as nx
from pyvis.network import Network

def summarize(df):
    df2 = df.rename(columns={c:'neighbor_'+c for c in df.columns})
    sr = df2.mean(axis=0)
    return sr

from tqdm import tqdm
def contains(G,H,j):
    ## is H contained in G?
    A = G
    GM = nx.algorithms.isomorphism.GraphMatcher(A,H)
    is_isomorphic = int(GM.subgraph_is_isomorphic())
    return is_isomorphic

#     if is_isomorphic:
#         GM_1 = nx.algorithms.isomorphism.ISMAGS(H,A)
#         iso_lst = list(GM_1.largest_common_subgraph())
#         for iso in iso_lst:
#             if (j in iso) and (len(iso)==len(H.nodes())):
# #                 print(iso[j])
#                 if (iso[j] == list(A.nodes())[0]):
#                     return 1
#     return 0

def get_first_order_egocentric_graph(G,i):
    Ai = G.loc[i]
#     print(np.where(Ai==1)[0])
    idx = [i]+list(Ai.loc[Ai>0].index)
    A_ego = G.loc[idx][idx]
    return A_ego

def get_x_u(V,X):
    return V[X]

def get_delta_u(G,Delta,Delta_col,idxs):
    df = pd.DataFrame()
    if len(Delta)==0:
        for i in tqdm(idxs):
            u_i = pd.DataFrame(np.array([]).reshape(1,-1),index=[i],columns=Delta_col)
            df = df.append(u_i)
        return df
    
    for i in tqdm(idxs):
        Gi = get_first_order_egocentric_graph(G,i)
        Gi1 = Gi.to_numpy()
        u_i = pd.DataFrame(np.array([ contains(nx.from_numpy_array(Gi1),delta[0],delta[1]) for delta in Delta]).reshape(1,-1),index=[i],columns=Delta_col)
        df = df.append(u_i)
    return df

def get_u(G,V,X,Delta,Delta_col):
    df_delta = get_delta_u(G,Delta,Delta_col)
    df_x = get_x_u(V,X)
    df = df_x.join(df_delta)
    return df

# def fit(U,tau):
#     X_train, X_test, y_train, y_test = train_test_split(U, tau, test_size=0.5, random_state=42)
#     model = tree.DecisionTreeClassifier().fit(X_train,y_train)
#     return model#, model.predict_proba(U,tau)

def set_data(outcome, treatment, df_unit, df_social_net, hypothesis):
    # adjacency list to adjacency matrix
    social = pd.DataFrame(np.zeros((df_unit.shape[0],df_unit.shape[0])),index=df_unit.index,columns=df_unit.index)
    for idx in tqdm(df_social_net.index):
        adj_list = set(df_social_net.loc[idx])
        for n in adj_list:
            try:
                if n!='0':
                    social.loc[idx,n] = social.loc[idx,n]+1
                    social.loc[n,idx] = social.loc[n,idx]+1
            except:
                continue
    social_b = (social>0).astype(int)
    
    # summarizing neighbors covariates 
    data_w_neighbor_cov = pd.DataFrame()
    idxs = set(df_unit.index)
    for idx in tqdm(idxs):
        n_idxs = np.unique(df_social_net.loc[idx])
        n_idxs = set(np.delete(n_idxs, np.where(n_idxs == '0')))
        n_idxs = n_idxs.intersection(idxs)
        sr = summarize(df_unit.loc[n_idxs])
        df2 = pd.DataFrame(df_unit.loc[idx].append(sr),columns=[idx]).T
        data_w_neighbor_cov = data_w_neighbor_cov.append(df2)
    
    #adding network pattern to data
    idxs = data_w_neighbor_cov.dropna().index
    sample_data = data_w_neighbor_cov.loc[idxs]

    Delta = hypothesis['Delta'] 
    Delta_col = hypothesis['Delta_col'] 

    df_delta = get_delta_u(social_b,Delta,Delta_col,idxs=idxs)
    sample_w_netstruct = sample_data.join(df_delta,how='inner')
    
    #estimating CDE and ADE
    Y = sample_w_netstruct[outcome].fillna(0)
    T = sample_w_netstruct[treatment].fillna(0)
    X = sample_w_netstruct.fillna(0)
    return X,Y,T

def get_cde(X,Y,T):
    model_c = lm.LinearRegression().fit(X.loc[T==0.0],Y.loc[T==0.0])
    model_t = lm.LinearRegression().fit(X.loc[T==1.0],Y.loc[T==1.0])

    CDE = (model_t.predict(X)[:] - model_c.predict(X)[:])
    df_CDE = pd.DataFrame(CDE,index=X.index,columns=['TE'])
    ADE = np.mean(CDE)
    print(ADE)
    print((ADE-2*np.std(CDE),ADE+2*np.std(CDE)))
    
    # fig = plt.figure(figsize=(16,8))
    # sns.distplot(CDE)
    # plt.axvline(np.mean(CDE),c='black')
    # plt.axvline(np.mean(CDE)-2*np.std(CDE),ls='--',c='black')
    # plt.axvline(np.mean(CDE)+2*np.std(CDE),ls='--',c='black')
    # plt.xlabel('Conditional Direct Effect (CDE) \n $CDE(x) = E(Y^{(1)}|x,n,g) - E(Y^{(0)}|x,n,g)$')
    # plt.ylabel('Estimated Probability Density of CDE')
    # plt.savefig('CDE.png')
    
    return X,Y,T,CDE,ADE
    
def test_heterogeneity(X,Y,T,CDE,ADE,hypothesis,discrete={},predictor='AdaBoost'):

    # Posterior Projecting CDE on Hypothesis Space
    estimator_f = {}
    Q = {}
    H = X[list(hypothesis['X_col']) + list(hypothesis['Delta_col'])]
    
    for col in tqdm(H.columns):      
        x_range = np.min(H[[col]]),np.max(H[[col]])
        if col in discrete:
            X1 = np.sort(H[[col]].to_numpy(),axis=0)#np.arange(int(x_range[0]),int(x_range[1]+1)).reshape(-1,1)
        else:
            X1 = np.sort(H[[col]].to_numpy(),axis=0)#np.linspace(x_range[0],x_range[1],num=samples).reshape(-1,1)
            
        if predictor=='Ridge':
            model = lm.RidgeCV().fit(H[[col]],CDE)
            se = np.sqrt(np.mean(np.square(CDE - model.predict(H[[col]]))))

            y_hat_mean = model.predict(X1)
            y_hat_std = np.ones_like(y_hat_mean)*se

            estimator_f[col] = [model]
        
        if predictor=='SVM':
            model = svm.SVR().fit(H[[col]],CDE)
            se = np.sqrt(np.mean(np.square(CDE - model.predict(H[[col]]))))
            print(model.score(H[[col]],CDE))

            y_hat_mean = model.predict(X1)
            y_hat_std = np.ones_like(y_hat_mean)*se

            estimator_f[col] = [model]
            
        if predictor=='AdaBoost':
            additive_model = ensemble.AdaBoostRegressor().fit(H[[col]],CDE)
            estimators = additive_model.estimators_
            y_hat_array = []
            for i in range(len(estimators)):
                estimator = estimators[i]
                y_hat = estimator.predict(X1)
                y_hat_array.append(y_hat)
                if col not in estimator_f:
                    estimator_f[col] = []
                estimator_f[col] = estimator_f[col] + [estimator]
            y_hat_array = np.array(y_hat_array)
            y_hat_mean = np.mean(y_hat_array,axis=0)
            y_hat_std = np.std(y_hat_array,axis=0)
        
        if predictor=='GradientBoosting':
            alpha = 0.95
            upper = ensemble.GradientBoostingRegressor(loss='quantile', alpha=alpha,n_estimators=1000,max_leaf_nodes=16).fit(H[[col]],CDE)
            lower = ensemble.GradientBoostingRegressor(loss='quantile', alpha=1.0-alpha,n_estimators=1000,max_leaf_nodes=16).fit(H[[col]],CDE)
            mean_model = svm.SVR().fit(H[[col]],CDE) #ensemble.RandomForestRegressor(n_estimators=100).fit(H[[col]],CDE)
            y_hat_mean = mean_model.predict(X1)
            y_hat_std = np.abs(upper.predict(X1) - lower.predict(X1))/4.0
        
        if predictor == 'BayesianRidge':
            model = lm.BayesianRidge().fit(H[[col]],CDE)
            y_hat_mean, y_hat_std = model.predict(X1, return_std=True)
            estimator_f[col] = [model]
            
        if predictor == 'GaussianProcess':
            kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-10, np.std(CDE)))
            model = gp.GaussianProcessRegressor(kernel=kernel, normalize_y=True).fit(H[[col]],CDE)
            y_hat_mean, y_hat_std = model.predict(X1, return_std=True)
            estimator_f[col] = [model]


        Q[col] = np.mean(((y_hat_mean - ADE)**2)/y_hat_std)
    
#         fig = plt.figure(figsize=(5,5))
#         if col in discrete:
#             if predictor == 'AdaBoost':
#                 for j_col in range(X1.shape[0]):
#                     plt.boxplot(y_hat_array[:,j_col],positions=[j_col])
# #             sns.boxplot(x=col,y='CDE',data=H,boxprops=dict(alpha=.15))
#             else:
#                 plt.errorbar(X1,y_hat_mean,yerr=y_hat_std,elinewidth=30,fmt='.k',capsize=0,alpha=0.3,ecolor='gray')
#                 plt.errorbar(X1,y_hat_mean,yerr=2*y_hat_std,elinewidth=2,fmt='.k',capsize=12,ecolor='black')
# #             plt.scatter(X1,y_hat_mean)
#         else:
#             sns.scatterplot(x=col,y='CDE',data=H,alpha=0.1)
#             plt.plot(X1,y_hat_mean,linewidth=3,c='black')
#             plt.fill_between(X1[:,0], y1=y_hat_mean+2*y_hat_std, y2=y_hat_mean-2*y_hat_std,alpha=0.25)
        # plt.xlabel(str(col))
        # plt.ylabel('E[CDE|%s]'%(col))
        # plt.axhline(0,c='red')
        # fig.savefig('adaBoost_f_%s.png'%(col))
    H['CDE'] = CDE
    df_CDE = H[['CDE']]
    return df_CDE,ADE,estimator_f,Q,H