from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import metrics
from sklearn.metrics import davies_bouldin_score
import umap.umap_ as umap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
from functools import reduce


##############################################################
# Data Cleaning
##############################################################
def clean_data(df, role = "None", patch = "All", stratified_sampling = False):
    games_df = df
    if patch != "All":
        games_df = games_df[ games_df['patch'] == patch]
    else:
        games_df = games_df
        
    if role != "None":
        games_df = games_df[ games_df['teamPosition'] == role ]
    else:
        games_df = games_df
    # list of champions with more than 100 games played
    top_champs = [i for i, x in games_df.championName.value_counts().to_dict().items() if x > 100]
    games_df = games_df[games_df['championName'].isin(top_champs)]
    if stratified_sampling:
        games_df = games_df.groupby(by='championName').apply(lambda x: x.sample(n=100)).reset_index(level=1, drop=True).drop(['championName'], axis=1).reset_index()
    try:
        games_df = games_df.drop(['teamPosition'], axis=1)
        games_df = games_df.drop(['patch'], axis=1)
        games_df = games_df.drop(['Unnamed: 0'], axis=1)
    except Exception as e:
        print(e)
    return games_df


##############################################################
# Clustering
##############################################################
def group_by_champions(df):
    df_champs = df.drop(['championId'], axis=1).groupby("championName").mean().reset_index(level=0)
    x = df_champs.iloc[:,1:]
    y = df_champs.iloc[:,:1]
    return x, y

def standarize_df(df):
    x_role, y_role = group_by_champions(df)
    ## standarize
    role_std_model = StandardScaler()
    x_role_std = role_std_model.fit_transform(x_role)
    
    return x_role_std, y_role

def kmeans_clustering_elbow(df, total_k = 20):
    distorsions = []
    K = range(1, total_k)
    for k in K:
        kmean_model = KMeans(n_clusters=k)
        kmean_model.fit(df)
        distorsions.append(kmean_model.inertia_)
        
    plt.figure(figsize=(16,8))
    plt.plot(K, distorsions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()
    
def apply_pca(df, variance_explained_specified):
    pca = PCA( variance_explained_specified )
    df = pca.fit_transform(df)
    return df

def apply_umap(df, n_components):
    reducer = umap.UMAP(n_components= n_components)
    df = reducer.fit_transform(df)
    return df

def apply_kmeans(df, k=2):
    kmeans = KMeans(n_clusters = k)
    cluster_labels = kmeans.fit_predict(df)
    
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(df, cluster_labels)
    calinski_harabasz = metrics.calinski_harabasz_score(df, labels)
    davies_bouldin = davies_bouldin_score(df, labels)
    
    return cluster_labels, silhouette_avg, calinski_harabasz, davies_bouldin

def apply_optics(df, min_samples=3):
    optics = OPTICS(min_samples=min_samples)
    cluster_labels = optics.fit_predict(df)
    
    labels = optics.labels_
    silhouette_avg = silhouette_score(df, cluster_labels)
    calinski_harabasz = metrics.calinski_harabasz_score(df, labels)
    davies_bouldin = davies_bouldin_score(df, labels)
    
    return cluster_labels, silhouette_avg, calinski_harabasz, davies_bouldin

def pca_kmeans(x_role, y_role, variance_explained_specified=0.85, k = 2 ):
    ## pca
    pca = PCA( variance_explained_specified )
    role_principal_components = pca.fit_transform(x_role)
    ## k-means
    # kmeans_clustering_elbow(role_principal_components, total_k = 20)
    role_kmeans_model = KMeans(n_clusters= k ).fit(role_principal_components)
    y_role['group'] = role_kmeans_model.predict(role_principal_components)
    y_role
    role_champions_list = y_role.groupby('group')['championName'].apply(list).to_dict()
    
    return y_role, role_champions_list, role_principal_components

def umap_kmeans(x_role, y_role, n_comps= 2 , k = 2 ):
    ## umap
    reducer = umap.UMAP(n_components= n_comps)
    role_umap = reducer.fit_transform(x_role)
    ## k-means
    # kmeans_clustering_elbow(role_umap, total_k = 20)
    role_kmeans_model = KMeans(n_clusters= k ).fit(role_umap)
    y_role['group'] = role_kmeans_model.predict(role_umap)
    role_champions_list = y_role.groupby('group')['championName'].apply(list).to_dict()
    
    return y_role, role_champions_list, role_umap

def umap_optics(x_role, y_role, n_comps= 2 , min_samples = 2 ):
    ## umap
    reducer = umap.UMAP(n_components= n_comps)
    role_umap = reducer.fit_transform(x_role)
    ## optics
    role_optics_model = OPTICS(min_samples=min_samples)
    y_role['group'] = role_optics_model.fit_predict(role_umap)
    role_champions_list = y_role.groupby('group')['championName'].apply(list).to_dict()
    
    return y_role, role_champions_list, role_umap


def get_best_clustering(x_general, pca_params, umap_params, kmeans_params, optics_params):
    results = {"pca": {"kmeans": [], "optics": []}, "umap": {"kmeans": [], "optics": []}}
    for pca_param in pca_params:
        x_general_pca = apply_pca(x_general, pca_param)
        for kmeans_param in kmeans_params:
            try:
                cluster_labels, silhouette_avg, calinski_harabasz, davies_bouldin = apply_kmeans(x_general_pca, k=kmeans_param)
                results['pca']['kmeans'].append({
                    "dimentionality reduction": "pca",
                    "clustering": "kmeans",
                    "dimentionality reduction param": pca_param,
                    "clustering param": kmeans_param,
                    "silhouette_avg": silhouette_avg,
                    "calinski_harabasz": calinski_harabasz,
                    "davies_bouldin": davies_bouldin,
                })
            except Exception as e:
                print(e)
        for optics_param in optics_params:
            try:
                cluster_labels, silhouette_avg, calinski_harabasz, davies_bouldin = apply_optics(x_general_pca, min_samples=optics_param)
                results['pca']['optics'].append({
                    "dimentionality reduction": "pca",
                    "clustering": "optics",
                    "dimentionality reduction param": pca_param,
                    "clustering param": optics_param,
                    "silhouette_avg": silhouette_avg,
                    "calinski_harabasz": calinski_harabasz,
                    "davies_bouldin": davies_bouldin,
                })
            except Exception as e:
                print(e)
    for umap_param in umap_params:
        x_general_umap = apply_umap(x_general, umap_param)
        for kmeans_param in kmeans_params:
            try:
                cluster_labels, silhouette_avg, calinski_harabasz, davies_bouldin = apply_kmeans(x_general_umap, k=kmeans_param)
                results['umap']['kmeans'].append({
                    "dimentionality reduction": "umap",
                    "clustering": "kmeans",
                    "dimentionality reduction param": umap_param,
                    "clustering param": kmeans_param,
                    "silhouette_avg": silhouette_avg,
                    "calinski_harabasz": calinski_harabasz,
                    "davies_bouldin": davies_bouldin,
                })
            except Exception as e:
                print(e)
        for optics_param in optics_params:
            try:
                cluster_labels, silhouette_avg, calinski_harabasz, davies_bouldin = apply_optics(x_general_umap, min_samples=optics_param)
                results['umap']['optics'].append({
                    "dimentionality reduction": "umap",
                    "clustering": "optics",
                    "dimentionality reduction param": umap_param,
                    "clustering param": optics_param,
                    "silhouette_avg": silhouette_avg,
                    "calinski_harabasz": calinski_harabasz,
                    "davies_bouldin": davies_bouldin,
                })
            except Exception as e:
                print(e)


    pca_kmeans_dict = pd.DataFrame.from_dict(results['pca']['kmeans'])
    pca_optics_dict = pd.DataFrame.from_dict(results['pca']['optics'])
    umap_kmeans_dict = pd.DataFrame.from_dict(results['umap']['kmeans'])
    umap_optics_dict = pd.DataFrame.from_dict(results['umap']['optics'])

    results_dict = pd.concat([pca_kmeans_dict, pca_optics_dict, umap_kmeans_dict, umap_optics_dict])
    results_dict = results_dict.sort_values(by=["silhouette_avg", "davies_bouldin"], ascending=False)
    
    return results_dict





##############################################################
# Predictions
##############################################################


def return_champion(row, champions):
    for i, x in champions['championName'].items():
        if x == row['championName']:
            return champions['group'][i]
    return 999


def insert_group_by_result(clean_soloq_games, role_groups, role, group_label, winner):
    role_soloq = clean_soloq_games[clean_soloq_games['teamPosition'] == role]
    if (winner):
        role_soloq = role_soloq[role_soloq['win'] == True]
    else:
        role_soloq = role_soloq[role_soloq['win'] == False]
    role_soloq = role_soloq[role_soloq['championName'].isin(role_groups['championName'].to_list())]
    role_soloq[group_label] = role_soloq.apply(lambda row: return_champion(row, role_groups), axis=1)
    role_soloq = role_soloq[role_soloq[group_label] != 999]
    return role_soloq

def insert_group_by_side(clean_soloq_games, role_groups, role, group_label, side):
    role_soloq = clean_soloq_games[clean_soloq_games['teamPosition'] == role]
    role_soloq = role_soloq[role_soloq['teamId'] == side]
    role_soloq = role_soloq[role_soloq['championName'].isin(role_groups['championName'].to_list())]
    role_soloq[group_label] = role_soloq.apply(lambda row: return_champion(row, role_groups), axis=1)
    role_soloq = role_soloq[role_soloq[group_label] != 999]
    if (side == "Blue"):
        role_soloq['team_win'] = role_soloq.apply(lambda x: "Blue" if x['win'] == True else "Red", axis=1)
    else:
        role_soloq['team_win'] = role_soloq.apply(lambda x: "Red" if x['win'] == True else "Blue", axis=1)
    return role_soloq
    
           
        
        
# competitive

def insert_group(clean_soloq_games, role_groups, role, group_label, winner):
    role_soloq = clean_soloq_games[clean_soloq_games['teamPosition'] == role]
    if (winner):
        role_soloq = role_soloq[role_soloq['win'] == 1]
    else:
        role_soloq = role_soloq[role_soloq['win'] == 0]
    role_soloq = role_soloq[role_soloq['championName'].isin(role_groups['championName'].to_list())]
    role_soloq[group_label] = role_soloq.apply(lambda row: return_champion(row, role_groups), axis=1)
    role_soloq = role_soloq[role_soloq[group_label] != 999]
    return role_soloq

def insert_all_groups(clean_soloq_games, role_groups, role, group_label):
    role_soloq = clean_soloq_games[clean_soloq_games['teamPosition'] == role]
    role_soloq = role_soloq[role_soloq['championName'].isin(role_groups['championName'].to_list())]
    role_soloq[group_label] = role_soloq.apply(lambda row: return_champion(row, role_groups), axis=1)
    role_soloq = role_soloq[role_soloq[group_label] != 999]
    return role_soloq




##############################################################
# Model Validation
##############################################################

def test_model(model, w_top_group, w_jungle_group, w_mid_group, w_adc_group, w_support_group, l_top_group, l_jungle_group, l_mid_group, l_adc_group, l_support_group):
    try:
        test = model.predict([[None, None,None,None,None,None,l_top_group,l_jungle_group,l_mid_group,l_adc_group,l_support_group]])
        result = test[0]
        prediction = 0
        if (result[1] == w_top_group):
            prediction = prediction + 0.2
        if (result[2] == w_jungle_group):
            prediction = prediction + 0.2
        if (result[3] == w_mid_group):
            prediction = prediction + 0.2
        if (result[4] == w_adc_group):
            prediction = prediction + 0.2
        if (result[5] == w_support_group):
            prediction = prediction + 0.2            
            
        return prediction
    
    except Exception as e:
        print(e)
        return 0
    
def test_predictions(w_top_group, w_jungle_group, w_mid_group, w_adc_group, w_support_group, w_top_random, w_jungle_random, w_mid_random, w_adc_random, w_support_random):
    prediction = 0
    try:
        if (w_top_group == w_top_random):
            prediction = prediction + 0.2
        if (w_jungle_group == w_jungle_random):
            prediction = prediction + 0.2
        if (w_mid_group == w_mid_random):
            prediction = prediction + 0.2
        if (w_adc_group == w_adc_random):
            prediction = prediction + 0.2
        if (w_support_group == w_support_random):
            prediction = prediction + 0.2
        return prediction
    except Exception as e:
        print(e)
        return 0
    