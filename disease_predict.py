import pandas as pd 
import numpy as np
from scipy.sparse import csr_matrix
from soyclustering import SphericalKMeans
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from sklearn.feature_extraction.text import TfidfTransformer  

# 计算apriori中置信度的函数
def calculate_confidence(x,df):
    if x.itemnums <= 1: return 0
    else:
        tmp_sub_df = df.copy()
        for item in list(x.itemsets)[:-1]:
            tmp_sub_df = tmp_sub_df[tmp_sub_df[item]==1]
        support_FD = len(tmp_sub_df)

        tmp_sub_df = tmp_sub_df[tmp_sub_df[list(x.itemsets)[-1]]==1]
        support_FD_n_R = len(tmp_sub_df)

        return support_FD_n_R/support_FD

# tf-idf预测中计算cosine相似性
def get_similarity(x,y):
    x = np.array(x)
    y = np.array(y)
    sum_xy = np.sum(x*y)
    sum_x = np.sqrt(np.sum(x**2))
    sum_y = np.sqrt(np.sum(y**2))
    return sum_xy/(sum_x*sum_y)

# 根据个人所在的类，找出个人与这个类中最相似的病人的索引
def get_most_similar(sample_data,weighted_cluster):
    sim_mat = np.zeros(len(weighted_cluster))
    for i in range(0,len(weighted_cluster)):
        x = weighted_cluster.iloc[i,:]
        similarity = get_similarity(x,sample_data)
        sim_mat[i] = similarity
    return np.argmin(sim_mat)


class disease_pred:
    def __init__(self,n_clusters,max_iter,verbose,init,sparsity,minimum_df_factor,confidence_thres):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.verbose = verbose
        self.init = init
        self.sparsity = sparsity
        self.minimum_df_factor = minimum_df_factor
        self.confidence_thres = confidence_thres
        
    def get_apriori_rules(self,cluster_df_dict):
        apriori_rules_dict = dict()
        for cluster in cluster_df_dict.keys():
            cluster_df = cluster_df_dict[cluster]
            
            tmp_ = apriori(cluster_df.iloc[:,3:],min_support = 2/cluster_df.iloc[:,3:].sum().max(),use_colnames=True)
            tmp_['itemnums'] = tmp_.itemsets.apply(lambda x:len(x))
            if len(tmp_) == 0:
                apriori_rules_dict[cluster] = pd.DataFrame(columns = ['support', 'itemsets', 'itemnums', 'confidence', 'input', 'pred'])
                continue
            tmp_['confidence'] = tmp_.apply(lambda x:calculate_confidence(x,cluster_df),axis = 1)
            tmp_ = tmp_[tmp_.confidence >self.confidence_thres]
            tmp_ = tmp_[tmp_.itemnums>1]
            tmp_['input'] = tmp_['itemsets'].apply(lambda x:list(x)[:-1])
            tmp_['pred'] = tmp_['itemsets'].apply(lambda x:list(x)[-1])
            apriori_rules_dict[cluster] = tmp_
        return apriori_rules_dict
    
    def fit(self,X):
        ## 聚类
        print('Spherical Kmeans Clustering Proceeding ...')
        spherical_kmeans = SphericalKMeans(
                                        n_clusters=self.n_clusters,
                                        max_iter=self.max_iter,
                                        verbose=self.verbose,
                                        init=self.init,
                                        sparsity=self.sparsity,
                                        minimum_df_factor=self.minimum_df_factor
                                        )
        sp_matrix = csr_matrix(X.values)
        labels = spherical_kmeans.fit_predict(sp_matrix)
        spherical_kmeans.fit(sp_matrix)
        new_X = X.reset_index(drop = True)
        cluster_dfs = dict()
        for i in range(25):
            cluster_dfs['cluster'+str(i)] = new_X.loc[labels == i]
        print('Spherical Kmeans Clustering DONE!!!')
        
        ## 关联分析
        print('Apriori Proceeding ...')
        apriori_rules_dict = self.get_apriori_rules(cluster_dfs)
        print('Apriori DONE!!!')
        ## TF-IDF权重
        print('TF-IDF weights Proceeding ...')
        freq_of_clusters = []
        for cluster in cluster_dfs.keys():
            freq_i = list(cluster_dfs[cluster].iloc[:,3:].sum())
            freq_of_clusters.append(freq_i)
        
        transformer = TfidfTransformer()  
        tfidf = transformer.fit_transform(freq_of_clusters)  
        print('TF-IDF DONE!!!')
        self.kmeans_model = spherical_kmeans
        self.apriori_rules_dict = apriori_rules_dict
        self.cluster_dfs = cluster_dfs
        self.tfidf = tfidf
        
    def predict(self,X):
        sparse_X = csr_matrix(X.values)
        labels = np.argmin(self.kmeans_model.transform(sparse_X),axis = 1)
        
        res = []
        ill_list = []
        for i in range(len(X)):
            belonged_cluster = labels[i]
            cluster_index = 'cluster'+str(belonged_cluster)
            sample_data = X.iloc[i,3:]
            # 根据tf-idf找出最相似的病人，将其患有的疾病当前病人没有的疾病作为预测
            weighted_sample_data = self.tfidf.toarray()[belonged_cluster] * sample_data
            weighted_cluster_data = self.tfidf.toarray()[belonged_cluster] * self.cluster_dfs[cluster_index].iloc[:,3:]
            most_similar_index = get_most_similar(weighted_sample_data,weighted_cluster_data)
            most_similar_data = self.cluster_dfs[cluster_index].iloc[most_similar_index,3:]
            most_similar_ill_list = list(most_similar_data[most_similar_data == 1].index)
            sample_ill_list = list(sample_data[sample_data == 1].index)
            tfidf_pred = list(set(most_similar_ill_list) - set(sample_ill_list))
            # 根据当前病人所患疾病查找apriori规则，有符合的频繁规则的话将规则后项作为预测
            apriori_pred = []
            apriori_rules = self.apriori_rules_dict[cluster_index]
            apriori_rules = apriori_rules[['input','pred']]
            if sample_ill_list in list(apriori_rules.input.values):
                index = list(apriori_rules.input.values).index(sample_ill_list)
                apriori_pred.append([apriori_rules.iloc[index,1]])
            
            total_pred = tfidf_pred+apriori_pred
            res.append(total_pred)
            ill_list.append(sample_ill_list)
        return pd.DataFrame({'AAC001':X.index,'input':ill_list,'pred':res})
    
    