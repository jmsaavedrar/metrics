import argparse
import numpy as np
import matplotlib.pyplot as plt

def get_ranking(catalog, queries):
    #normalize to unit
    norm_catalog = np.linalg.norm(catalog, axis = 1, keepdims=True)
    catalog = catalog / norm_catalog
    norm_queries = np.linalg.norm(queries, axis = 1, keepdims=True)
    queries= queries / norm_queries
    cossim = np.matmul(queries, np.transpose(catalog))
    idxs = np.argsort(-cossim, axis = 1)
    return idxs

def compute_p11(ranking, relevants):
    p_11 = np.zeros(11)
    n_queries = ranking.shape[0]
    n_catalog = ranking.shape[1]
    for q in np.arange(n_queries):     
        if n_relevants[q] > 0 :   
            positions = np.arange(n_catalog) + 1        
            relevants_idx_q = relevants[q][ranking[q]];    
            relevants_idx_q_cum = np.cumsum(relevants_idx_q)        
            recall = relevants_idx_q_cum / n_relevants[q] 
            precision = relevants_idx_q_cum / positions;
            p = precision[relevants_idx_q==1]
            for r in np.arange(11):
                p_11[r] += np.max(precision[recall >= r*0.1])
    return p_11/n_queries    

if __name__ == '__main__' :
    # parser = argparse.ArgumentParser(description = 'compute recall precision')
    # parser.add_argument("-catalog", type = str, help = "<str> catalog embeddings", required = True)
    # parser.add_argument("-queries", type = str, help = "<str> query embeddings", required = True)
    # parser.add_argument("-relevants", type = str, help = "<str> relevants", required = True)    
    #
    # args = parser.parse_args()
    catalog_file = '/home/vision/smb-datasets/VETE/recall_precision/pepeganga_catalog_data.npy'
    queries_file = '/home/vision/smb-datasets/VETE/recall_precision/Pepeganga/query_embeddings.npy'
    relevants_file = '/home/vision/smb-datasets/VETE/recall_precision/pepeganga_relevants.npy'
    
    catalog_file_v = '/home/vision/smb-datasets/VETE/recall_precision/pepeganga_vete_catalog_data.npy'
    queries_file_v = '/home/vision/smb-datasets/VETE/recall_precision/PepegangaCLIPBASE/query_embeddings.npy'
    
    
    catalog = np.load(catalog_file)    
    queries = np.load(queries_file)
    
    catalog_v = np.load(catalog_file_v)    
    queries_v = np.load(queries_file_v)
    
    relevants = np.load(relevants_file)
    n_relevants = np.sum(relevants, axis = 1)
    print('Catalog loaded {} {}!'.format(catalog.shape, catalog.dtype))
    print('Queries loaded {} {}!'.format(queries.shape, queries.dtype))
    print('Relevants loaded {} {}!'.format(relevants.shape, relevants.dtype))
    #compute distances
    ranking = get_ranking(catalog, queries)
    ranking_v = get_ranking(catalog_v, queries_v)
    
    p_11 = compute_p11(ranking, relevants)
    p_11_v = compute_p11(ranking_v, relevants)
    #print(p_11)
    r11 =np.arange(0,11,1)/10 
    plt.plot(r11, p_11,'r-')
    plt.plot(r11, p_11_v, 'b-')
    plt.yticks(ticks=np.arange(0,11,1)/10)
    plt.show()
    

         
    
    