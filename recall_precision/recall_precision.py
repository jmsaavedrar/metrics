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
    parser = argparse.ArgumentParser(description = 'compute recall precision')
    parser.add_argument("-catalog", type = str, help = "<str> catalog embeddings", required = True)
    parser.add_argument("-queries", type = str, help = "<str> query embeddings", required = True)
    parser.add_argument("-relevants", type = str, help = "<str> relevants", required = True)    
    
    args = parser.parse_args()
    catalog = np.load(args.catalog)    
    queries = np.load(args.queries)
    relevants = np.load(args.relevants)
    n_relevants = np.sum(relevants, axis = 1)
    print('Catalog loaded {} {}!'.format(catalog.shape, catalog.dtype))
    print('Queries loaded {} {}!'.format(queries.shape, queries.dtype))
    print('Relevants loaded {} {}!'.format(relevants.shape, relevants.dtype))
    #compute distances
    ranking = get_ranking(catalog, queries)
    p_11 = compute_p11(ranking, relevants)
    print(p_11)
    plt.plot(np.arange(0,11,1)/10, p_11)
    plt.yticks(ticks=np.arange(0,11,1)/10)
    plt.show()
    

         
    
    