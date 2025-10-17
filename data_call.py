

import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import torch

def get_ERV_ES_idx(option,stem_path,d):

    gene_df = pd.read_csv(stem_path + "stem_gene_names.csv", index_col=0)
    # print(gene_df.values)
    all_gene_names = []
    for ell in gene_df.values:
        all_gene_names.append(ell[0])


    with open(stem_path + 'scTE_features.txt', 'r') as file:
        # Read all lines from the file
        names = file.readlines()

    # Remove any trailing newline characters from each name
    TEnames = [name.strip() for name in names]

    top_gene_names = pd.read_csv(stem_path + "top_2kgene_names.csv", header=None).squeeze().tolist()

    print(len(all_gene_names), len(TEnames), len(top_gene_names))

    all_idx = [i for i in range(d)]

    te_idx = []
    top_idx = []
    for ell in range(len(all_gene_names)):
        if all_gene_names[ell] in TEnames:
            te_idx.append(ell)

        if all_gene_names[ell] in top_gene_names:
            top_idx.append(ell)

    rem_idx = list(set(all_idx) - set(te_idx) - set(top_idx))


    if option=='top2k':
        return top_idx

    elif option=='all':
        return all_idx

    elif option=='TE':
        return te_idx
    elif option=='rem':
        return rem_idx

    else:
        raise KeyError(f"Option '{option}' not found")



from sklearn.decomposition import TruncatedSVD

ch='test'


def get_dataset(name,mode='PCA', dpath_in = None):
    import numpy as np
    dpath = '/Users/chandrasekharmukherjee/Data/'
    if dpath_in is not None:
        dpath = dpath_in

    if name=='CIFAR10-embedding':
        X = np.load(dpath+'CIFAR-embedding/cifar10_embeddings.npy')
        label = np.load(dpath+'CIFAR-embedding/cifar10_saved_labels.npy')

    elif name=='CIFAR10-clip':
        import random

        filename = '/Users/chandrasekharmukherjee/Downloads/NeurIPS2024-Manifold/data/cifar10-clipfeat.pt'
        fulldata = torch.load(filename)
        X = fulldata['features'].numpy()
        label_0 = fulldata['labels']
        label = [ell.item() for ell in label_0]

        # Define the range and sample size
        start = 0
        end = 50000 - 1
        sample_size = 10000

        # Randomly sample 10,000 points without replacement
        sampled_points = random.sample(range(start, end + 1), sample_size)

        X = X[sampled_points]
        label = np.array(label)[sampled_points].tolist()


    elif name=='CIFAR100-embedding':
        X = np.load(dpath+'CIFAR-embedding/cifar100_embeddings.npy')
        label = np.load(dpath+'CIFAR-embedding/cifar100_saved_labels.npy')



    elif name=='Fashion-MNIST':
        X=np.load(dpath+'Fashion-MNIST/'+ch+'.npy')
        X=X.reshape(X.shape[0], -1)
        label=np.load(dpath+'Fashion-MNIST/'+ch+'_label.npy')


    elif name=='MNIST':
        X = np.load(dpath + 'MNIST/'+ch+'.npy')
        X = X.reshape(X.shape[0], -1)
        label = np.load(dpath + 'MNIST/'+ch+'_label.npy')


    elif name=='KMNIST':
        X = np.load(dpath + 'KMNIST/'+ch+'.npy')
        X = X.reshape(X.shape[0], -1)
        label = np.load(dpath + 'KMNIST/'+ch+'_label.npy')


    elif name=='usps':
        from sklearn.datasets import fetch_openml

        # Load USPS dataset from OpenML
        usps = fetch_openml('usps', version=1)

        # Access the data and target
        X = usps.data  # 16x16 images as flattened arrays (array of shape [N, 256])
        y = usps.target  # Target labels for classification (0-9)

        X = X.to_numpy()  # or X = X.values
        y = y.to_numpy()

        # Optionally convert the target to integer values
        label = y.astype(int).tolist()


    elif name in ['cifar10_clip_large','cifar10_clip_small']:

        X=np.load(dpath+'clip_emb/'+name+'.npy')
        label=np.load(dpath+'clip_emb/cifar10_labels.npy')

        n= X.shape[0]
        #indices = np.random.choice(np.arange(n), size=int(0.2 * n), replace=False)
        #X = X[indices]
        #label = np.array(label)[indices]


    elif name in ['cifar100_clip_large', 'cifar100_clip_small']:
        X= np.load(dpath + 'clip_emb/' + name + '.npy')
        label = np.load(dpath + 'clip_emb/cifar100_labels.npy')


    elif name in ['cifar20_clip_large', 'cifar20_clip_small']:

        import torchvision
        from torchvision import transforms
        rename={}
        rename['cifar20_clip_large']= 'cifar100_clip_large'
        rename['cifar20_clip_small'] = 'cifar100_clip_small'

        X = np.load(dpath + 'clip_emb/' + rename[name] + '.npy')
        label0 = np.load(dpath + 'clip_emb/cifar100_labels.npy')
        n=X.shape[0]

        label_map={0: 4,1: 1,2: 14,3: 8,4: 0,5: 6,6: 7,7: 7,8: 18,9: 3,10: 3,11: 14,12: 9,13: 18,14: 7,15: 11, 16: 3,17: 9,18: 7,19: 11,20: 6,21: 11,22: 5,23: 10,24: 7,25: 6,26: 13,27: 15,28: 3,29: 15,30: 0,31: 11,32: 1,33: 10,
34: 12,35: 14,36: 16,37: 9,38: 11,39: 5,40: 5,41: 19,42: 8,43: 8,44: 15,45: 13,46: 14,47: 17,48: 18,49: 10,50: 16, 51: 4,52: 17,53: 4,54: 2,55: 0,56: 17,57: 4,58: 18,59: 17, 60: 10,61: 3,62: 2,63: 12,64: 12,65: 16,66: 12,67: 1,68: 9,69: 19,70: 2,71: 10,72: 0,
73: 1,74: 16,75: 12,76: 9,77: 13,78: 15,79: 13,80: 16, 81: 19,82: 2,83: 4,84: 6,85: 19,86: 5,87: 5,88: 8,89: 19,90: 18,91: 1, 92: 2,93: 15,94: 6,95: 0,96: 17,  97: 8,98: 14,99: 13}

        label=-1*np.ones(n)
        for i in range(n):
            label[i]=label_map[label0[i]]




    elif name in ['svhn_clip_large','svhn_clip_small']:
        X= np.load(dpath + 'clip_emb/' + name + '.npy')
        label = np.load(dpath + 'clip_emb/svhn_labels.npy')



    elif name=='ERV-ES':

        option='top2k'

        stem_path='/Users/chandrasekharmukherjee/Data/ERV-ES/'

        path = stem_path + 'RNA_counts.mtx'
        # Load the matrix from the .mtx file
        matrix = scipy.io.mmread(path).T
        n=matrix.shape[0]
        d=matrix.shape[1]
        dense_matrix = matrix.toarray()

        idx=get_ERV_ES_idx(option,stem_path,d)

        pca_comp = 50

        # Initialize Truncated SVD
        svd = TruncatedSVD(n_components=pca_comp)

        M_top = dense_matrix[:, idx] + 1
        PX = svd.fit_transform(np.log2(M_top))

        #Getting cell_states as label for now.
        meta_df = pd.read_csv(stem_path + "stem_metadata.csv", index_col=0)
        label = meta_df['cellstate'].to_numpy()

        return PX,label


    elif name in ['miRNA' , 'mRNA']:
        X,label=local_bulkRNA(name,dpath,mode=mode)


    elif name in ['Tcell-medicine','Zheng','Zhengmix8eq','ALM','AMB','VISP','Muraro', 'Baron_Human',  'Baron_Mouse', 'Xin', 'TM', 'Segerstolpe']:

        if mode=='PCA':
            sc_path = dpath + 'scRNA_pca/' + name + '/'
            X = np.load(sc_path + 'data.npy')
            label = np.load(sc_path + 'labels.npy')



        else:
            sc_path=dpath+'scRNA/'+name+'/'

            X_0=scipy.sparse.load_npz(sc_path+'data.npz')
            label = np.load(sc_path+'labels.npy')
            X_0=X_0.log1p()
            #
            # c_num=max(50,len(set(label)))
            #
            # svd = TruncatedSVD(n_components=c_num)
            # X = svd.fit_transform(X_0)
            X=X_0


    elif name.startswith("IBD"):
        result = "_".join(name.split("_")[1:])
        sc_path = dpath + 'scRNA_IBD/' + result + '_final.h5ad'
        X=sc.read_h5ad(sc_path)
        label=None



    elif name =='CIFAR-10':

        from tensorflow.keras.datasets import cifar10
        import numpy as np

        # Load CIFAR-10/CIFAR-100 dataset
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # Normalize and flatten
        x_test = x_test.reshape(-1, 3072).astype(np.float32)
        x_train = x_train.reshape(-1, 3072).astype(np.float32)

        label = []
        for ell in range(y_train.shape[0]):
            label.append(y_train[ell][0])

        X = x_train

    elif name =='CIFAR-100':
        from tensorflow.keras.datasets import cifar100
        import numpy as np

        # Load CIFAR-10/CIFAR-100 dataset
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()

        # Normalize and flatten
        x_test = x_test.reshape(-1, 3072).astype(np.float32)
        x_train=x_train.reshape(-1, 3072).astype(np.float32)

        label = []
        for ell in range(y_train.shape[0]):
            label.append(y_train[ell][0])


        X=x_train


    elif name in ['bbc_news', 'BBC_Sports', 'biorxiv', 'reddit', '20NewsGroups', 'big_patent']:
        features = torch.load(dpath+f'{name}/features.pt',weights_only=False)
        label = torch.load(dpath+f'{name}/labels.pt',weights_only=False)
        pca = TruncatedSVD(n_components=500)
        X = pca.fit_transform(features)



    elif name == '20NewsGroups_tfdif':

        import datasets
        from sklearn.feature_extraction.text import TfidfVectorizer
        raw_data = datasets.load_dataset("mteb/twentynewsgroups-clustering")['test'][9]
        label = np.array(raw_data['labels'])
        text = raw_data['sentences']
        tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                                stop_words='english')
        features = tfidf.fit_transform(text).toarray()
        pca = TruncatedSVD(n_components=50)
        X = pca.fit_transform(features)

    else:
        raise KeyError(f"Dataset'{name}' not found")





    return X,label


def bulkRNA_process(X, labels,mode='PCA'):
    lset = labels[:, 0]

    idx0 = []
    for i in range(X.shape[0]):
        if (X[i, 0] in lset):
            idx0.append(i)
    idx0 = np.array(list(idx0))
    print(idx0[0:10])

    X1 = X[idx0, :].copy()
    print(X1.shape)

    idx = []
    for i in range(X1.shape[0]):
        c = 0
        t1 = X1[i, 0]
        for j in lset:
            if (j == t1):
                idx.append(c)
                break
            c = c + 1

    #print('selected vertices ', len(idx))

    label = labels[idx, 1]
    #print(label.shape, Counter(label))

    Y = np.log2(X1[:, 1:].astype(float) + 1)

    if mode == 'PCA':
        svd = TruncatedSVD(n_components=50)
        PX = svd.fit_transform(Y)

    else:
        PX= Y.copy()


    return PX,label



def local_bulkRNA(name, dpath,mode='PCA',survive=0):
    datapath1 = dpath + 'Multiomics/'

    if name == 'miRNA':
        df_labels1 = pd.read_csv(datapath1 + 'sample_sheet_mirna.csv')
        x3 = df_labels1[['Sample ID', 'Project ID']]
        dfl = x3.to_numpy()
        dfr = pd.read_csv(datapath1 + 'miRNA_raw.csv')
        Xdf = dfr.to_numpy()

    elif name == 'mRNA':
        df_labels2 = pd.read_csv(datapath1 + 'sample_sheet_mrna.csv')
        x3 = df_labels2[['Sample ID', 'Project ID']]
        dfl = x3.to_numpy()
        dfr = pd.read_csv(datapath1 + 'mRNA_pc_gene_raw.csv')
        Xdf = dfr.to_numpy()

    else:
        raise KeyError(f"Wrong name choice:'{name}'")


    if survive == 1:
        survive = pd.read_csv(datapath1 + 'survival.csv')
        survive = survive.to_numpy()
        dfsurvive = survive[:, [0, 2]]

        Xdft = Xdf.copy()
        for i in range(Xdft.shape[0]):
            ell = Xdft[i, 0]
            ell1 = ell[:-4]
            Xdft[i, 0] = ell1

        Xdf = Xdft.copy()
        dfl = dfsurvive.copy()




    X,label=bulkRNA_process(Xdf,dfl,mode=mode)

    return X,label



