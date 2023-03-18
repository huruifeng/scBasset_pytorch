import os
import tarfile
import urllib.request

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from utils import *

# %%
## Download data
# download_savepath = 'examples/example_data/10X_PBMC'
# if not os.path.exists(download_savepath):
#     os.makedirs(download_savepath, exist_ok=True)
#
# if not os.path.exists('%s/buen_ad_sc.h5ad'%download_savepath):
#     urllib.request.urlretrieve("https://storage.googleapis.com/scbasset_tutorial_data/buen_ad_sc.h5ad", download_savepath+"/buen_ad_sc.h5ad")
#
# if not os.path.exists('%s/buen_model_sc.h5'%download_savepath):
#     urllib.request.urlretrieve("https://storage.googleapis.com/scbasset_tutorial_data/buen_model_sc.h5", download_savepath+"/buen_model_sc.h5")
#
# if not os.path.exists('%s/pbmc_ad.h5ad'%download_savepath):
#     urllib.request.urlretrieve("https://storage.googleapis.com/scbasset_tutorial_data/pbmc_ad.h5ad", download_savepath+"/pbmc_ad.h5ad")
#
# if not os.path.exists('%s/pbmc_model.h5'%download_savepath):
#     urllib.request.urlretrieve("https://storage.googleapis.com/scbasset_tutorial_data/pbmc_model.h5", download_savepath+"/pbmc_model.h5")
#
# # download motif injection fasta files for CISBP-1.0 motifs and unzip:
# if not os.path.exists('%s/Homo_sapiens_motif_fasta.tar.gz'%download_savepath):
#     urllib.request.urlretrieve("https://storage.googleapis.com/scbasset_tutorial_data/Homo_sapiens_motif_fasta.tar.gz", download_savepath+"/Homo_sapiens_motif_fasta.tar.gz")
#
# # open file
# file = tarfile.open(download_savepath+'/Homo_sapiens_motif_fasta.tar.gz')
# # print file names
# print(file.getnames())
# # extract files
# file.extractall(download_savepath)
# # close file
# file.close()

# %%
## https://support.10xgenomics.com/single-cell-multiome-atac-gex/datasets/2.0.0/pbmc_granulocyte_sorted_3k
example_data_path = 'example_data/10X_PBMC/from10X'

h5_file = example_data_path + '/outputs/pbmc_granulocyte_sorted_3k_filtered_feature_bc_matrix.h5'
bed_file = example_data_path + '/outputs/pbmc_granulocyte_sorted_3k_atac_peaks.bed'

peak = pd.read_csv(bed_file, sep='\t', names=['chr','start','end'])
peak = peak.loc[~peak["chr"].str.startswith("#"),:]
ad = sc.read_10x_h5(h5_file, gex_only=False)

ad_rna = ad[:, ad.var['feature_types']=='Gene Expression']
ad_atac = ad[:, ad.var['feature_types']=='Peaks']
ad_atac.var['chr'] = peak['chr'].values
ad_atac.var['start'] = peak['start'].values
ad_atac.var['end'] = peak['end'].values

# basic stats
sc.pp.filter_cells(ad_rna, min_genes=0)
sc.pp.filter_genes(ad_rna, min_cells=0)
sc.pp.filter_cells(ad_atac, min_genes=0)
sc.pp.filter_genes(ad_atac, min_cells=0)

# a gene need to be expressed in 5% cells
# a peak need to be accessible in 5% cells
thres = int(ad.shape[0]*0.05)
ad_rna = ad_rna[:, ad_rna.var['n_cells']>thres]
ad_atac = ad_atac[:, ad_atac.var['n_cells']>thres]

output_path = 'PBMC_example'
if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

chrs = ['chr'+str(i) for i in range(1,23)] + ['chrX', 'chrY']
ad_atac = ad_atac[:, ad_atac.var['chr'].isin(chrs)]
ad_atac.write(output_path+'/atac_ad.h5ad')

# %%
fasta_dict = read_fasta("example_data/10X_PBMC/hg38.fa")
make_h5_sparse(ad_atac, '%s/all_seqs.h5' % output_path, fasta_dict)

seqs_data = h5py.File('%s/all_seqs.h5' % output_path, 'r')
n_cells = ad_atac.shape[0]

m = ad_atac.X
m = m.tocoo().transpose().tocsr() # sparse matrix, rows as seqs, cols are cells

seq_X = seqs_data['X']
n_peaks = seq_X.shape[0]
X_dataset = []
Y_dataset = []
for i in range(n_peaks):
    x = seq_X[i]
    x_ohseq = sparse.coo_matrix((np.ones(1344), (np.arange(1344), x)),shape=(1344,4), dtype='int8').toarray().transpose()
    y = m.indices[m.indptr[i]:m.indptr[i + 1]]
    y_ones = np.zeros(n_cells, dtype='int8')
    y_ones[y] = 1
    X_dataset.append(x_ohseq)
    Y_dataset.append(y_ones)
X_dataset = np.array(X_dataset)
Y_dataset = np.array(Y_dataset)

f = h5py.File('%s/processed_data.h5' % output_path, "w")
f.create_dataset("X", data=X_dataset)
f.create_dataset("Y", data=Y_dataset)
f.close()


# %%




