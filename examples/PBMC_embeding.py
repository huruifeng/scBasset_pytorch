import pandas as pd
import scanpy as sc
import scipy
import torch
from matplotlib import pyplot as plt
import seaborn as sns


from scBasset import scBasset
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_path = 'PBMC_example'
if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

ad_file = 'PBMC_example/atac_ad.h5ad'
trained_model = 'PBMC_example/trained_model.pt'

# read h5ad file
ad = anndata.read_h5ad(ad_file)
n_cells = ad.shape[0]
n_peaks = ad.shape[1]


model = scBasset(seq_len=1344, init_dim=288, bottleneck_size=32,cell_num=n_cells,device=device)
device = torch.device("cpu")
model.load_state_dict(torch.load(trained_model, map_location=device))

intercept = model.get_intercept().numpy() # get_intercept function
sc.pp.filter_cells(ad, min_counts=0)

df = pd.DataFrame({"intercept":intercept,"n_peaks": np.log10(ad.obs['n_genes']).values})

f, ax = plt.subplots(figsize=(4,4))
r = scipy.stats.pearsonr(intercept,np.log10(ad.obs['n_genes']))[0]
sns.scatterplot(df, x="intercept",y="n_peaks", ax=ax)
ax.set_xlabel('intercept')
ax.set_ylabel('log10(n_peaks)')
ax.set_title('Pearson R: %.3f'%r)
plt.tight_layout()
f.savefig(output_path+'/intercept.pdf')

proj = model.get_cell_embedding().numpy() # get_cell_embedding function
pd.DataFrame(proj).to_csv(output_path+'/projection_atac.csv')

f, ax = plt.subplots(figsize=(4, 4))
ad.obsm['projection'] = pd.read_csv(output_path+'/projection_atac.csv', index_col=0).values
sc.pp.neighbors(ad, use_rep='projection')
sc.tl.umap(ad)
sc.tl.leiden(ad)
sc.pl.umap(ad, color='leiden', ax=ax)
plt.tight_layout()
f.savefig(output_path+'/umap.pdf')







