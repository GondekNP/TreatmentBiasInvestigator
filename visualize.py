import seaborn as sns
import numpy as np

def bias_over_time(df, select, group, facet = None, save = False):
    select_mask = np.ones(df.shape[0],).astype(bool)
    for key, val in select.items():
        select_mask = select_mask & (df[key] == val)
    vdf = df[select_mask]
    if facet:
        vdf.loc[:,facet] = vdf.loc[:,facet].round(4) 
        g = sns.FacetGrid(data = vdf, col=facet, col_wrap=3)
        g.map(sns.lineplot, "n_steps", "estimate")
    else:
        vdf.loc[:,group] = vdf.loc[:,group].round(4) 
        sns.lineplot(data = vdf, x = 'n_steps', y = 'estimate', hue = group)