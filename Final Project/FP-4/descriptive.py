# %%
from IPython.display import display,Markdown #,HTML
import numpy as np
from scipy import stats, linalg
from matplotlib import pyplot as plt
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def display_title(s, pref='Figure', num=1, center=False):
    ctag = 'center' if center else 'p'
    s    = f'<{ctag}><span style="font-size: 1.2em;"><b>{pref} {num}</b>: {s}</span></{ctag}>'
    if pref=='Figure':
        s = f'{s}<br><br>'
    else:
        s = f'<br><br>{s}'
    display( Markdown(s) )



# %% [markdown]
import parse_data as prdt

prdt.df.describe()

# %% [markdown]
# To create a custom display of descriptive statistics, let's first define functions that will calculate central tendency and dispersion metrics.
# 
# Refer also to [this notebook](https://github.com/0todd0000/OpenBook-DataAnalysisPracticeInPythonAndJupyter/blob/master/Lessons/Lesson04/5-Examples/DescriptiveStatsExamples.ipynb) for details regarding how create custom descriptive statistics tables.

# %%
def central(x, print_output=True):
    x0     = np.mean( x )
    x1     = np.median( x )
    x2     = stats.mode( x ).mode
    return x0, x1, x2


def dispersion(x, print_output=True):
    y0 = np.std( x ) # standard deviation
    y1 = np.min( x )  # minimum
    y2 = np.max( x )  # maximum
    y3 = y2 - y1      # range
    y4 = np.percentile( x, 25 ) # 25th percentile (i.e., lower quartile)
    y5 = np.percentile( x, 75 ) # 75th percentile (i.e., upper quartile)
    y6 = y5 - y4 # inter-quartile range
    return y0,y1,y2,y3,y4,y5,y6

# %% [markdown]
# Let's now assemble and display a central tendency table:
# %%
def display_central_tendency_table(num=1):
    display_title('Central tendency summary statistics.', pref='Table', num=num, center=False)
    df_central = prdt.df.apply(lambda x: central(x), axis=0)
    round_dict = {'Gender': 3, 'Status': 3, 'Income': 3, 'P': 3, 'SC': 3, 'SI': 3,'NE': 3, 'H': 3, 'IBB': 3}
    df_central = df_central.round( round_dict )
    row_labels = 'mean', 'median', 'mode'
    df_central.index = row_labels
    display( df_central )

display_central_tendency_table(num=1)


# %%
def display_dispersion_table(num=1):
    display_title('Dispersion summary statistics.', pref='Table', num=num, center=False)
    round_dict            = {'Gender': 3, 'Status': 3, 'Income': 3, 'P': 3, 'SC': 3, 'SI': 3,
                             'NE': 3, 'H': 3, 'IBB': 3}
    df_dispersion         = prdt.df.apply(lambda x: dispersion(x), axis=0).round( round_dict )
    row_labels_dispersion = 'st.dev.', 'min', 'max', 'range', '25th', '75th', 'IQR'
    df_dispersion.index   = row_labels_dispersion
    display( df_dispersion )

display_dispersion_table(num=2)


# %% [markdown]
# From the dispersion table, let's plot them into box plots for more captivating visualilzations

# %%
def box_plot(num = 1):
    fig, axs = plt.subplots(1, 5, figsize = (16,4), tight_layout = True)
    axs[0].boxplot( prdt.df['P'], tick_labels = ['Promotion'], notch = True)
    axs[1].boxplot( prdt.df['SC'], tick_labels = ['Self-Control'], notch = True)
    axs[2].boxplot( prdt.df['SI'], tick_labels = ['Social Influence'], notch = True)
    axs[3].boxplot( prdt.df['NE'], tick_labels = ['Normative Evaluation'], notch = True)
    axs[4].boxplot( prdt.df['H'], tick_labels = ['Happiness'], notch = True)

    plt.show()

    display_title("Box plot for independent variables's distribution visualization.", pref='Figure', num=1)

box_plot(num = 1)

# %% [markdown]
# To ensure that Gender, Status, and Income is valid control variables, we will test the multicollinearity using the Variance Inflation Factor (VIF) between the CVs and the IVs

# %%
def vif_table(num = 3):
    display_title('Variance Inflation Factor Score', pref='Table', num=num, center=False)
    threshold = {1: 'Low', 2: 'Moderate', 3: 'High', 4: 'Severe'}      # Threshold of severity for VIF
    df_multi = prdt.df[['P', 'SC', 'SI', 'NE', 'H', 'Gender', 'Status', 'Income']]
    vif_data = pd.DataFrame()
    vif_data['Features'] = df_multi.columns    
    vif_data['VIF'] = [variance_inflation_factor(df_multi.values, i) for i in range(df_multi.shape[1])]   # test VIF for every features with shape[1] return number of columns
    
    def classify_vif(v):
        if v < 1:
            return threshold[1]
        elif 1 <= v < 5:
            return threshold[2]
        elif 5 <= v < 10:
            return threshold[3]
        else:
            return threshold[4]
    
    vif_data['Multicollinearity'] = vif_data['VIF'].apply(classify_vif)      # Create multicollinearity column to alert to people unfamiliar with VIF 

    return vif_data


vif_table(num = 3)    
    

# %% [markdown]
# From the score, we can see that Gender, Status and Income have moderate multicollinearity, but the IVs shows severe multicollinearity (VIF > 10).
# We will reduce multicollinearity using elastic net regression later for predictive analysis

# %% [markdown]
# Another descriptive analysis we will do is Principle Component Analysis (PCA). PCA is useful in modeling because it reduces dimensionality while keeping the most important features for later predictive analysis.
# Let's apply PCA to our independent variables

# %%
# Standardize the variables before applying PCA
df_dataframe = prdt.df[['P', 'SC', 'SI', 'NE', 'H']]

# Create a PCA table with eigenvalues and variance explained
def pca_table(num = 4):
    display_title('Principle Component Analysis.', pref='Table', num=num, center=False)
    corr_matrix = df_dataframe.corr()    # use correlation matrix for calculating eigenvalues
    a = linalg.eigvals(corr_matrix, homogeneous_eigvals=True)
    
    pca_data = pd.DataFrame()    # create a Dataframe to store informations
    pca_data['Variables'] = df_dataframe.columns
    pca_data['Eigenvalues'] = pd.Series(a.ravel()) 
    
    pca = PCA(n_components=5)    # Applying PCA
    df_pca = pca.fit_transform(df_dataframe)   
    explained_vr = pca.explained_variance_ratio_     # get the explained variance ratio for each variable
    pca_data['Variance Explained'] = [i for i in explained_vr]
    
    return pca_data

pca_table(num=4)

# %% [markdown]
# A great way to visualize the variance explained is through a scree plot

# %%
def scree_plt_ve(num = 2):
    pca = PCA(n_components=5)    # Applying PCA
    df_pca = pca.fit_transform(df_dataframe)   
    explained_vr = pca.explained_variance_ratio_     # get the explained variance ratio for each variable

    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(explained_vr) + 1), explained_vr * 100, marker = 'o', c = 'orange', lw = 2)    # times 100 to return percentages

    plt.grid()
    plt.xlabel('Principle Component')
    plt.xticks(range(1, len(explained_vr) + 1))
    plt.ylabel('Explained Variance (%)')

    plt.show()

    display_title("Scree plot of Variance Explained for Principle Components.", pref='Figure', num=num)

scree_plt_ve(num=2)

# %% [markdown]
# Let's save the variables in easier-to-use variable names:

# %%
y = prdt.df['IBB']
P = prdt.df['P']
SC = prdt.df['SC']
SI = prdt.df['SI']
NE = prdt.df['NE']
H = prdt.df['H']
G = prdt.df['Gender']
S = prdt.df['Status']
I = prdt.df['Income']

# %% [markdown]
# Let's create scatterplots for the DV (IBB) vs. each of the five IVs (Promotion, Self-Control, Social Influence, Normative Evaluation, Happiness) and CVs (Gender, Status, Income):

# %%
fig,axs = plt.subplots( 2, 4, figsize=(16,6), tight_layout=True )
axs[0, 0].scatter( P, y, alpha=0.5, color= 'b' )
axs[0, 1].scatter( SC, y, alpha=0.5, color= 'r' )
axs[0, 2].scatter( SI, y, alpha=0.5, color= 'g' )
axs[0, 3].scatter( NE, y, alpha=0.5, color= 'c' )
axs[1, 0].scatter( H, y, alpha=0.5, color= 'm' )
axs[1, 1].scatter( G, y, alpha=0.5, color= 'y' )
axs[1, 2].scatter( S, y, alpha=0.5, color= 'orange' )
axs[1, 3].scatter( I, y, alpha=0.5, color= 'brown' )

xlabels = 'Promotion', 'Self-Control', 'Social Influence', 'Normative Evaluation', 'Happiness', 'Gender', 'Status', 'Income'
[ax.set_xlabel(s) for ax,s in zip(axs.ravel(),xlabels)]
axs[0, 0].set_ylabel('Impulsive Buying Behavior')
axs[1, 0].set_ylabel('Impulsive Buying Behavior')
[ax.set_yticklabels([]) for ax in axs[:, 1:].ravel()]
plt.show()

# %% [markdown]
# Next let's add regression lines and correlation coefficients to each plot:

# %%
def corrcoeff(x, y):
    r = np.corrcoef(x, y)[0,1]
    return r

def plot_regression_line(ax, x, y, **kwargs):
    a,b   = np.polyfit(x, y, deg=1)
    x0,x1 = min(x), max(x)
    y0,y1 = a*x0 + b, a*x1 + b
    ax.plot([x0,x1], [y0,y1], **kwargs)


# %%
fig,axs = plt.subplots( 2, 4, figsize=(16,6), tight_layout=True )
ivs     = [P, SC, SI, NE, H, G, S, I]
colors  = 'b', 'r', 'g', 'c', 'm', 'y', 'orange', 'brown'
for ax,x,c in zip(axs.ravel(), ivs, colors):
    ax.scatter( x, y, alpha=0.5, color=c )
    plot_regression_line(ax, x, y, color='k', ls='-', lw=2)
    r   = corrcoeff(x, y)
    ax.text(0.7, 0.3, f'r = {r:.3f}', color=c, transform=ax.transAxes, bbox=dict(color='0.8', alpha=0.7))

xlabels = 'Promotion', 'Self-Control', 'Social Influence', 'Normative Evaluation', 'Happiness', 'Gender', 'Status', 'Income'
[ax.set_xlabel(s) for ax,s in zip(axs.ravel(),xlabels)]
axs[0, 0].set_ylabel('Impulsive Buying Behavior')
axs[1, 0].set_ylabel('Impulsive Buying Behavior')
[ax.set_yticklabels([]) for ax in axs[:, 1:].ravel()]
plt.show()

# %% [markdown]
# The correlation coefficients are all relatively low, suggesting no clear linear correlation between the DV and the IVs.
# For the CVs, we see that they show little correlation with the DV. This proves that the CVs are the relevant controllers.

# %% [markdown]
# Let's now assemble all results into a single figure for reporting purposes:

# %%
def plot_descriptive(num = 3):
    
    ig,axs = plt.subplots( 1, 5, figsize=(16,3), tight_layout=True )
    ivs     = [P, SC, SI, NE, H]
    colors  = 'b', 'r', 'g', 'c', 'm'
    for ax,x,c in zip(axs, ivs, colors):
        ax.scatter( x, y, alpha=0.5, color=c )
        plot_regression_line(ax, x, y, color='k', ls='-', lw=2)
        r   = corrcoeff(x, y)
        ax.text(0.7, 0.3, f'r = {r:.3f}', color=c, transform=ax.transAxes, bbox=dict(color='0.8', alpha=0.7))

    xlabels = 'Promotion', 'Self-Control', 'Social Influence', 'Normative Evaluation', 'Happiness'
    [ax.set_xlabel(s) for ax,s in zip(axs.ravel(),xlabels)]
    axs[0].set_ylabel('Impulsive Buying Behavior')
    [ax.set_yticklabels([]) for ax in axs[1:]]


    panel_labels = 'a', 'b', 'c', 'd', 'e'
    [ax.text(0.90, 0.92, f'({s})', size=12, transform=ax.transAxes)  for ax,s in zip(axs, panel_labels)]
    plt.show()
    
    display_title('Correlations amongst main variables.', pref='Figure', num=3)

    
plot_descriptive()


