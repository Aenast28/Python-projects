#!/usr/bin/env python
# coding: utf-8

# Customer Personality Analysis is a comprehensive examination of an enterprise's ideal clientele, facilitating a profound comprehension of customer demographics. This analytical process empowers businesses to tailor their products more effectively by discerning the distinct needs, behaviors, and concerns across various customer segments.

# By conducting a thorough customer personality analysis, enterprises can strategically adapt their product offerings to align with the preferences of specific customer groups. This approach enables businesses to optimize their marketing efforts by allocating resources strategically. For instance, rather than deploying resources to promote a new product indiscriminately to the entire customer database, companies can identify the most receptive customer segments and target marketing initiatives exclusively toward those specific segments. This targeted approach enhances efficiency, streamlines resource allocation, and ultimately contributes to a more refined and effective product-market fit.

# In[ ]:


# --- Installing Libraries ---
get_ipython().system('pip install ydata-profiling')
get_ipython().system('pip install pywaffle')
get_ipython().system('pip install yellowbrick')
get_ipython().system('pip uninstall markupsafe')
get_ipython().system('pip install markupsafe==2.0.1')
get_ipython().system('pip install Jinja2')


# In[1]:


# --- Importing Libraries ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ydata_profiling
import seaborn as sns
import warnings
import os
import yellowbrick
import scipy.cluster.hierarchy as shc
import matplotlib.patches as patches

from matplotlib.patches import Rectangle
from pywaffle import Waffle
from math import isnan
from ydata_profiling import ProfileReport
from random import sample
from numpy.random import uniform
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from yellowbrick.style import set_palette
from yellowbrick.contrib.wrapper import wrap

# --- Libraries Settings ---
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 600
sns.set(rc = {'axes.facecolor': '#FBFBFB', 'figure.facecolor': '#FBFBFB'})
class clr:
    start = '\033[93m'+'\033[1m'
    color = '\033[93m'
    end = '\033[0m'


# In[522]:


#Data
data = pd.read_csv('marketing_campaign.csv',sep='\t')

#Exploratory analysis
print(clr.start+'.: Imported Dataset :.'+clr.end)
print(clr.color+'*' * 23)
data.head().style.background_gradient(cmap='YlOrBr').hide_index()


# In[3]:


# --- Dataset Report ---
plt.rcParams['font.family'] = 'Arial'
profile = ProfileReport(data, title='Customer Personality Report')
profile.to_notebook_iframe()


# In[83]:


# --- Correlation Map (Heatmap) ---
mask = np.triu(np.ones_like(data.corr(), dtype=bool))
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(data.corr(), mask=mask, annot=True, cmap='inferno', linewidths=0.1, cbar=False, annot_kws={"size":5})
yticks, ylabels = plt.yticks()
xticks, xlabels = plt.xticks()
ax.set_xticklabels(xlabels, size=6, fontfamily='serif')
ax.set_yticklabels(ylabels, size=6, fontfamily='serif')
plt.suptitle('Correlation Map of Numerical Variables', fontweight='heavy', x=0.327, y=0.96, ha='left', fontsize=13, fontfamily='serif')
plt.title('Some variables have significant correlations with other variables (> 0.5).\n', fontsize=8, fontfamily='serif', loc='left')
plt.tight_layout(rect=[0, 0.04, 1, 1.01])
plt.show();


# # Exploratory Data Analysis

# In this section we will focus on the EDA part to gain more in depth insight of the whole dataset. Specifically, we will look for variables with unusually high/low correlation coefficient. This will help us determine which variables are important enough to keep in our training dataset. Firstly, we will try to figure out how to handle missing values that we've discovered in cathegorical variable "Income". This can be handled in several ways, such as simply filling them with means or median values of "Income", or use predictive imputation using KNN which uses its nearest neighbours to compute the mean of the NA value.

# In[523]:


# --- List Null Columns ---
null_columns = data.columns[data.isnull().any()].tolist()
print(null_columns)
# --- Perform Imputation ---
imputer = KNNImputer()
df_imp = pd.DataFrame(imputer.fit_transform(data[null_columns]), columns=null_columns)
data = data.fillna(df_imp)

missing_values = data.isnull().sum()
print(missing_values)


# With the successful handling of missing values, we can now proceed to analyze the data report and correlation matrix. Through this examination, we observe the right skewness in the majority of columns. Furthermore, several variables demonstrate a positive linear correlation (+0.5). Our attention will be directed towards exploring specific variables exhibiting such correlation to gain deeper insights into our dataset. By delving into these correlations, we aim to enhance our understanding of the data's underlying patterns and relationships, which will be instrumental in guiding our subsequent analytical endeavors.

# ## Catalog purchases X Amount spent on wine

# In[138]:


sns.set(style="whitegrid")
plt.figure(figsize=(6.5, 6.5))
colors = sns.color_palette("husl", len(data["NumCatalogPurchases"]))
# Parametr 's' určuje velikost bodů, 'c' určuje barvu bodů (může být také použito 'color')
plt.scatter(data["NumCatalogPurchases"], data["MntWines"], s=100, c=colors, alpha=0.8)

# Nastavení popisků os a titulku
plt.suptitle('Scatter plot of Catalog purchases and the amount of money spent on wine', fontweight='heavy', x=0.327, y=0.96, ha='center', fontsize=13, fontfamily='serif')
plt.title('There is positive correlation between the variables.\n', fontsize=8, fontfamily='serif', loc='center')
plt.xlabel("Number of catalog purchases")
plt.ylabel("Amount spent on Bought Wine")
plt.tight_layout(rect=[0, 0.04, 1, 1.01])


# Zobrazení grafu
plt.show()


# The scatter plot illustrates a distinct positive linear correlation between two variables. The clear relationship observed indicates that as customers use catalogues more frequently for making purchases, their expenditure on wine increases accordingly. This finding suggests the presence of a particular customer segment exhibiting this behavior pattern. Such a correlation serves as a promising indicator, affirming the suitability of our data for a classification problem. The existence of this coherent relationship provides valuable insights, which can be leveraged to effectively classify and predict customer preferences and behaviors based on their catalogue usage and wine spending patterns.

# # Feature engineering

# In[524]:


#Feature Engineering
#Age of customer today 
data["Age"] = 2023-data["Year_Birth"]

#Total spendings on various items
data["Spent"] = data["MntWines"]+ data["MntFruits"]+ data["MntMeatProducts"]+ data["MntFishProducts"]+ data["MntSweetProducts"]+ data["MntGoldProds"]

#Deriving living situation by marital status"Alone"
data["Living_With"]=data["Marital_Status"].replace({"Married":"Partner", "Together":"Partner", "Absurd":"Alone", "Widow":"Alone", "YOLO":"Alone", "Divorced":"Alone", "Single":"Alone",})

#Feature indicating total children living in the household
data["Children"]=data["Kidhome"]+data["Teenhome"]

#Deriving if they accepted the promotion or not
data["Accepted"] = data["AcceptedCmp1"]+ data["AcceptedCmp2"]+ data["AcceptedCmp3"]+ data["AcceptedCmp4"]+ data["AcceptedCmp5"]
data["Accepted"]=data["Accepted"].replace({"0":"0", "1":"1", "2":"1", "3":"1", "4":"1", "5":"1",})

#Transforming the Education variable
data['Education'] = data['Education'].replace(['PhD','2n Cycle','Graduation', 'Master'],'Post Graduate')  
data['Education'] = data['Education'].replace(['Basic'], 'Under Graduate')

#Deriving how many purchases they made in the last 2 years
data['NumTotalPurchases'] = data['NumWebPurchases'] + data['NumCatalogPurchases'] + data['NumStorePurchases'] + data['NumDealsPurchases']

to_drop = ["Dt_Customer","Z_CostContact", "Z_Revenue", "Year_Birth", "ID","Marital_Status","AcceptedCmp1","AcceptedCmp2","AcceptedCmp3","AcceptedCmp4","AcceptedCmp5"]
data = data.drop(to_drop, axis=1)


# ### Dropping outliers
# 

# In[526]:


import pandas as pd
from scipy import stats

def remove_outliers_zscore(df, columns, threshold=3):
    z_scores = np.abs(stats.zscore(df[columns]))
    df_cleaned = df[(z_scores < threshold).all(axis=1)]
    return df_cleaned

# Specify the columns for outlier detection and removal
columns_to_check = ['Income', 'Age',"Spent"]
#columns_to_check = data.columns.to_list()
data = remove_outliers_zscore(data, columns_to_check)

print("Number of rows in the cleaned dataset:", len(data))


# After eliminating redundant variables and introducing new ones, we have enhanced our ability to discern and distinguish customers within our clusters effectively. These data modifications contribute to a more refined and informative representation of the customer profiles, enabling us to achieve a deeper understanding of the underlying patterns and relationships within the data. As a result, we can now employ these enriched features to accurately identify and classify customers based on their distinct characteristics and attributes. The improved dataset strengthens our clustering analysis, providing valuable insights for devising targeted strategies and personalized approaches to cater to the unique needs and preferences of different customer segments.

# # PCA - Principal Component Analysis
# 

# In[527]:


from sklearn.preprocessing import LabelEncoder
#Encoding
non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns
label_encoder = LabelEncoder()
for col in non_numeric_columns:
    data[col] = label_encoder.fit_transform(data[col])
#Scaling
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(data))
#Initiating PCA to reduce dimentions aka features to 3
pca = PCA(n_components=3)
pca.fit(X)
PCA_ds = pd.DataFrame(pca.transform(X), columns=(["col1","col2", "col3"]))
PCA_ds.describe().T

#A 3D Projection Of Data In The Reduced Dimension
x =PCA_ds["col1"]
y =PCA_ds["col2"]
z =PCA_ds["col3"]
#To plot
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x,y, c="maroon", marker="o" )
ax.set_title("A 3D Projection Of Data In The Reduced Dimension")
plt.show()


# Label Encoding:
# The first step is to encode non-numeric columns in the dataset. Label encoding is applied using the LabelEncoder from Scikit-learn. This process converts categorical data (non-numeric) into numerical representations, which allows algorithms to work with such data.
# 
# Scaling:
# Next, the data is scaled using the StandardScaler. Scaling ensures that all features have a mean of 0 and a standard deviation of 1. It helps in maintaining consistency and avoiding any dominance of one feature over another during the PCA process.
# 
# PCA for Dimensionality Reduction:
# PCA is then applied to reduce the number of dimensions (features) in the dataset to three (n_components=3). PCA transforms the original features into a new set of uncorrelated features called principal components. These principal components are linear combinations of the original features and are ranked by their ability to explain the most variance in the data.
# 
# Creating a DataFrame with Reduced Dimensions:
# The result of PCA transformation is stored in a new DataFrame called PCA_ds, containing the reduced three-dimensional representation of the original data.
# 
# Visualizing the Reduced Data:
# To visualize the reduced data in a 3D space, the col1, col2, and col3 columns from the PCA_ds DataFrame are extracted and used as the X, Y, and Z coordinates for the 3D scatter plot.
# 
# Plotting the 3D Projection:
# Finally, the 3D scatter plot is created using Matplotlib to visualize the data in the reduced dimensionality.

# In[528]:


# --- Define K-Means Functions ---
def kmeans():
    
    # --- Figures Settings ---
    color_palette=['#FFCC00', '#54318C']
    set_palette(color_palette)
    title=dict(fontsize=12, fontweight='bold', style='italic', fontfamily='serif')
    text_style=dict(fontweight='bold', fontfamily='serif')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Elbow Score ---
    elbow_score = KElbowVisualizer(KMeans(random_state=32, max_iter=500), k=(2, 10), ax=ax1)
    elbow_score.fit(X)
    elbow_score.finalize()
    elbow_score.ax.set_title('Distortion Score Elbow\n', **title)
    elbow_score.ax.tick_params(labelsize=7)
    for text in elbow_score.ax.legend_.texts:
        text.set_fontsize(9)
    for spine in elbow_score.ax.spines.values():
        spine.set_color('None')
    elbow_score.ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), borderpad=2, frameon=False, fontsize=8)
    elbow_score.ax.grid(axis='y', alpha=0.5, color='#9B9A9C', linestyle='dotted')
    elbow_score.ax.grid(axis='x', alpha=0)
    elbow_score.ax.set_xlabel('\nK Values', fontsize=9, **text_style)
    elbow_score.ax.set_ylabel('Distortion Scores\n', fontsize=9, **text_style)
    
    # --- Elbow Score (Calinski-Harabasz Index) ---
    elbow_score_ch = KElbowVisualizer(KMeans(random_state=32, max_iter=500), k=(2, 10), metric='calinski_harabasz', timings=False, ax=ax2)
    elbow_score_ch.fit(X)
    elbow_score_ch.finalize()
    elbow_score_ch.ax.set_title('Calinski-Harabasz Score Elbow\n', **title)
    elbow_score_ch.ax.tick_params(labelsize=7)
    for text in elbow_score_ch.ax.legend_.texts:
        text.set_fontsize(9)
    for spine in elbow_score_ch.ax.spines.values():
        spine.set_color('None')
    elbow_score_ch.ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), borderpad=2, frameon=False, fontsize=8)
    elbow_score_ch.ax.grid(axis='y', alpha=0.5, color='#9B9A9C', linestyle='dotted')
    elbow_score_ch.ax.grid(axis='x', alpha=0)
    elbow_score_ch.ax.set_xlabel('\nK Values', fontsize=9, **text_style)
    elbow_score_ch.ax.set_ylabel('Calinski-Harabasz Score\n', fontsize=9, **text_style)
    
    plt.suptitle('Credit Card Customer Clustering using K-Means', fontsize=14, **text_style)
    plt.tight_layout()
    plt.show();

# --- Calling K-Means Functions ---
kmeans();


# Distortion Score for Elbow Method (K-Means Clustering):
# 
# The Distortion score is a metric used to evaluate the quality of clustering in K-Means. After applying the Elbow method, which helped us determine that five clusters were ideal for our dataset, we use the Distortion score to assess how well the data points are grouped within these five clusters.

# In[538]:


# K-means
kmeans = KMeans(n_clusters=4, random_state=42, max_iter=100)
y_kmeans = kmeans.fit_predict(X)
data["Clusters"]= y_kmeans
# Create the 3D Projection
x = PCA_ds["col1"]
y = PCA_ds["col2"]
z = PCA_ds["col3"]

# Visualisation
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=y_kmeans, s=40, marker='o', cmap='viridis')
ax.set_title("The Plot Of The Clusters")
plt.show()


# We used K-means clustering to partition the data into five clusters based on three-dimensional features (col1, col2, col3). The 3D scatter plot visually represents the clusters, each marked by a unique color and centered around a centroid. This visualization provides valuable insights into the distinct customer segments and their relationships, offering a more informative representation than a traditional 2D plot.

# In[539]:


# --- Evaluate Clustering Quality Function ---
def evaluate_clustering(X, y):
    db_index = round(davies_bouldin_score(X, y), 3)
    s_score = round(silhouette_score(X, y), 3)
    ch_index = round(calinski_harabasz_score(X, y), 3)
    print(clr.start+'.: Evaluate Clustering Quality :.'+clr.end)
    print(clr.color+'*' * 34+clr.end)
    print('.: Davies-Bouldin Index: '+clr.start, db_index)
    print(clr.end+'.: Silhouette Score: '+clr.start, s_score)
    print(clr.end+'.: Calinski Harabasz Index: '+clr.start, ch_index)
    return db_index, s_score, ch_index

# --- Evaluate K-Means Cluster Quality ---
db_kmeans, ss_kmeans, ch_kmeans = evaluate_clustering(X, y_kmeans)


# In[546]:


# Plotting countplot of clusters
pal = ["#682F2F", "#B9C0C9", "#9F8A78", "#F3AB60","#F3AC60","#300F2F",]
plt.figure(figsize=(12, 6))
sns.countplot(x=y_kmeans)
plt.title("Distribution of Data Points in Each Cluster")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.show()


# In[547]:


# --- Define Dendrogram ---
def agg_dendrogram():
    
    # --- Figure Settings ---
    color_palette=['#472165', '#FFBB00', '#3C096C', '#9D4EDD', '#FFE270']
    set_palette(color_palette)
    text_style=dict(fontweight='bold', fontfamily='serif')
    ann=dict(textcoords='offset points', va='center', ha='center', fontfamily='serif', style='italic')
    title=dict(fontsize=12, fontweight='bold', style='italic', fontfamily='serif')
    bbox=dict(boxstyle='round', pad=0.3, color='#FFDA47', alpha=0.6)
    fig=plt.figure(figsize=(14, 5))
    
    # --- Dendrogram Plot ---
    ax1=fig.add_subplot(1, 2, 1)
    dend=shc.dendrogram(shc.linkage(X, method='ward', metric='euclidean'))
    plt.axhline(y=115, color='#3E3B39', linestyle='--')
    plt.xlabel('\nData Points', fontsize=9, **text_style)
    plt.ylabel('Euclidean Distances\n', fontsize=9, **text_style)
    plt.annotate('Horizontal Cut Line', xy=(15000, 130), xytext=(1, 1), fontsize=8, bbox=bbox, **ann)
    plt.tick_params(labelbottom=False)
    for spine in ax1.spines.values():
        spine.set_color('None')
    plt.grid(axis='both', alpha=0)
    plt.tick_params(labelsize=7)
    plt.title('Dendrograms\n', **title)
    
    # --- Elbow Score (Calinski-Harabasz Index) ---
    ax2=fig.add_subplot(1, 2, 2)
    elbow_score_ch = KElbowVisualizer(AgglomerativeClustering(), metric='calinski_harabasz', timings=False, ax=ax2)
    elbow_score_ch.fit(X)
    elbow_score_ch.finalize()
    elbow_score_ch.ax.set_title('Calinski-Harabasz Score Elbow\n', **title)
    elbow_score_ch.ax.tick_params(labelsize=7)
    for text in elbow_score_ch.ax.legend_.texts:
        text.set_fontsize(9)
    for spine in elbow_score_ch.ax.spines.values():
        spine.set_color('None')
    elbow_score_ch.ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), borderpad=2, frameon=False, fontsize=8)
    elbow_score_ch.ax.grid(axis='y', alpha=0.5, color='#9B9A9C', linestyle='dotted')
    elbow_score_ch.ax.grid(axis='x', alpha=0)
    elbow_score_ch.ax.set_xlabel('\nK Values', fontsize=9, **text_style)
    elbow_score_ch.ax.set_ylabel('Calinski-Harabasz Score\n', fontsize=9, **text_style)
    
    plt.suptitle('Customer Behavior Clustering using Hierarchical Clustering\n', fontsize=14, **text_style)
    plt.tight_layout()
    plt.show();

# --- Calling Dendrogram Functions ---
agg_dendrogram();


# In[548]:


# --- Implementing Hierarchical Clustering ---
agg_cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
y_agg_cluster = agg_cluster.fit_predict(X)
    
# --- Define Hierarchical Clustering Distributions ---
def agg_visualizer(agg_cluster, y_agg_cluster):
    
    # --- Figures Settings ---
    cluster_colors=['#FFBB00', '#3C096C', '#9D4EDD', '#FFE270']
    labels = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']
    suptitle=dict(fontsize=14, fontweight='heavy', fontfamily='serif')
    title=dict(fontsize=10, fontweight='bold', style='italic', fontfamily='serif')
    scatter_style=dict(linewidth=0.65, edgecolor='#100C07', alpha=0.85)
    legend_style=dict(borderpad=2, frameon=False, fontsize=9)
    fig=plt.figure(figsize=(14, 7))
    
    # --- Percentage Labels ---
    unique, counts = np.unique(y_agg_cluster, return_counts=True)
    df_waffle = dict(zip(unique, counts))
    total = sum(df_waffle.values())
    wfl_square = {key: value/100 for key, value in df_waffle.items()}
    wfl_label = {key: round(value/total*100, 2) for key, value in df_waffle.items()}

    # --- Clusters Distribution ---
    y_agg_labels = list(set(y_agg_cluster.tolist()))
    ax1=fig.add_subplot(1, 3, (1, 2))
    for i in y_agg_labels:
        ax1.scatter(X[y_agg_cluster==i, 0], X[y_agg_cluster == i, 1], s=50, c=cluster_colors[i], label=labels[i], **scatter_style)
    for spine in ax1.spines.values():
        spine.set_color('None')
    for spine in ['bottom', 'left']:
        ax1.spines[spine].set_visible(True)
        ax1.spines[spine].set_color('#CAC9CD')
    ax1.legend([f"Cluster {i+1} - ({k}%)" for i, k in wfl_label.items()], bbox_to_anchor=(1.3, -0.03), ncol=4, **legend_style)
    ax1.grid(axis='both', alpha=0.3, color='#9B9A9C', linestyle='dotted')
    ax1.tick_params(left=False, right=False , labelleft=False , labelbottom=False, bottom=False)
    plt.title('Scatter Plot Clusters Distributions\n', **title)


# In[549]:


db_agg, ss_agg, ch_agg = evaluate_clustering(X, y_agg_cluster)


# In[558]:


# Comparison of models
comparison = {
    'Davies-Bouldin Index': [db_agg, db_kmeans],
    'Silhouette Score': [ss_agg, ss_kmeans],
    'Calinski Harabasz Index': [ch_agg, ch_kmeans]
}
index_names=["Agglomerative clustering","Kmeans clustering"]
# Vytvoření DataFrame
model_comparison = pd.DataFrame(comparison,index=index_names)
model_comparison.style.background_gradient(cmap='YlOrBr')


# In[541]:


# --- Add K-Means Prediction to Data Frame ----
data['cluster_result'] = y_kmeans+1
data['cluster_result'] = 'Cluster '+data['cluster_result'].astype(str)

# --- Calculationg Overall Mean from Current Data Frame ---
df_profile_overall = pd.DataFrame()
df_profile_overall['Overall'] = data.describe().loc[['mean']].T

# --- Summarize Mean of Each Clusters --- 
df_cluster_summary = data.groupby('cluster_result').describe().T.reset_index().rename(columns={'level_0': 'Column Name', 'level_1': 'Metrics'})
df_cluster_summary = df_cluster_summary[df_cluster_summary['Metrics'] == 'mean'].set_index('Column Name')

# --- Combining Both Data Frame ---
print(clr.start+'.: Summarize of Each Clusters :.'+clr.end)
print(clr.color+'*' * 33)
df_profile = df_cluster_summary.join(df_profile_overall).reset_index()
df_profile.style.background_gradient(cmap='YlOrBr').hide_index()


# In[542]:


pl = sns.scatterplot(data = data,x=data["Spent"], y=data["Income"],hue=data["cluster_result"])
pl.set_title("Cluster's Profile Based On Income And Spending")
plt.legend()
plt.show()


# In[543]:


pl = sns.scatterplot(data = data,x=data["Age"], y=data["Spent"],hue=data["cluster_result"])
pl.set_title("Cluster's Profile Based On Income And Spending")
plt.legend()
plt.show()


# In[544]:


#Plotting count of total campaign accepted.
plt.figure()
pl = sns.countplot(x=data["Living_With"],hue=data["cluster_result"], palette= pal)
pl.set_title("Count Of Promotion Accepted")
pl.set_xlabel("Number Of Total Accepted Promotions")
plt.show()


# In[545]:


# Total spent for each cluster
total_spent_per_cluster = data.groupby('cluster_result')['Spent'].sum().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='cluster_result', y='Spent', data=total_spent_per_cluster, palette= pal)


# In[ ]:





# In[ ]:




