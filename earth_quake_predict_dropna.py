import pandas as pd, seaborn as sns, matplotlib.pyplot as plt, numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Function to one-hot coding using get_dummies() method
def one_hot(matrix, column):
    type_dummies = pd.get_dummies(matrix[column], prefix='type')
    matrix.drop([column], axis=1, inplace=True)
    matrix = pd.concat([matrix, type_dummies], axis=1)
    return matrix


# Function to split the training set and validation set
def split_data(x_data, y_data, test_size):
    x_train, x_test, y_train, y_test = train_test_split(x_data,
                                                        y_data,
                                                        test_size=test_size,
                                                        random_state=42)
    return x_train, x_test, y_train, y_test


# Read data
# Change data path on your computer
data_path = 'earthquakes.csv'
data = pd.read_csv(data_path)
# Print the data information
data.info()
# Print the data description
data.describe()

# Remove some features that aren't important for classify / clustering the alert of the earthquake
# Feature 'date' is already converted to float type in the original dataset
data.drop(['id', 'date', 'title', 'url', 'detailUrl',
           'ids', 'sources', 'types', 'net', 'code',
           'geometryType', 'status', 'postcode', 'what3words',
           'locationDetails'], axis=1, inplace=True)
# Replace Null value of column 'alert' to unknown
data.fillna({'alert': 'unknown'}, inplace=True)
# Remove samples contain Null values
data.dropna(how='any', axis=0, inplace=True)

# Columns will be one-hot coding
string_columns = data.select_dtypes(include=['object']).columns.tolist()
string_columns.remove('alert')

for column in string_columns:
    data = one_hot(data, column)

x_data = data.drop(['alert'], axis=1)
x_data = StandardScaler().fit_transform(x_data)
y_data = data['alert']

# Dimensionality reduction
# Using PCA method
pca = PCA(n_components=2, random_state=42)
x_data_pca = pca.fit_transform(x_data)

# Data visualization (2d scatter plot)
pc1 = x_data_pca[:, 0]
pc2 = x_data_pca[:, 1]
# Creat scatter plot
custom_palette = {'red': "red", 'yellow': "yellow", 'green': "green", 'unknown': "purple", 'orange': "orange"}
plot = sns.scatterplot(x=pc1, y=pc2, hue=y_data, palette=custom_palette)
# Rename axis
plot.set(xlabel="First Principal Component", ylabel="Second Principal Component")
plt.show()

# Print the explained variance after reduce dimension of the data
print(f'Explained variance: {np.cumsum(pca.explained_variance_ratio_)}')
# Print the statistic values of Principal Component
pd.DataFrame(data=x_data_pca,
             columns=['First Principal Component',
                      'Second Principal Component']).describe()

pca_pair = PCA(n_components=6)
X_pca = pca_pair.fit_transform(x_data)
df_pca = pd.DataFrame(X_pca, columns=[f'PC{i + 1}' for i in range(6)])
df_pca['target'] = y_data
plt.figure(figsize=(20, 20))
custom_palette = {'red': "red", 'yellow': "yellow", 'green': "green", 'unknown': "purple", 'orange': "orange"}
sns.pairplot(df_pca, hue='target', diag_kind='kde',
             plot_kws={'alpha': 0.6, 's': 50, 'edgecolor': 'k'},
             vars=[f'PC{i + 1}' for i in range(6)], palette=custom_palette)  # Sử dụng 6 thành phần chính
plt.suptitle("Scatter Plots of Pairwise Principal Components", y=1.02)
plt.show()

# Dimensionality reduction
# Using LDA method
lda = LDA(n_components=2)
x_data_lda = lda.fit_transform(x_data, y_data)
# Data visualization (2d scatter plot)
pc1 = x_data_lda[:, 0]
pc2 = x_data_lda[:, 1]
# Creat scatter plot
custom_palette = {'red': "red", 'yellow': "yellow", 'green': "green", 'unknown': "purple", 'orange': "orange"}
plot_lda = sns.scatterplot(x=pc1, y=pc2, hue=y_data, palette=custom_palette)
# Rename axis
plot_lda.set(xlabel="First Principal Component", ylabel="Second Principal Component")
plt.show()
