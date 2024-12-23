import pandas as pd, seaborn as sns, matplotlib.pyplot as plt, numpy as np, time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import silhouette_score, davies_bouldin_score

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
data.drop(['id', 'date', 'title', 'url', 'detailUrl', 'ids', 'sources',
           'types', 'net', 'code', 'geometryType', 'status',
           'postcode', 'what3words', 'locationDetails'], axis=1, inplace=True)

# Replace Null value of column 'alert' to unknown
data.fillna({'alert': 'unknown'}, inplace=True)

# NA-value columns will be filled by 'unknown' value
na_column = data.columns[data.isna().any()].tolist()

for column in na_column:
    data.fillna({column: 'unknown'}, inplace=True)

# String-value columns will be one-hot coding
string_columns = data.select_dtypes(include=['object']).columns.tolist()
string_columns.remove('alert')

# One-hot coding
for column in string_columns:
    dummies = pd.get_dummies(data[column], prefix='type')
    data.drop([column], axis=1, inplace=True)
    data = pd.concat([data, dummies], axis=1)

# Sample set x_data and label set y_data
x_data = data.drop(['alert'], axis=1)
x_data = StandardScaler().fit_transform(x_data)
y_data = data['alert']

# Dimensionality reduction
# Using PCA method
pca_plot = PCA(n_components=3, random_state=42)
x_data_pca_plot = pca_plot.fit_transform(x_data)

# Data visualization
pc_x = x_data_pca_plot[:, 0]
pc_y = x_data_pca_plot[:, 1]
pc_z = x_data_pca_plot[:, 2]
# Reset color
custom_palette = {'red': "red", 'yellow': "yellow", 'green': "green", 'unknown': "purple", 'orange': "orange"}

# Creat 2D scatter plot
plot_pca = sns.scatterplot(x=pc_x, y=pc_y, hue=y_data, palette=custom_palette)
# Rename axis
plot_pca.set(xlabel="First Principal Component", ylabel="Second Principal Component")
plt.show()

# Tạo biểu đồ 3D
fig_pca = plt.figure(figsize=(10, 7))
ax_pca = fig_pca.add_subplot(111, projection='3d')

# Vẽ scatter plot
colors = [custom_palette[label] for label in y_data]
sc_pca = ax_pca.scatter(pc_x, pc_y, pc_z, c=colors, alpha=0.8)

# Đặt tên trục
ax_pca.set_xlabel("First Principal Component")
ax_pca.set_ylabel("Second Principal Component")
ax_pca.set_zlabel("Third Principal Component")

# Tạo chú thích màu sắc
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
           for label, color in custom_palette.items()]
ax_pca.legend(handles=handles, loc='best')

# Hiển thị biểu đồ
plt.show()

# Print the explained variance after reduce dimension of the data
print(f'Explained variance: {np.cumsum(pca_plot.explained_variance_ratio_)}')
# Print the statistic values of Principal Component
pd.DataFrame(data=x_data_pca_plot,
             columns=['First Principal Component', 'Second Principal Component',
                      'Third Principal Component']).describe()

# Apply PCA with 4 components
pca_pair = PCA(n_components=4)
x_data_pca_pair = pca_pair.fit_transform(x_data)

# Create a DataFrame for the PCA data
data_pca_pair = pd.DataFrame(x_data_pca_pair, columns=[f'PC{i + 1}' for i in range(4)])
data_pca_pair['target'] = y_data

# Plot the pairplot
plt.figure(figsize=(15, 10))
sns.pairplot(data_pca_pair, hue='target', diag_kind='kde',
             plot_kws={'alpha': 0.6, 's': 50, 'edgecolor': 'k'},
             vars=[f'PC{i + 1}' for i in range(4)], palette=custom_palette)
plt.suptitle("Scatter Plots of Pairwise Principal Components", y=1.02)
plt.show()

# Create a copy of data to store the clustering results
data_gmm = data.copy()
dummies = pd.get_dummies(data_gmm['alert'], prefix='type')
data_gmm.drop(['alert'], axis=1, inplace=True)
data_gmm = pd.concat([data_gmm, dummies], axis=1)
pd.DataFrame(data_gmm)

# Train the model
gmm = GaussianMixture(n_components=5, random_state=42)
gmm.fit(data_gmm)

# Predict the labels
labels_gmm = gmm.predict(data_gmm)
# Thêm cột nhãn cụm vào dữ liệu
data_gmm = np.hstack((data, labels_gmm.reshape(-1, 1)))

# Visualize the results
plt.scatter(data_gmm[:, 0], data_gmm[:, 1], c=labels_gmm, cmap='viridis')
plt.title('Clustering with Gaussian Mixture Model')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()


# Function to split the training set and validation set
def split_data(x_data, y_data, test_size):
    x_train, x_test, y_train, y_test = train_test_split(x_data,
                                                        y_data,
                                                        test_size=test_size,
                                                        random_state=42)
    return x_train, x_test, y_train, y_test


def evaluate_model(start_time, end_time, y_test, y_pred):
    print(f'Training time: {end_time - start_time}s')
    print(f'Accuracy score: {accuracy_score(y_test, y_pred)}')
    print(f'Recall score: {recall_score(y_test, y_pred, average='macro')}')
    print(f'Precision score: {precision_score(y_test, y_pred, average='macro', zero_division=0)}')


# This function uses the Naive Bayes classifier approach
def naive_bayes_approach(x_train, x_test, y_train, y_test):
    # Train model
    start_time = time.perf_counter()
    model = GaussianNB()
    model.fit(x_train, y_train)
    end_time = time.perf_counter()

    # Predict the result
    y_pred = model.predict(x_test)

    # Evaluate the model
    evaluate_model(start_time, end_time, y_test, y_pred)


# This function uses the Multinomial Logistic Regression approach
def softmax_approach(x_train, x_test, y_train, y_test):
    # Train model
    start_time = time.perf_counter()
    model = LogisticRegression(max_iter=5000, solver='saga')
    model.fit(x_train, y_train)
    end_time = time.perf_counter()

    # Predict the result
    y_pred = model.predict(x_test)

    # Evaluate the model
    evaluate_model(start_time, end_time, y_test, y_pred)


def classification(x_train, x_test, y_train, y_test):
    print('Naive Bayes approach')
    naive_bayes_approach(x_train, x_test, y_train, y_test)
    print('\nSoftmax approach')
    softmax_approach(x_train, x_test, y_train, y_test)


# This function classifies based on the original data
def original_data(x_data, y_data, test_size):
    x_train, x_test, y_train, y_test = split_data(x_data, y_data, test_size)
    print(f'Phân loại dựa trên dữ liệu gốc với tỷ lệ train:test là: {int((1 - test_size) * 10)}:{int(test_size * 10)}')
    classification(x_train, x_test, y_train, y_test)


# This function first reduces dimensionality, then splits the data into train and validation sets, and finally classifies
def dim_reduction_split(x_data, y_data, test_size, dim_reduction_type, dim):
    if (dim_reduction_type == 'pca'):
        pca = PCA(n_components=dim, random_state=42)
        x_data_dim_reduction = pca.fit_transform(x_data)
    elif (dim_reduction_type == 'lda'):
        lda = LDA(n_components=dim)
        x_data_dim_reduction = lda.fit_transform(x_data, y_data)

    x_train, x_test, y_train, y_test = split_data(x_data_dim_reduction, y_data, test_size)
    print(
        f'Phân loại dựa trên dữ liệu giảm chiều rồi chia tỷ lệ train:test là: {int((1 - test_size) * 10)}:{int(test_size * 10)}')
    classification(x_train, x_test, y_train, y_test)


# This function first splits the data into train and validation sets, then reduces dimensionality, and finally classifies
def split_dim_reduction(x_data, y_data, test_size, dim_reduction_type, dim):
    x_train, x_test, y_train, y_test = split_data(x_data, y_data, test_size)

    if (dim_reduction_type == 'pca'):
        pca = PCA(n_components=dim, random_state=42)
        x_train_dim_reduction = pca.fit_transform(x_train)
        x_test_dim_reduction = pca.transform(x_test)
    elif (dim_reduction_type == 'lda'):
        lda = LDA(n_components=dim)
        x_train_dim_reduction = lda.fit_transform(x_train, y_train)
        x_test_dim_reduction = lda.transform(x_test)

    print(
        f'Phân loại dựa trên dữ liệu được chia tỷ lệ train:test là: {int((1 - test_size) * 10)}:{int(test_size * 10)} rồi giảm chiều')
    classification(x_train_dim_reduction, x_test_dim_reduction, y_train, y_test)


dim_pca = x_data.shape[1] // 3

original_data(x_data, y_data, test_size=0.2)

original_data(x_data, y_data, test_size=0.3)

original_data(x_data, y_data, test_size=0.4)

dim_reduction_split(x_data, y_data, dim_reduction_type='pca', test_size=0.2, dim=dim_pca)

dim_reduction_split(x_data, y_data, dim_reduction_type='pca', test_size=0.3, dim=dim_pca)

dim_reduction_split(x_data, y_data, dim_reduction_type='pca', test_size=0.4, dim=dim_pca)

split_dim_reduction(x_data, y_data, dim_reduction_type='pca', test_size=0.2, dim=dim_pca)

split_dim_reduction(x_data, y_data, dim_reduction_type='pca', test_size=0.3, dim=dim_pca)

split_dim_reduction(x_data, y_data, dim_reduction_type='pca', test_size=0.4, dim=dim_pca)
