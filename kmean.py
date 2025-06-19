import numpy as np 
# import tensorflow as tf 
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
pd.set_option('display.max_columns', None)  # Hiện tất cả các cột
pd.set_option('display.width', None)        # Không giới hạn chiều ngang
pd.set_option('display.max_colwidth', None) # Hiển thị toàn bộ nội dung cột (nếu có text dài)
sns.set_style("whitegrid")
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

path = 'bank_transactions_data.csv'
df = pd.read_csv(path)

#Chỉ lấy các trường đặc trưng
numerical_columns = df.describe().columns.tolist()
df_numerical = df[numerical_columns]
df_numerical = df.describe().columns.tolist()

def Elbow(df_numerical):
    # Tiền xử lý dữ liệu số
    X_num = df[df_numerical]
    X_num = X_num.fillna(X_num.median())
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)
    # Dùng trực tiếp X_num_scaled cho KMeans
    X_combined = X_num_scaled  # KHÔNG thêm one-hot nữa

    # Elbow method
    wcss = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_combined)
        wcss.append(kmeans.inertia_)

    # Vẽ biểu đồ elbow
    plt.plot(range(1, 10), wcss, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("WCSS")
    plt.title("Elbow Method using Only Numerical Features")
    plt.grid(True)
    plt.show()

X = df[df_numerical]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

df['KMeans_Cluster'] = kmeans_labels

distances = np.linalg.norm(X_scaled - kmeans.cluster_centers_[kmeans_labels], axis=1)

threshold = np.percentile(distances, 95)  

df['Potential_Fraud'] = distances > threshold

frauds = df[df['Potential_Fraud']]
non_frauds = df[~df['Potential_Fraud']]

#Vẽ đồ thị cụm
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=kmeans_labels, palette='viridis', s=60, alpha=0.5)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', label='Centroids')
plt.scatter(X_scaled[distances > threshold, 0], X_scaled[distances > threshold, 1], 
            color='black', s=50, label='Potential Frauds', marker='X')
plt.title('K-means Clustering with Potential Frauds Highlighted')
plt.xlabel('Scaled Amount')
plt.ylabel('Scaled Age')
plt.legend()
plt.show()

print(f"Số giao dịch bất thường : {len(frauds)}")
frauds.to_csv("potential_frauds.csv", index=False)

for i in frauds.index:
    cluster_id = df.loc[i, 'KMeans_Cluster']
    centroid = kmeans.cluster_centers_[cluster_id]
    scaled_values = X_scaled[i]
    diff = scaled_values - centroid
    contrib = pd.Series(np.abs(diff), index=numerical_columns)
    top_features = contrib.sort_values(ascending=False).head(3)
    print(f"\ngiao dịch thứ: {i}:")
    print(top_features)
