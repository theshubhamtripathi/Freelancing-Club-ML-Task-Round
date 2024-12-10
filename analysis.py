# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Step 1: Cleaned dataset (mock data, replace with actual data if available)
data_cleaned = pd.DataFrame({
    'Category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B'],
    'Food_Description': ['Food1', 'Food2', 'Food3', 'Food4', 'Food5', 'Food6', 'Food7', 'Food8'],
    'N': [10, 15, 14, 16, 10, 13, 18, 15],
    'Adenine': [0.5, 0.7, 0.6, 0.8, 0.5, 0.9, 0.8, 0.7],
    'Guanine': [0.3, 0.4, 0.35, 0.45, 0.4, 0.38, 0.46, 0.41],
    'Hypoxanthine': [0.1, 0.15, 0.12, 0.16, 0.13, 0.11, 0.14, 0.15],
    'Xanthine': [0.05, 0.07, 0.06, 0.08, 0.05, 0.09, 0.08, 0.07],
    'Total': [11, 16.3, 15.1, 17.5, 10.93, 14.49, 18.78, 16.33]
})

# Task 1: K-Means Clustering
numeric_data = data_cleaned.select_dtypes(include=np.number)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(numeric_data)
data_cleaned['Cluster'] = clusters

# Evaluate clustering
silhouette_avg = silhouette_score(numeric_data, clusters)

# Save Task 1 visualization to PDF
pdf_path_task1 = "Task1_Visualization.pdf"
with PdfPages(pdf_path_task1) as pdf:
    for col in numeric_data.columns:
        plt.figure(figsize=(6, 4))
        plt.hist(numeric_data[col], bins=10, color='skyblue', edgecolor='black')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        pdf.savefig()
        plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(numeric_data.iloc[:, 0], numeric_data.iloc[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    plt.title('Cluster Visualization')
    plt.xlabel(numeric_data.columns[0])
    plt.ylabel(numeric_data.columns[1])
    pdf.savefig()
    plt.close()

# Task 2: Imputation and MLP Training
data_imputed = data_cleaned.copy()
data_imputed.loc[2, 'Total'] = np.nan
data_imputed.loc[5, 'Adenine'] = np.nan

imputer = KNNImputer(n_neighbors=3)
imputed_data = imputer.fit_transform(data_imputed.select_dtypes(include=np.number))
data_imputed[data_imputed.select_dtypes(include=np.number).columns] = imputed_data

X = data_imputed.drop(columns=['Total', 'Category', 'Food_Description', 'Cluster'])
y = data_imputed['Total']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlp = MLPRegressor(hidden_layer_sizes=(50, 25), activation='relu', solver='adam', max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Save Task 2 visualization to PDF
pdf_path_task2 = "Task2_Visualization.pdf"
with PdfPages(pdf_path_task2) as pdf:
    plt.figure(figsize=(6, 4))
    plt.plot(mlp.loss_curve_)
    plt.title('MLP Training Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color='blue', edgecolor='k')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)
    plt.title('Actual vs Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    pdf.savefig()
    plt.close()

print("Task 1 PDF:", pdf_path_task1)
print("Task 2 PDF:", pdf_path_task2)
print("Silhouette Score:", silhouette_avg)
print("RMSE:", rmse)
