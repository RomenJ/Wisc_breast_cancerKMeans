from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

dataFull=pd.read_csv('breastCancer.csv')

dataFull= dataFull.drop(columns=['Unnamed: 32'])
data=dataFull[['radius_worst', 'concave points_mean', 'radius_mean','area_mean','diagnosis', 'concavity_worst']]

print("Values Full+++++:")
print(dataFull.value_counts())
print("Datafull:")
print(dataFull.head(5))
print("Shape:")
print(data.shape)
print("Info:")
print(data.info())
data.dropna()

data['diagNum'] = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)
dataFull['diagNum'] = dataFull['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)


# Conteo de valores nulos por columna
null_counts = data.isna().sum()

X = dataFull.select_dtypes(include=['int', 'float'])
print("columnas de X")
print(X.columns)
xPrev=X
selected_features = xPrev[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                          'smoothness_mean', 'compactness_mean', 'concavity_mean',
                          'concave points_mean', 'symmetry_mean',
                           'radius_worst', 'texture_worst', 'diagNum', 'concavity_worst']]

# Crear una nueva matriz de correlación con las variables seleccionadas
correlation_matrix_selected = selected_features.corr()


# Crear mapa de calor
plt.figure(figsize=(10, 12))
heatmap = sns.heatmap(correlation_matrix_selected, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=30,fontname='Arial', fontsize=10)
heatmap.set_yticklabels(heatmap.get_yticklabels(),fontname='Arial', fontsize=10)
plt.title('Matriz de Correlación de las variables del dataset (Means)')
plt.savefig('MatrizdeCorrelaciones.png')
plt.show()

print(data.head(5))

#scatterplot 1
colores_geniales0 = ["#3366FF80", "#FF573380"]
# Grafica el scatterplot con los colores personalizados
sns.scatterplot(data=data, x="radius_worst", y="concave points_mean" , hue="diagNum", size="radius_mean", palette=colores_geniales0, alpha=0.5 )
plt.title("Relacion entre radius_worst y concave points_mean (0=Benigno, 1=maligno). Tamaño: radius_mean")
plt.savefig('scatterplot_1.png')
plt.show()

#scatterplot 2


colores_geniales1 = ["#3366FF", "#FF5733",]
sns.scatterplot(data=data, x="concavity_worst", y="concave points_mean", hue="diagNum", size="area_mean", palette=colores_geniales1, alpha=0.5)
plt.title("Relacion entre concavity_worst y concave points_mean (0=Benigno, 1=maligno)")
plt.savefig('scatterplot_2.png')
plt.show()

NumData = data.select_dtypes(include=['float64', 'int64'])
data=NumData
print("Data NUMDATA")
print(data.head(5))

scaler=StandardScaler()
data_scaled=scaler.fit_transform(NumData)

print("Data Scaled")
print(data_scaled)

silhouette_scores=[]
for k in range (2,11):
    print("For:", k )
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(data_scaled)
    score=silhouette_score(data_scaled, kmeans.labels_)
    silhouette_scores.append(score)

plt.plot(range(2,11),silhouette_scores, marker='o' )    
plt.xlabel('Número de clusters')
plt.ylabel('Coeficiente de silhouette')
plt.title('Coeficientes k')
plt.savefig('Coeficientes_k.png')
plt.show()

k=2
kmeans=KMeans(n_clusters=k, init='k-means++', random_state=42)
kmeans.fit(data_scaled)
labels=kmeans.labels_
centroids=kmeans.cluster_centers_

plt.scatter(data_scaled[:,0], data_scaled[:,1] ,c=labels, s=50, cmap='viridis', alpha=0.8)
plt.scatter(centroids[:,0], centroids[:, 1], marker='x', s=200, c='red', alpha=0.8)
plt.ylabel('concave points_mean')
plt.xlabel('Radius_worst')
plt.title('Centroides de los Kmeans de "concave points_mean" y "radius_worst" ')
plt.savefig('Clasificacion01.png')
plt.show()

plt.scatter(data_scaled[:,1], data_scaled[:,3] ,c=labels, s=50, cmap='coolwarm', alpha=0.8)
plt.scatter(centroids[:,1], centroids[:, 3], marker='x', s=200, c='black', alpha=0.8)
plt.xlabel('concave points_mean')
plt.ylabel('area_mean')
plt.title('Centroides 2 de los Kmeans de "concave points_mean"  y "area_mean"')
plt.savefig('Clasificacion02.png')
plt.show()

summaryfull = dataFull.describe()
summarydata = data.describe()

# Mostrar solo las filas que contienen los valores mínimo y máximo
min_max_valuesfull = summaryfull.loc[['min', 'max']]
print(min_max_valuesfull)
min_max_valuesdata = summarydata.loc[['min', 'max']]
print(min_max_valuesdata)



