import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
X = StandardScaler().fit_transform(X)
X_pca = PCA(n_components=2).fit_transform(X)
df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df['target'] = y
colors = ['red', 'magenta', 'green']
for i, name in enumerate(target_names):
    plt.scatter(df[df.target == i]['PC1'], df[df.target == i]['PC2'], label=name, alpha=0.6, color=colors[i])

plt.title('PCA of Iris Dataset (2 Components)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(title='Target')
plt.grid(alpha=0.3)
plt.show()
