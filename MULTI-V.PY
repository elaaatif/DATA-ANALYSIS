import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Load Data from CSV
df = pd.read_csv('HPI_master.csv')

# Identify numerical and non-numerical columns
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
non_numerical_columns = df.select_dtypes(exclude=['float64', 'int64']).columns

# Impute missing values
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df[numerical_columns]), columns=numerical_columns)

# Multivariate Analysis
# Correlation Matrix (excluding non-numeric columns)
correlation_matrix = df_imputed.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Pairplot (excluding non-numeric columns)
sns.pairplot(df_imputed, diag_kind='kde')
plt.show()

# PCA for dimensionality reduction
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(df_imputed)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
plt.title('PCA of numerical columns')
plt.show()
