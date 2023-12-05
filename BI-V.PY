import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data from CSV
df = pd.read_csv('HPI_master.csv')

# Identify numerical and non-numerical columns
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
non_numerical_columns = df.select_dtypes(exclude=['float64', 'int64']).columns

# Bivariate analysis for numerical-numerical relationships
for i in range(len(numerical_columns)):
    for j in range(i + 1, len(numerical_columns)):
        scatter_plot_title = f'Scatter Plot of {numerical_columns[i]} vs {numerical_columns[j]}'
        plt.scatter(df[numerical_columns[i]], df[numerical_columns[j]])
        plt.title(scatter_plot_title)
        plt.xlabel(numerical_columns[i])
        plt.ylabel(numerical_columns[j])
        plt.show()

        # Pairplot for multiple numerical columns
        numerical_data = df[[numerical_columns[i], numerical_columns[j]]]
        sns.pairplot(numerical_data)
        plt.show()

# Bivariate analysis for numerical-categorical relationships
for num_column in numerical_columns:
    for cat_column in non_numerical_columns:
        box_plot_title = f'Box Plot of {num_column} by {cat_column}'
        sns.boxplot(x=df[cat_column], y=df[num_column])
        plt.title(box_plot_title)
        plt.xlabel(cat_column)
        plt.ylabel(num_column)
        plt.show()

# Bivariate analysis for categorical-categorical relationships
for i in range(len(non_numerical_columns)):
    for j in range(i + 1, len(non_numerical_columns)):
        crosstab = pd.crosstab(df[non_numerical_columns[i]], df[non_numerical_columns[j]])
        sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Heatmap of {non_numerical_columns[i]} vs {non_numerical_columns[j]}')
        plt.xlabel(non_numerical_columns[j])
        plt.ylabel(non_numerical_columns[i])
        plt.show()
