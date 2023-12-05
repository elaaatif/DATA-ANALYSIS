import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# Load Data from CSV
df = pd.read_csv('HPI_master.csv')

# Identify numerical and non-numerical columns
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
non_numerical_columns = df.select_dtypes(exclude=['float64', 'int64']).columns

# Bivariate analysis for numerical-categorical relationships
for num_column in numerical_columns:
    for cat_column in non_numerical_columns:
        box_plot_title = f'Box Plot of {num_column} by {cat_column}'
        sns.boxplot(x=df[cat_column], y=df[num_column])
        plt.title(box_plot_title)
        plt.xlabel(cat_column)
        plt.ylabel(num_column)
        plt.show()

        # T-test between the two categories
        category_1 = df[df[cat_column] == 'Category_1'][num_column]
        category_2 = df[df[cat_column] == 'Category_2'][num_column]

        t_statistic, p_value = ttest_ind(category_1, category_2)

        print(f'T-test results for {num_column} between Category_1 and Category_2:')
        print(f'T-statistic: {t_statistic}')
        print(f'P-value: {p_value}\n')
