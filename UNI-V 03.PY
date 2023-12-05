import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data from CSV
df = pd.read_csv('HPI_master.csv')

# Print column names
print(df.columns)

# Print unique values for 'yr' column
print(df['yr'].unique())

# Identify numerical and non-numerical columns
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
non_numerical_columns = df.select_dtypes(exclude=['float64', 'int64']).columns

# Univariate analysis for numerical columns
for column in numerical_columns:
    descriptive_stats = df[column].describe()

    frequency_table = pd.DataFrame(df[column].value_counts())
    cumulative_frequency_table = frequency_table.cumsum()

    effective_frequency_table = pd.DataFrame({
        'Effective': frequency_table[column],
        'Cumulative Effective': cumulative_frequency_table[column],
        'Frequency': frequency_table[column],
        'Cumulative Frequency': cumulative_frequency_table[column],
    })

    print(f'Descriptive Statistics for {column}:\n{descriptive_stats}\n')
    print(f'Table for {column}:\n{effective_frequency_table}\n')

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.hist(df[column], bins='auto', alpha=0.7, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

    plt.subplot(2, 2, 2)
    sns.boxplot(x=df[column], color='lightcoral')
    plt.title(f'Box Plot of {column}')

    plt.subplot(2, 2, 3)
    sns.kdeplot(df[column], fill=True, color='lightgreen')
    plt.title(f'Density Plot of {column}')
    plt.xlabel(column)
    plt.ylabel('Density')

    plt.tight_layout()
    plt.show()

# Univariate analysis for non-numerical columns
for column in non_numerical_columns:
    frequency_table = pd.DataFrame(df[column].value_counts())
    cumulative_frequency_table = frequency_table.cumsum()

    effective_frequency_table = pd.DataFrame({
        'Effective': range(1, len(frequency_table) + 1),
        'Cumulative Effective': cumulative_frequency_table[column],
        'Frequency': frequency_table[column],
        'Cumulative Frequency': cumulative_frequency_table[column],
    })

    print(f'Table for {column}:\n{effective_frequency_table}\n')

    # Plotting bar plot for non-numerical data
    plt.figure(figsize=(10, 5))
    sns.countplot(x=column, data=df, order=df[column].value_counts().index, palette='viridis')
    plt.title(f'Bar Plot of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    plt.show()
