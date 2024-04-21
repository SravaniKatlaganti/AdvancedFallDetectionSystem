import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Simulated metrics for the models
data = {
    "Model": ["SVM", "Random Forest", "LSTM"],
    "Accuracy": [0.9913, 0.9920, 0.9950],
    "Precision": [0.99, 0.99, 1.00],
    "Recall": [0.99, 0.99, 1.00],
    "F1 Score": [0.99, 0.99, 1.00]
}

df = pd.DataFrame(data)
df.set_index("Model", inplace=True)

# TABLE
table = df.copy()
# format the Accuracy column as a percentage
table['Accuracy'] = table['Accuracy'].map('{:.2%}'.format)
# display the formatted dataframe
print(table)

# BAR PLOTS
# Setting the style
plt.style.use('seaborn-darkgrid')

# Creating subplots
fig, ax = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Performance Metrics Comparison Among SVM, Random Forest, and LSTM', fontsize=16)

sns.barplot(ax=ax[0, 0], x=df.index, y=df['Accuracy'], palette='viridis')
ax[0, 0].set_title('Accuracy Comparison')
ax[0, 0].set_ylim(0.90, 1.00)

sns.barplot(ax=ax[0, 1], x=df.index, y=df['Precision'], palette='viridis')
ax[0, 1].set_title('Precision Comparison')
ax[0, 1].set_ylim(0.90, 1.00)

sns.barplot(ax=ax[1, 0], x=df.index, y=df['Recall'], palette='viridis')
ax[1, 0].set_title('Recall Comparison')
ax[1, 0].set_ylim(0.90, 1.00)

sns.barplot(ax=ax[1, 1], x=df.index, y=df['F1 Score'], palette='viridis')
ax[1, 1].set_title('F1 Score Comparison')
ax[1, 1].set_ylim(0.90, 1.00)

# Automatically adjust subplot params so that the subplot(s) fits in to the figure area.
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Show plot
plt.show()

# HEATMAP
plt.figure(figsize=(10, 6))
sns.heatmap(df, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Heatmap of Model Metrics')
plt.show()