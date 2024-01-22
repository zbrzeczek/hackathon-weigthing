import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Replace 'your_excel_file.xlsx' with the actual file patht
train_file_path = 'train_fixed.csv'
test_file_path = 'test_fixed.csv'

# Read Excel file into a DataFrame
train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

# Display the DataFrame (optional)
#print(df)

sector_column = 'Sector'
target_column = 'Rating'
other_variable_columns = ['Liquidity1', 'Liquidity2', 'Liquidity3',
                          'Profitability1', 'Profitability2', 'Profitability3', 'Profitability4', 'Profitability5', 'Profitability6', 'Profitability7',
                          'Efficiency1', 'Efficiency2', 'Debt1', 'Debt2', 'TaxRate', 'CashFlow1', 'CashFlow2', 'CashFlow3',
                          'Expectations1', 'Profitability1', 'Expectations2', 'CashFlow4', 'CashFlow5', 'Efficiency3']

train_x = pd.get_dummies(train_df[other_variable_columns + [sector_column]])
test_x = pd.get_dummies(test_df[other_variable_columns + [sector_column]])
y_labels = train_df[target_column]
unique_sectors = train_df[sector_column].unique()
print("Number of samples in training set:", len(train_x))
print("Number of samples in testing set:", len(test_x))
# Display the unique strings
print("Unique Economy Sectors:")
for sector in unique_sectors:
    print(sector)
# Use LabelEncoder to convert categorical labels to numeric values
label_encoder = LabelEncoder()
label_mapping = {'Low Risk': 0, 'Medium Risk': 1, 'High Risk': 2, 'Highest Risk': 3}
train_y_numeric = [label_mapping[label] for label in y_labels]


# Train a RandomForestRegressor model
model = RandomForestRegressor()
model.fit(train_x, train_y_numeric)

# Make predictions on the test set
y_pred = model.predict(test_x)

print(y_pred)
# Decode the numeric predictions back to original labels
y_pred_rounded = pd.Series(y_pred).round().astype(int)

print(y_pred_rounded)
# Create a confusion matrix
##conf_matrix = confusion_matrix(test, y_pred_rounded)
#print(conf_matrix)
#error_weights = [[0,1,2,4],[2,0,1,2],[4,2,0,1],[8,4,2,0]]
#print(error_weights)
#unique_labels = sorted(pd.unique(train_y_numeric))

#weighted_conf_matrix = conf_matrix * error_weights
# Create a sorted confusion matrix
#conf_matrix_df = pd.DataFrame(conf_matrix, index=unique_labels, columns=unique_labels)
#weight_matrix_df = pd.DataFrame(conf_matrix, index=unique_labels, columns=unique_labels)

#print("Confusion Matrix:")
#print(conf_matrix_df)

print(weighted_conf_matrix)

total = weighted_conf_matrix.sum()

print(total)