import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, roc_curve, roc_auc_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVR, SVC
from sklearn.feature_selection import SelectKBest, f_regression, f_classif

# Load dataset
df = pd.read_csv("variant1.csv")

# Drop the date column from the dataset
df_num = df.drop(["date"], axis=1)

# Create correlation matrix to find which columns are related to price
corr_matrix = df_num.corr()

# Check correlation values of each column with the price column
no_corr = []
for col in corr_matrix:
    if col == "price":
        continue
    corr_value = corr_matrix.loc[col, "price"]
    if abs(corr_value) < 0.1:
        print(col + " has no correlation with price. {:.2f}".format(corr_value))
        no_corr.append(col)
    elif corr_value < 0:
        print(col + " has an inverse correlation with price. {:.2f}".format(corr_value))
    else:
        print(col + " has positive correlation with price. {:.2f}".format(corr_value))

# Append "price" to the no_corr list to include it in the X dataset
no_corr.append("price")
print(no_corr.__len__())

# Split the dataset into training and test sets
X = df_num.drop("price", axis=1)
y = df_num["price"]
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.4, random_state=42)

# Select the top k features based on correlation with price
selector = SelectKBest(score_func=f_regression, k=15)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Get the column indices of the selected features
selected_cols = selector.get_support(indices=True)

# Train the linear regression model on the selected features
lr = LinearRegression()
lr.fit(X_train_selected, y_train)
y_pred_lr = lr.predict(X_test_selected)

# Calculate the evaluation metrics for the linear regression model
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)

# Calculate R2 score
r2 = r2_score(y_test, y_pred_lr)
print("Linear Regression R2 Score:", r2)

# Perform 5-fold cross-validation for the linear regression model
cv_scores = cross_val_score(lr, X, y, cv=5)
avg_cv_score = np.mean(cv_scores)
print("Linear Regression 5-Fold Cross-Validation R^2 Score:", avg_cv_score)

# Print the selected features and their coefficients
selected_features = X.columns[selected_cols]
print("Selected features:", selected_features)
print("Coefficients: \n", lr.coef_)

# Print the evaluation metrics
print("Linear Regression MAE:", mae_lr)
print("Linear Regression MSE:", mse_lr)
print("Linear Regression RMSE:", rmse_lr)

# Fit a first-order polynomial to the data
p = np.polyfit(y_test, y_pred_lr, 1)
trendline = np.polyval(p, y_test)

# Create a scatter plot of predicted vs actual prices
plt.scatter(y_test, y_pred_lr)
plt.plot(y_test, trendline, color='red', label='Trendline')
plt.plot(y_test, y_test, 'k--', lw=2, label='Perfect predictions') # line of perfect predictions
plt.xlabel("Actual prices")
plt.ylabel("Predicted prices")
plt.title("Linear Regression: Predicted vs Actual House Prices")
plt.legend(loc='lower right')
plt.show()

# Drop unnecessary columns from the dataset
X = df_num.drop(['id', 'sqft_lot', 'condition', 'yr_built', 'zipcode', 'long', 'sqft_lot15',"price"], axis=1)
y = df_num["price"]

# Preprocessing: Classify house prices into 2 groups: >= median and < median
price_median = df_num["price"].median()
X = df_num.drop(['price'], axis=1)
y = df_num["price"].apply(lambda x: 1 if x >= price_median else 0)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

# Perform univariate feature selection with SelectKBest
selector = SelectKBest(score_func=f_classif, k=15)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Get the selected feature names
selected_feature_names = X.columns[selector.get_support()].tolist()

# Print the selected feature names
print("Selected Features:", selected_feature_names)

# Train logistic regression with the selected features
lr = LogisticRegression(max_iter=1000)

# Perform 5-fold cross-validation for logistic regression
cv_scores = cross_val_score(lr, X_train_selected, y_train, cv=5)
avg_cv_score = np.mean(cv_scores)
print("Logistic Regression 5-Fold Cross-Validation Accuracy:", avg_cv_score)

# Fit the logistic regression model with the selected features
lr.fit(X_train_selected, y_train)
y_pred_lr = lr.predict(X_test_selected)

# Calculate the accuracy of the logistic regression model
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Accuracy with Selected Features:", accuracy_lr)

# Compute the predicted probabilities for the positive class
y_pred_prob_lr = lr.predict_proba(X_test_selected)[:, 1]

# Compute the ROC curve and AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_lr)
auc_score = roc_auc_score(y_test, y_pred_prob_lr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='Logistic Regression (AUC = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

