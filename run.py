import numpy as np
import matplotlib.pyplot as plt

from helpers import *
from implementations import *

# Load data

input("Hello")

data_path = 'dataset/'

# The dataset is subsampled to make it run faster
x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path, sub_sample=True)
print("x_train :", x_train.shape )
print("y_train :", y_train.shape )
print("x_test :", x_test.shape )

# Checking composition of dataset labels
print("Proportion of unhealthy :", np.sum(y_train == 1)/len(y_train) * 100, "%")
x_train_resampled, y_train_resampled = downsampling(x_train,y_train)

# Feature selection
x_train_clean,filter= data_cleaning(x_train_resampled)
X_train,continuous_columns,categorical_columns,mean_x,std_x,unique_categories = data_normalize(x_train_clean,split_type=True)

# Model Fitting and Comparison
tx_tr = generate_tx(X_train)
initial_w = initialize_weight(tx_tr,seed=123)

y_train_resampled = (1+y_train_resampled)/2

# Grid search looking for best gamma and lambda
def grid_search_gamma(y,tx,initial_w,method,gammas = np.array([0.001,0.01,0.1,0.5])):

    loss_tr = []

    for gamma in gammas:
        if method == 'GD':
            _, loss = mean_squared_error_gd(y, tx, initial_w, max_iters=10000, gamma=gamma)
        elif method == 'SGD':
            _, loss = mean_squared_error_sgd(y, tx, initial_w, max_iters=10000, gamma=gamma)
        elif method == 'LogisticRegression':
            _, loss = logistic_regression(y, tx, initial_w, max_iters=1000, gamma=gamma)
        elif method == 'RegLogisticRegression':
            _, loss = reg_logistic_regression(y=y, tx=tx, initial_w=initial_w, max_iters=1000, gamma=gamma, lambda_=1)
        else:
            raise ValueError('Invalid method specified. Choose from "GD", "SGD", "LogisticRegression", or "RegLogisticRegression".')

        if np.isnan(loss):
            print(f"NaN loss detected at iteration for gamma={gamma}")
            break

        loss_tr.append(loss)

    best_loss = np.min(loss_tr)
    best_gamma = gammas[np.argmin(loss_tr)]

    print(f'Best gamma for {method}: {best_gamma}, loss: {best_loss}')

    return best_gamma

gamma_gd = grid_search_gamma(y_train_resampled,tx_tr,initial_w,method='GD')
gamma_sgd = grid_search_gamma(y_train_resampled,tx_tr,initial_w,method='SGD')
gamma_lr = grid_search_gamma(y_train_resampled,tx_tr,initial_w,method='LogisticRegression')
gamma_rlr = grid_search_gamma(y_train_resampled,tx_tr,initial_w,method='RegLogisticRegression')

def grid_search_lambda(y, tx, initial_w=None, method = 'RidgeRegression', lambdas=np.array([0.01,0.05,0.1,0.2,0.5,1, 10]),gamma = None):
    loss_tr = []

    for lambda_ in lambdas:
        if method == 'RidgeRegression':
            _, loss = ridge_regression(y, tx, lambda_=lambda_)
        elif method == 'RegLogisticRegression':
            _, loss = reg_logistic_regression(y=y, tx=tx, initial_w=initial_w, max_iters=1000, gamma= gamma,lambda_=lambda_)
        else:
            print('ValueError: Method not supported')
            continue

        # Check for NaN loss
        if np.isnan(loss):
            print(f"Warning: NaN loss for lambda={lambda_} in method={method}.")
            loss_tr.append(np.inf)  # Assign a large number to avoid selecting this lambda
        else:
            loss_tr.append(loss)

    best_loss = np.min(loss_tr)
    best_lambda = lambdas[np.argmin(loss_tr)]

    print(f'Best lambda for {method}: {best_lambda}, loss: {best_loss}')

    return best_lambda

lambda_rr = grid_search_lambda(y_train_resampled,tx=tx_tr,method='RidgeRegression')
lambda_rlr = grid_search_lambda(y_train_resampled,tx=tx_tr,method='RegLogisticRegression',initial_w=initial_w,gamma=gamma_rlr)

# Training and evaluation in cross-validation settings
def cross_validate(y, tx, k_fold=4,method='GD',initial_w =None):
    """ Perform k-fold cross-validation """
    seed = 12
 
    k_fold = k_fold
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    accuracy_scores = []
    precision_scores = []
    f1_scores = []

    w = initial_w
    
    for k in range(k_fold):
        train_indices = np.ones(tx.shape[0], dtype=bool)
        train_indices[k_indices[k]] = False
        
        x_te = tx[k_indices[k]]
        y_te = y[k_indices[k]]

        x_tr = tx[train_indices]
        y_tr = y[train_indices] 

        if method == 'RidgeRegression':
            w, loss_tr = ridge_regression(y_tr, x_tr, lambda_=lambda_rr)
            predictions_te = ((x_te@w)>=0).astype(int)
        elif method == 'RegLogisticRegression':
            w, loss_tr = reg_logistic_regression(y=y_tr, tx=x_tr, initial_w=w, max_iters=10000, gamma= gamma_rlr,lambda_=lambda_rlr)
            predictions_te = (sigmoid(x_te@w)>=0.5).astype(int)
        elif method == 'GD':
            w, loss_tr = mean_squared_error_gd(y=y_tr, tx=x_tr, initial_w=w, max_iters=1000, gamma=gamma_gd)
            predictions_te = ((x_te@w)>=0).astype(int)
        elif method == 'SGD':
            w, loss_tr = mean_squared_error_sgd(y=y_tr, tx=x_tr, initial_w=w, max_iters=10000, gamma=gamma_sgd)
            predictions_te = ((x_te@w)>=0).astype(int)
        elif method == 'LogisticRegression':
            w, loss_tr = logistic_regression(y=y_tr, tx=x_tr, initial_w=w, max_iters=1000, gamma=gamma_lr)
            predictions_te = (sigmoid(x_te@w)>=0.5).astype(int)
        else:
            raise ValueError('Invalid method specified. Choose from "GD", "SGD", "RidgeRegression", "LogisticRegression", or "RegLogisticRegression".')
 
        
        # Compute metrics 
        accuracy_ = scores(y_pred=predictions_te,y_true=y_te)[0]
        precision_ = scores(y_pred=predictions_te,y_true=y_te)[1]
        f1_score_ = scores(y_pred=predictions_te,y_true=y_te)[3]
        
        accuracy_scores.append(accuracy_)
        precision_scores.append(precision_)
        f1_scores.append(f1_score_)
    
    return accuracy_scores, precision_scores, f1_scores,w

models = ["GD", "SGD", "RidgeRegression","LogisticRegression","RegLogisticRegression"]

accuracy_m = [cross_validate(y_train_resampled,tx_tr,method= model,initial_w=initial_w)[0] for model in models]
precision_m = [cross_validate(y_train_resampled,tx_tr,method= model,initial_w=initial_w)[1] for model in models]
f1_score_m = [cross_validate(y_train_resampled,tx_tr,method= model,initial_w=initial_w)[2] for model in models]
w = [cross_validate(y_train_resampled,tx_tr,method= model,initial_w=initial_w)[3] for model in models]

# Combine metrics into a single structure for plotting
metrics_data = {
    'Accuracy': accuracy_m,
    'Precision': precision_m,
    'F1 Score': f1_score_m
}

fig, axs = plt.subplots(3, 1, figsize=(10, 18)) 


for i, (metric, scores) in enumerate(metrics_data.items()):
    
    axs[i].boxplot(scores, labels=models)

    
    means = [np.mean(score) for score in scores]
    stds = [np.std(score) for score in scores]

    #
    axs[i].errorbar(range(1, len(models) + 1), means, yerr=stds, fmt='o', color='red', label='Mean ± Std')

    
    for j, mean in enumerate(means):
        axs[i].annotate(f'{mean:.2f}', xy=(j + 1, mean), 
                        textcoords='offset points', 
                        xytext=(0, 5), 
                        ha='center', color='black')

    axs[i].set_title(f'Comparison of {metric}')
    axs[i].set_ylabel('Scores')
    axs[i].grid(axis='y')
    axs[i].legend()


plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Predition
x_test_clean,_ = data_cleaning(x_test,columns_to_delete=filter)
x_test_continuous = x_test_clean[:, continuous_columns]
x_test_categorical = x_test_clean[:, categorical_columns]

x_test_cont_filled = fill_missing_value(x_test_continuous)
x_test_cat_filled = fill_missing_value(x_test_categorical,data_type='catagorical')


x_test_standardized = (x_test_cont_filled - mean_x) / std_x
x_test_onehot = one_hot_encode(x_test_cat_filled, unique_categories)

tx_te = generate_tx(np.hstack((x_test_standardized, x_test_onehot)))


# Calculate probability prediction, insert the corresponding w
probabilities_pred = sigmoid(tx_te @ w[models=="LogisticRegression"]) # Sigmoid is logistic regression
    
# Convert probabilities to binary labels
y_pred = np.where(probabilities_pred >= 0.5, 1, -1)   # 0.5 if sigmoid

print(y_pred)
print(y_pred.shape)
print(np.unique(y_pred))
print("Unhealthy :", np.sum(y_pred == 1)/len(y_pred) * 100, "%")

ids = []

for n in range(y_pred.shape[0]):
    ids.append(328135 + n)

create_csv_submission(ids, y_pred, "submission-logistic_regerssion.csv")

