# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def compute_loss(y, tx, w):
    """Computes the Loss function."""

    e = y - tx @ w
    loss = (1 / (2*len(y))) * np.sum(e**2)

    return loss


def scores_f(y_true, y_pred):

    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))

    accuracy = (TP + TN) / (TP + FP + FN + TN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy,precision,recall,f1_score
    
def compute_gradient(y, tx, w):
    """Computes the Gradient at w."""
    N, D = tx.shape
    error = y - tx.dot(w)

    gradient = -1 / N * tx.T.dot(error)

    return gradient


def sigmoid(t):
    """Computes the Sigmoid."""
    return 1 / (1 + np.exp(-t))


def calculate_sigmoid_loss(y, tx, w):
    """Computes the Sigmoid loss."""

    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    N, D = tx.shape
    pred_probs = sigmoid(tx @ w)
    loss = (
        -1
        / N
        * np.sum(
            y.T @ np.log(pred_probs) + (np.ones((1, N)) - y.T) @ np.log(1 - pred_probs)
        )
    )

    return loss


def calculate_sigmoid_gradient(y, tx, w):
    """Computes the Sigmoid gradient."""
    N, D = tx.shape
    pred_probs = sigmoid(tx @ w)
    grad = (1 / N) * tx.T @ (pred_probs - y)

    return grad


def least_squares(y, tx):
    """calculate the Least Squares."""
    N, D = tx.shape

    w = np.linalg.solve(tx.T @ tx, tx.T @ y)

    loss = compute_loss(y, tx, w)

    return w, loss


def ridge_regression(y, tx, lambda_):
    """implement Ridge Regression."""

    N, D = tx.shape
    A = tx.T @ tx + 2 * N * lambda_ * np.identity(D)
    b = tx.T @ y
    w = np.linalg.solve(A, b)
    loss = compute_loss(y, tx, w)

    return w, loss


# buggy buggy...
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """Generate a minibatch iterator for a dataset."""

    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm."""

    w = initial_w
    loss = compute_loss(y, tx, w)

    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)

        w = w - gamma * gradient
        loss = compute_loss(y, tx, w)
        # store w and loss

    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent (SGD) algorithm."""

    w = initial_w
    loss = compute_loss(y, tx, w)

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, shuffle=True):
            stoch_grad = compute_gradient(y_batch, tx_batch, w)
            w = w - gamma * stoch_grad
        loss = compute_loss(y, tx, w)

    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """The Logistic Regression algorithm."""
    w = initial_w
    loss = calculate_sigmoid_loss(y, tx, w)
    losses = []
    threshold = 1e-8

    for iter in range(max_iters):
        grad = calculate_sigmoid_gradient(y, tx, w)
        w = w - gamma * grad
        loss = calculate_sigmoid_loss(y, tx, w)

        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w.reshape(-1), loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """The Regularized Logistic Regression algorithm."""
    w = initial_w
    loss = calculate_sigmoid_loss(y, tx, w)
    losses = []
    threshold = 1e-8

    for iter in range(max_iters):
        # sigmoid function
        grad = calculate_sigmoid_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad
        loss = calculate_sigmoid_loss(y, tx, w)
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w.reshape(-1), loss

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x


def one_hot_encode(data, unique_categories):
    """Given the catgories of the training set, one-hot encode the testing set."""
    encoded_columns = []
    for col in range(data.shape[1]):
        # Get the unique categories and map for this column based on training data
        categories = unique_categories[col]
        category_map = {category: i for i, category in enumerate(categories)}
        
        # Create one-hot encoding for the column
        col_encoded = np.zeros((data.shape[0], len(categories)))
        for i, value in enumerate(data[:, col]):
            if value in category_map:
                col_encoded[i, category_map[value]] = 1
        encoded_columns.append(col_encoded)
    
    return np.hstack(encoded_columns)

def visualize_nan_hist(data,title = 'NaN Count per Column'):
    # Count NaNs per column
    nan_counts = np.isnan(data).sum(axis=0)

    # Plot histogram of NaN per column
    plt.figure(figsize=(12, 8))
    plt.bar(range(data.shape[1]), nan_counts, color='blue', edgecolor='black')
    plt.xlabel('Column Index')
    plt.ylabel('Number of NaN Values')
    plt.title(title)
    plt.show()

def visualize_nan_proportion(data,title = 'NaN Proportion per Column'):
    # Count NaNs per column
    nan_frac = np.isnan(data).sum(axis=0)/data.shape[0]

    # Plot histogram of NaN per column
    plt.figure(figsize=(12, 8))
    plt.bar(range(data.shape[1]), nan_frac, color='blue', edgecolor='black', )
    plt.xlabel('Column Index')
    plt.ylabel('Percentage of NaN Values')
    plt.ylim(0, 1)
    plt.title(title)
    plt.show()

def visualize_nan_heatmap_proportion(data_cont,data_cat,title = "Before replacing (NaN proportion)"):
    # Continuous values count NaNs per column
    nan_cont = np.isnan(data_cont).sum(axis=0)/data_cont.shape[0]

    # Categorical values count NaNs per column
    nan_cat = np.isnan(data_cat).sum(axis=0)/data_cat.shape[0]

    # Plotting both heatmaps side by side
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # First heatmap for NaN counts of the continuous array
    cax1 = ax[0].imshow(nan_cont[np.newaxis, :], cmap="jet", aspect="auto", vmin=0)
    fig.colorbar(cax1, ax=ax[0], label="Fraction of NaNs for Continuous Columns")
    ax[0].set_title("Continuous features")
    ax[0].set_xlabel("Columns")
    ax[0].set_xticks(np.arange(data_cont.shape[1]))
    ax[0].set_xticklabels(np.arange(data_cont.shape[1]), rotation=90)
    ax[0].set_yticks([])

    # Second heatmap for NaN counts of the catagorical array
    cax2 = ax[1].imshow(nan_cat[np.newaxis, :], cmap="viridis", aspect="auto", vmin=0)
    fig.colorbar(cax2, ax=ax[1], label="Fraction of NaNs for Categorical Columns")
    ax[1].set_title("Categorical features")
    ax[1].set_xticks(np.arange(data_cat.shape[1]))
    ax[1].set_xticklabels(np.arange(data_cat.shape[1]), rotation=90)
    ax[1].set_yticks([])

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def downsampling(X,y):
    # Count the samples in each class
    num_unhealthy = np.sum(y == 1)
    num_healthy = np.sum(y == -1)

    # Get indices for each class
    unhealthy = np.where(y == 1)[0]
    healthy = np.where(y == -1)[0]

    healthy_ratio = int(num_unhealthy * 4)  # Set the ratio to 4 to 1

    # Randomly downsample the majority class to match the minority count
    downsampled_healthy = np.random.choice(healthy, healthy_ratio, replace=False)

    # Combine indices of balanced classes
    balanced_id = np.concatenate([unhealthy, downsampled_healthy])

    # Resample X and y
    x_resampled = X[balanced_id]
    y_resampled = y[balanced_id]

    print("New proportion of unhealthy :", np.sum(y_resampled == 1)/len(y_resampled) * 100, "%")

    return x_resampled,y_resampled

def data_cleaning(x,columns_to_delete=None,nan_threshold = 0.15):
    # Step 1
    x = np.delete(x, np.s_[:26], axis=1)

    # Step 2
    # Calculate Q1 (25th percentile) and Q3 (75th percentile) for each feature
    Q1 = np.percentile(x, 25, axis=0)
    Q3 = np.percentile(x, 75, axis=0)
    IQR = Q3 - Q1
    # Find the indices of extreme outliers
    refused = (x > (Q3 + 3 * IQR))
    # Replace extreme outliers with NaN
    x[refused] = np.nan 

    # Step 3
    #visualize_nan_hist(x,title = 'NaN Count per Column before cleaning')
    visualize_nan_proportion(x,title = 'NaN percentage per Column before cleaning')


    if columns_to_delete is None:
        cleaning_threshold = x.shape[0] * nan_threshold # Columns with more than 15% NaNs will be deleted
        nb_nan = np.isnan(x).sum(axis=0)
        columns_to_delete = np.where(nb_nan > cleaning_threshold)[0] 
    
    x_clean = np.delete(x, columns_to_delete, axis=1)

    visualize_nan_proportion(x_clean,title = 'NaN Count per Column after cleaning')

    return x_clean, columns_to_delete

def data_spliting(x,continuous_threshold = 25):
    continuous_columns = []
    categorical_columns = []
    continuous_threshold = 25

    # Determine all the possible values in each column
    for n in range(x.shape[1]):
        if np.unique(x[:,n]).size < continuous_threshold :
            categorical_columns.append(n)
        else : 
            continuous_columns.append(n)

    # N° of columns and the quantity for each type
    print(f"Continuous columns :{continuous_columns},{len(continuous_columns)} features; Categorical columns :{categorical_columns},{len(categorical_columns)} features")
    
    x_continuous = x[:, continuous_columns]
    x_categorical = x[:, categorical_columns]

    return x_continuous,x_categorical,continuous_columns,categorical_columns

def fill_missing_value(data,data_type='continuous'):
    """Fill missing value in columns"""

    for col_idx in range(data.shape[1]):
        if data_type =='continuous':
            col_mean = np.nanmean(data[:, col_idx])
            data[np.isnan(data[:, col_idx]), col_idx] = col_mean 

        else:
            values, counts = np.unique(data[~np.isnan(data[:, col_idx]), col_idx], return_counts=True)
            # Calculate the mode of the column, ignoring NaNs
            mode = values[np.argmax(counts)]
            data[np.isnan(data[:, col_idx]), col_idx] = mode

    return data

# def data_normalize(x,split_type = True, continuous_columns = None, categorical_columns = None):
#     #If split_type is true, we devide the columns using the number of categories. 
#     #If split_type is false, we select the columns using a "manual" filter
#     if split_type:
#         x_continuous,x_categorical,continuous_columns,categorical_columns = data_spliting(x)
#     else: 
#         x_continuous = x[:,continuous_columns]
#         x_categorical = x [:,categorical_columns]

#     visualize_nan_heatmap(x_continuous,x_categorical,title="Before replacing")

#     # Copy to preserve the initial arrays for later use
#     x_continuous_filled = fill_missing_value(x_continuous)
#     x_categorical_filled = fill_missing_value(x_categorical,data_type='catagorical')

#     visualize_nan_heatmap(x_continuous,x_categorical,title="After replacing")

#     x_cont_standardized,mean_x,std_x = standardize(x_continuous_filled)

#     unique_categories = [np.unique(x_categorical_filled[:, col]) for col in range(x_categorical_filled.shape[1])]
#     x_categorical_encoded = one_hot_encode(x_categorical_filled,unique_categories=unique_categories)

#     return np.hstack((x_cont_standardized, x_categorical_encoded)),continuous_columns,categorical_columns,mean_x,std_x,unique_categories

def data_normalize(x,split_type = True, continuous_columns = None, categorical_columns = None):
    #If split_type is true, we devide the columns using the number of categories. 
    #If split_type is false, we select the columns using a "manual" filter
    if split_type:
        x_continuous,x_categorical,continuous_columns,categorical_columns = data_spliting(x)
    else: 
        x_continuous = x[:,continuous_columns]
        x_categorical = x [:,categorical_columns]

    visualize_nan_heatmap_proportion(x_continuous,x_categorical,title="Before replacing")

    # Copy to preserve the initial arrays for later use
    x_continuous_filled = fill_missing_value(x_continuous)
    x_categorical_filled = fill_missing_value(x_categorical,data_type='catagorical')

    visualize_nan_heatmap_proportion(x_continuous,x_categorical,title="After replacing")

    x_cont_standardized,mean_x,std_x = standardize(x_continuous_filled)

    unique_categories = [np.unique(x_categorical_filled[:, col]) for col in range(x_categorical_filled.shape[1])]
    x_categorical_encoded = one_hot_encode(x_categorical_filled,unique_categories=unique_categories)

    return np.hstack((x_cont_standardized, x_categorical_encoded)),continuous_columns,categorical_columns,mean_x,std_x,unique_categories


def train_test_split(X,y,test_size = 0.2,seed = 15):
    test_size = 0.20    # 80% training, 20% testing
    num_samples = len(y)
    num_test_samples = int(test_size * num_samples)

    # Separate indices for each class
    healthy_indices = np.where(y == -1)[0]
    unhealthy_indices = np.where(y == 1)[0]

    # Determine number of test samples per class based on original distribution
    num_test_samples_healthy = int(test_size * len(healthy_indices))
    num_test_samples_unhealthy = num_test_samples - num_test_samples_healthy

    print("Healthy :", num_test_samples_healthy, "; Unhealthy :", num_test_samples_unhealthy)

    # Randomly shuffle and split indices for each class
    np.random.seed(seed)
    np.random.shuffle(healthy_indices)
    np.random.shuffle(unhealthy_indices)

    # Get test indices while maintaining class proportions
    test_indices = np.concatenate([healthy_indices[:num_test_samples_healthy],
                                unhealthy_indices[:num_test_samples_unhealthy]])
    train_indices = np.concatenate([healthy_indices[num_test_samples_healthy:], 
                                unhealthy_indices[num_test_samples_unhealthy:]])
    
    # For further convenience, map {-1,1} to {0,1}
    y = (np.ones(y.shape)+y)/2

    # Create train and test sets
    X_tr,X_te = X[train_indices],X[test_indices]
    y_tr,y_te = y[train_indices],y[test_indices]

    # Visualize the repartition of the data in both set
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))  # Adjust the figure size as needed

    axs[0].hist(y_tr, bins=10, color='blue', alpha=0.7)
    axs[0].set_title('Training Labels Histogram')  
    axs[0].set_ylabel('Output Labels') 
    axs[0].set_xlabel('Values')  

    axs[1].hist(y_te, bins=10, color='orange', alpha=0.7)
    axs[1].set_title('Testing Labels Histogram')  
    axs[1].set_ylabel('Output Labels')  
    axs[1].set_xlabel('Values')  
    
    plt.tight_layout()
    plt.show()

    return X_tr,y_tr,X_te,y_te


def initialize_weight(x,seed = None):
    N,D = x.shape

    if seed is not None:
        np.random.seed(seed)
    
    w = np.random.randn(D)
    
    return w

def generate_tx(x):
    return np.c_[np.ones((x.shape[0], 1)), x] 

def metrics(y_true,predictions):
    model_metrics = {}
    for model, preds in predictions.items():
        model_metrics[model] = {
            'Accuracy': scores_f(y_true, preds)[0],
            'Precision': scores_f(y_true, preds)[1],
            'F1 Score': scores_f(y_true, preds)[3]
        }

    return model_metrics

def visualize_metrics(model_metrics,title = "Model Metrics for Training model"):
    model_names = list(model_metrics.keys())
    
    acc_scores = np.array([model_metrics[model]['Accuracy'] for model in model_names])
    prec_scores = np.array([model_metrics[model]['Precision'] for model in model_names])
    f1_scores = np.array([model_metrics[model]['F1 Score'] for model in model_names])

    fig, axs = plt.subplots(3, 1, figsize=(10, 12)) 

    axs[0].bar(model_names, acc_scores, color='blue', alpha=0.7)
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy Score')
    axs[0].set_ylim(0, None)  
    axs[0].grid(axis='y')
    for i, v in enumerate(acc_scores):
        axs[0].text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom')

    axs[1].bar(model_names, prec_scores, color='orange', alpha=0.7)
    axs[1].set_title('Model Precision')
    axs[1].set_ylabel('Precision Score')
    axs[1].set_ylim(0, None)  
    axs[1].grid(axis='y')
    for i, v in enumerate(prec_scores):
        axs[1].text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom')

    axs[2].bar(model_names, f1_scores, color='green', alpha=0.7)
    axs[2].set_title('Model F1 Score')
    axs[2].set_ylabel('F1 Score')
    axs[2].set_ylim(0, None)  
    axs[2].grid(axis='y')
    for i, v in enumerate(f1_scores):
        axs[2].text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row) 
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)] 
    return np.array(k_indices) 

def cross_validation(y,x,k_fold, lambdas,method = 'ridge_regression',initial_w = None,):
    """cross validation over regularisation parameter lambda."""

    seed = 12
    k_fold = k_fold
    lambdas = lambdas
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []


    for lambda_ in lambdas:
        for k in np.arange(k_fold):
            N = x.shape[0]
    
            train_indices = np.ones(N, dtype=bool)
            train_indices[k_indices[k]] = False
    
            x_te = x[k_indices[k]]
            y_te = y[k_indices[k]]

            x_tr = x[train_indices]
            y_tr = y[train_indices]

            if method == 'ridge_regression':
                w,mse = ridge_regression(y_tr, x_tr,lambda_=lambda_)
                
            elif method == "reg_logistic_regression":
                w,mse = reg_logistic_regression(y_tr, x_tr,lambda_=lambda_,initial_w=initial_w,max_iters=500,gamma=0.01)
        
        rmse_tr_k = np.sqrt(2*mse)
        rmse_te_k = compute_loss(y_te,x_te,w)
        rmse_tr.append(np.mean(rmse_tr_k))
        rmse_te.append(np.mean(rmse_te_k))

    best_rmse = np.min(rmse_te)
    best_lambda = lambdas[np.argmin(rmse_te)]

    print(
        "The choice of lambda which leads to the best test rmse is %.5f with a test rmse of %.3f"
        % (best_lambda, best_rmse)
    )
    return best_lambda, best_rmse

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

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row) 
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)] 
    return np.array(k_indices) 

def cross_validate(y, tx, k_fold=4,method='GD',initial_w =None, gammas=None, lambdas=None):
    """ Perform k-fold cross-validation """
    seed = 12
 
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
            w, loss_tr = ridge_regression(y_tr, x_tr, lambda_=lambdas[0])
            predictions_te = ((x_te@w)>=0).astype(int)
        elif method == 'RegLogisticRegression':
            w, loss_tr = reg_logistic_regression(y=y_tr, tx=x_tr, initial_w=w, max_iters=10000, gamma= gammas[3],lambda_=lambdas[1])
            predictions_te = (sigmoid(x_te@w)>=0.5).astype(int)
        elif method == 'GD':
            w, loss_tr = mean_squared_error_gd(y=y_tr, tx=x_tr, initial_w=w, max_iters=1000, gamma=gammas[0])
            predictions_te = ((x_te@w)>=0).astype(int)
        elif method == 'SGD':
            w, loss_tr = mean_squared_error_sgd(y=y_tr, tx=x_tr, initial_w=w, max_iters=10000, gamma=gammas[1])
            predictions_te = ((x_te@w)>=0).astype(int)
        elif method == 'LogisticRegression':
            w, loss_tr = logistic_regression(y=y_tr, tx=x_tr, initial_w=w, max_iters=1000, gamma=gammas[2])
            predictions_te = (sigmoid(x_te@w)>=0.5).astype(int)
        else:
            raise ValueError('Invalid method specified. Choose from "GD", "SGD", "RidgeRegression", "LogisticRegression", or "RegLogisticRegression".')
 
        
        # Compute metrics 
        accuracy_ = scores_f(y_pred=predictions_te,y_true=y_te)[0]
        precision_ = scores_f(y_pred=predictions_te,y_true=y_te)[1]
        f1_score_ = scores_f(y_pred=predictions_te,y_true=y_te)[3]
        
        accuracy_scores.append(accuracy_)
        precision_scores.append(precision_)
        f1_scores.append(f1_score_)
    
    return accuracy_scores, precision_scores, f1_scores,w

