import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
import math

# Possible values from the original dataset
outlook_options = ["Sunny", "Overcast", "Rain"]
temperature_options = ["Hot", "Mild", "Cool"]
humidity_options = ["High", "Normal"]
wind_options = ["Weak", "Strong"]
play_tennis_options = ["Yes", "No"]

# How many samples you want
num_samples = 14 * 3  # Tripling original dataset size

# Generate the dataset
dataset = {
    "Outlook": [random.choice(outlook_options) for _ in range(num_samples)],
    "Temperature": [random.choice(temperature_options) for _ in range(num_samples)],
    "Humidity": [random.choice(humidity_options) for _ in range(num_samples)],
    "Wind": [random.choice(wind_options) for _ in range(num_samples)],
    "PlayTennis": [random.choice(play_tennis_options) for _ in range(num_samples)]
}

# Convert to DataFrame
df = pd.DataFrame(dataset)


print(df['Outlook'])

# Split into features and target

X = df.drop(columns=["PlayTennis"])
y = df["PlayTennis"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Reset index to avoid KeyError during bootstrap sampling
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


# Utility functions
def entropy(p):
    if p == 0 or p == 1:
        return 0
    else:
        return - (p * np.log2(p) + (1 - p) * np.log2(1 - p))


def information_gain(left_child, right_child):
    parent = left_child + right_child
    p_parent = parent.count(1) / len(parent) if len(parent) > 0 else 0
    p_left = left_child.count(1) / len(left_child) if len(left_child) > 0 else 0
    p_right = right_child.count(1) / len(right_child) if len(right_child) > 0 else 0
    IG_p = entropy(p_parent)
    IG_l = entropy(p_left)
    IG_r = entropy(p_right)
    return IG_p - len(left_child) / len(parent) * IG_l - len(right_child) / len(parent) * IG_r


def get_quality_of_split(left_child, right_child):
    p_left = left_child.count('Yes') / len(left_child) if len(left_child) > 0 else 0
    p_right = right_child.count('Yes') / len(right_child) if len(right_child) > 0 else 0
    return math.exp(-(p_left + p_right))


def draw_bootstrap(X_train, y_train):
    bootstrap_indices = list(np.random.choice(range(len(X_train)), len(X_train), replace=True))
    oob_indices = [i for i in range(len(X_train)) if i not in bootstrap_indices]
    X_bootstrap = X_train.iloc[bootstrap_indices].values
    y_bootstrap = y_train.iloc[bootstrap_indices].values
    X_oob = X_train.iloc[oob_indices].values
    y_oob = y_train.iloc[oob_indices].values
    return X_bootstrap, y_bootstrap, X_oob, y_oob


def oob_score(tree, X_test, y_test):
    mis_label = 0
    for i in range(len(X_test)):
        pred = predict_tree(tree, X_test[i])
        if pred != y_test[i]:
            mis_label += 1
    return mis_label / len(X_test)


def find_split_point(X_bootstrap, y_bootstrap, max_features):
    global quality_of_split
    feature_ls = []
    num_features = len(X_bootstrap[0])

    while len(feature_ls) <= max_features:
        feature_idx = random.sample(range(num_features), 1)
        if feature_idx not in feature_ls:
            feature_ls.extend(feature_idx)

    # best_info_gain = -999
    best_quality_of_split = -999
    node = None
    for feature_idx in feature_ls:
        for split_point in X_bootstrap[:, feature_idx]:
            left_child = {'X_bootstrap': [], 'y_bootstrap': []}
            right_child = {'X_bootstrap': [], 'y_bootstrap': []}

            # split children for continuous variables
            if type(split_point) in [int, float]:
                for i, value in enumerate(X_bootstrap[:, feature_idx]):
                    if value <= split_point:
                        left_child['X_bootstrap'].append(X_bootstrap[i])
                        left_child['y_bootstrap'].append(y_bootstrap[i])
                    else:
                        right_child['X_bootstrap'].append(X_bootstrap[i])
                        right_child['y_bootstrap'].append(y_bootstrap[i])
            # split children for categoric variables
            else:
                for i, value in enumerate(X_bootstrap[:, feature_idx]):
                    if value == split_point:
                        left_child['X_bootstrap'].append(X_bootstrap[i])
                        left_child['y_bootstrap'].append(y_bootstrap[i])
                    else:
                        right_child['X_bootstrap'].append(X_bootstrap[i])
                        right_child['y_bootstrap'].append(y_bootstrap[i])

            quality_of_split = get_quality_of_split(left_child['y_bootstrap'], right_child['y_bootstrap'])
            if quality_of_split > best_quality_of_split:
                best_quality_of_split = quality_of_split
                left_child['X_bootstrap'] = np.array(left_child['X_bootstrap'])
                right_child['X_bootstrap'] = np.array(right_child['X_bootstrap'])
                node = {'quality_of_split': best_quality_of_split,
                        'left_child': left_child,
                        'right_child': right_child,
                        'split_point': split_point,
                        'feature_idx': feature_idx}

            # split_info_gain = information_gain(left_child['y_bootstrap'], right_child['y_bootstrap'])
            # if split_info_gain > best_info_gain:
            #     best_info_gain = split_info_gain
            #     left_child['X_bootstrap'] = np.array(left_child['X_bootstrap'])
            #     right_child['X_bootstrap'] = np.array(right_child['X_bootstrap'])
            #     node = {'information_gain': split_info_gain,
            #             'left_child': left_child,
            #             'right_child': right_child,
            #             'split_point': split_point,
            #             'feature_idx': feature_idx}
    return node


def terminal_node(node):
    y_bootstrap = node['y_bootstrap']
    pred = max(set(y_bootstrap), key=y_bootstrap.count)
    return pred


def split_node(node, max_features, min_samples_split, max_depth, depth):
    left_child = node['left_child']
    right_child = node['right_child']

    del node['left_child']
    del node['right_child']

    if len(left_child['y_bootstrap']) == 0 or len(right_child['y_bootstrap']) == 0:
        empty_child = {'y_bootstrap': left_child['y_bootstrap'] + right_child['y_bootstrap']}
        node['left_split'] = terminal_node(empty_child)
        node['right_split'] = terminal_node(empty_child)
        return

    if depth >= max_depth:
        node['left_split'] = terminal_node(left_child)
        node['right_split'] = terminal_node(right_child)
        return

    if len(left_child['X_bootstrap']) <= min_samples_split:
        node['left_split'] = terminal_node(left_child)
    else:
        node['left_split'] = find_split_point(left_child['X_bootstrap'], left_child['y_bootstrap'], max_features)
        split_node(node['left_split'], max_features, min_samples_split, max_depth, depth + 1)

    if len(right_child['X_bootstrap']) <= min_samples_split:
        node['right_split'] = terminal_node(right_child)
    else:
        node['right_split'] = find_split_point(right_child['X_bootstrap'], right_child['y_bootstrap'], max_features)
        split_node(node['right_split'], max_features, min_samples_split, max_depth, depth + 1)


def build_tree(X_bootstrap, y_bootstrap, max_features, max_depth, min_samples_split):
    root = find_split_point(X_bootstrap, y_bootstrap, max_features)
    split_node(root, max_features, min_samples_split, max_depth, 1)
    return root


def random_forest(x_train, y_train, n_estimators, max_features, max_depth, min_samples_split):
    tree_ls = []
    oob_ls = []
    for _ in range(n_estimators):
        x_bootstrap, y_bootstrap, x_oob, y_oob = draw_bootstrap(x_train, y_train)
        tree = build_tree(x_bootstrap, y_bootstrap, max_features, max_depth, min_samples_split)
        tree_ls.append(tree)
        oob_error = oob_score(tree, x_oob, y_oob)
        oob_ls.append(oob_error)
    print("OOB estimate: {:.2f}".format(np.mean(oob_ls)))
    return tree_ls


def predict_tree(tree, x_test):
    feature_idx = tree['feature_idx']
    split_point = tree['split_point']
    feature_value = x_test[feature_idx]

    if isinstance(split_point, (int, float)):  # Numerical feature
        if feature_value <= split_point:
            branch = tree['left_split']
        else:
            branch = tree['right_split']
    else:  # Categorical feature
        if feature_value == split_point:
            branch = tree['left_split']
        else:
            branch = tree['right_split']

    if isinstance(branch, dict):
        return predict_tree(branch, x_test)
    else:
        return branch


def predict_rf(tree_ls, x_test):
    pred_ls = list()
    for i in range(len(x_test)):
        ensemble_preds = [predict_tree(tree, x_test.values[i]) for tree in tree_ls]
        final_pred = max(ensemble_preds, key=ensemble_preds.count)
        pred_ls.append(final_pred)
    return np.array(pred_ls)


# Train Random Forest
n_estimators = 3
max_features = 3
max_depth = 3
min_samples_split = 2

model = random_forest(X_train, y_train, n_estimators, max_features, max_depth, min_samples_split)

# Make predictions
preds = predict_rf(model, X_test)

# Evaluate
acc = sum(preds == y_test.values) / len(y_test)
print("abc")
print("Testing accuracy: {}".format(np.round(acc, 3)))
