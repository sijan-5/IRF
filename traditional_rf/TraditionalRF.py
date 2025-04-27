import random
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from r_q_str_corr.find_r_q_s_c import compute_r, compute_q, compute_strength, compute_correlation
from feature_weight_update.feature_weight_update import compute
from treenum.treenum import compute_accuracy, compute_qu_qv, compute_nu, compute_l, compute_deltaB
from feature_ranking.feature_ranking import LocalGlobalWt

# How many samples you want
num_samples = 14 * 3  # Tripling original dataset size
each_tree_ft_wt = []
each_tree_normalized_wt = []

global tree_ls, nav, v, n_estimators

# Options for each feature
outlook_options = ["Sunny", "Overcast", "Rain"]
temperature_options = ["Hot", "Mild", "Cool"]
humidity_options = ["High", "Normal"]
wind_options = ["Weak", "Strong"]
cloudcover_options = ["None", "Partial", "Full"]
pressure_options = ["High", "Medium", "Low"]
visibility_options = ["Clear", "Foggy"]
dewpoint_options = ["Low", "Medium", "High"]
uvindex_options = ["Low", "Moderate", "High"]
rainfall_options = ["No", "Light", "Heavy"]
snowfall_options = ["No", "Light", "Heavy"]
thunder_options = ["No", "Yes"]
storm_options = ["No", "Yes"]
heatwave_options = ["No", "Yes"]
coldsnap_options = ["No", "Yes"]
pollutionlevel_options = ["Low", "Moderate", "High"]
airquality_options = ["Good", "Moderate", "Poor"]
winddirection_options = ["North", "South", "East", "West"]
sunshine_options = ["Low", "Medium", "High"]
humiditylevel_options = ["Low", "Moderate", "High"]
temperaturefeel_options = ["Cold", "Warm", "Hot"]
pressuretrend_options = ["Rising", "Falling", "Steady"]
visibilitytrend_options = ["Improving", "Worsening"]
season_options = ["Summer", "Winter", "Monsoon", "Spring"]
lightning_options = ["None", "Mild", "Severe"]
play_tennis_options = ["Yes", "No"]

# Build the dataset dictionary
dataset = {
    "Outlook": [random.choice(outlook_options) for _ in range(num_samples)],
    "Temperature": [random.choice(temperature_options) for _ in range(num_samples)],
    "Humidity": [random.choice(humidity_options) for _ in range(num_samples)],
    "Wind": [random.choice(wind_options) for _ in range(num_samples)],
    "CloudCover": [random.choice(cloudcover_options) for _ in range(num_samples)],
    "Pressure": [random.choice(pressure_options) for _ in range(num_samples)],
    "Visibility": [random.choice(visibility_options) for _ in range(num_samples)],
    "DewPoint": [random.choice(dewpoint_options) for _ in range(num_samples)],
    "UVIndex": [random.choice(uvindex_options) for _ in range(num_samples)],
    "Rainfall": [random.choice(rainfall_options) for _ in range(num_samples)],
    "Snowfall": [random.choice(snowfall_options) for _ in range(num_samples)],
    "Thunder": [random.choice(thunder_options) for _ in range(num_samples)],
    "Storm": [random.choice(storm_options) for _ in range(num_samples)],
    "Heatwave": [random.choice(heatwave_options) for _ in range(num_samples)],
    "ColdSnap": [random.choice(coldsnap_options) for _ in range(num_samples)],
    "PollutionLevel": [random.choice(pollutionlevel_options) for _ in range(num_samples)],
    "AirQuality": [random.choice(airquality_options) for _ in range(num_samples)],
    "WindDirection": [random.choice(winddirection_options) for _ in range(num_samples)],
    "Sunshine": [random.choice(sunshine_options) for _ in range(num_samples)],
    "HumidityLevel": [random.choice(humiditylevel_options) for _ in range(num_samples)],
    "TemperatureFeel": [random.choice(temperaturefeel_options) for _ in range(num_samples)],
    "PressureTrend": [random.choice(pressuretrend_options) for _ in range(num_samples)],
    "VisibilityTrend": [random.choice(visibilitytrend_options) for _ in range(num_samples)],
    "Season": [random.choice(season_options) for _ in range(num_samples)],
    "Lightning": [random.choice(lightning_options) for _ in range(num_samples)],
    "PlayTennis": [random.choice(play_tennis_options) for _ in range(num_samples)],
}

# Convert to DataFrame
df = pd.DataFrame(dataset)
f = int(math.sqrt(len(df.columns)))
v = len(df.columns)  # Total columns (features + label)
n_estimators = 10
feature_ranking = LocalGlobalWt(len(df.columns) - 1)

feature_names = list(df.columns) # e.g. ['Outlook', 'Temperature', …, 'PlayTennis']
label = feature_names[-1]          # 'PlayTennis'


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
    left_child_entropy = entropy(p_left)
    right_child_entropy = entropy(p_right)
    return math.exp(-(left_child_entropy + right_child_entropy))

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

    best_quality_of_split = -999999
    node = None
    for feature_idx in feature_ls:
        for split_point in X_bootstrap[:, feature_idx]:
            left_child = {'X_bootstrap': [], 'y_bootstrap': []}
            right_child = {'X_bootstrap': [], 'y_bootstrap': []}

            if type(split_point) in [int, float]:
                for i, value in enumerate(X_bootstrap[:, feature_idx]):
                    if value <= split_point:
                        left_child['X_bootstrap'].append(X_bootstrap[i])
                        left_child['y_bootstrap'].append(y_bootstrap[i])
                    else:
                        right_child['X_bootstrap'].append(X_bootstrap[i])
                        right_child['y_bootstrap'].append(y_bootstrap[i])
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
    return node

def terminal_node(node):
    y_bootstrap = node['y_bootstrap']
    pred = max(set(y_bootstrap), key=y_bootstrap.count)
    return pred

def split_node(node, max_features, min_samples_split, max_depth, depth):
    left = node['left_child']
    right = node['right_child']
    del node['left_child'], node['right_child']

    # If one side has no samples, make this a terminal node
    if not left['y_bootstrap'] or not right['y_bootstrap']:
        merged = {'y_bootstrap': left['y_bootstrap'] + right['y_bootstrap']}
        node['left_split']  = terminal_node(merged)
        node['right_split'] = terminal_node(merged)
        return

    # Only enforce depth limit if max_depth is given
    if max_depth is not None and depth >= max_depth:
        node['left_split']  = terminal_node(left)
        node['right_split'] = terminal_node(right)
        return

    # Left child: either terminal (too few samples) or split further
    if len(left['X_bootstrap']) <= min_samples_split:
        node['left_split'] = terminal_node(left)
    else:
        node['left_split'] = find_split_point(
            left['X_bootstrap'], left['y_bootstrap'], max_features
        )
        split_node(node['left_split'],
                   max_features,
                   min_samples_split,
                   max_depth,
                   depth + 1)

    # Right child: same logic
    if len(right['X_bootstrap']) <= min_samples_split:
        node['right_split'] = terminal_node(right)
    else:
        node['right_split'] = find_split_point(
            right['X_bootstrap'], right['y_bootstrap'], max_features
        )
        split_node(node['right_split'],
                   max_features,
                   min_samples_split,
                   max_depth,
                   depth + 1)


def build_tree(X_bootstrap, y_bootstrap, max_features, max_depth, min_samples_split):
    root = find_split_point(X_bootstrap, y_bootstrap, max_features)
    split_node(root, max_features, min_samples_split, max_depth, 1)
    return root

def random_forest(x_train, y_train, n_estimators, max_features, max_depth, min_samples_split):
    global tree_ls
    tree_ls = []
    global nav
    oob_ls = []
    for _ in range(n_estimators):
        x_bootstrap, y_bootstrap, x_oob, y_oob = draw_bootstrap(x_train, y_train)
        tree = build_tree(x_bootstrap, y_bootstrap, max_features, max_depth, min_samples_split)
        tree_ls.append(tree)
        oob_error = oob_score(tree, x_oob, y_oob)
        each_tree_ft_wt.append(feature_ranking.find_local_weight_feature(tree))
        oob_ls.append(oob_error)

    nav = int(feature_ranking.counter / len(tree_ls))
    each_tree_normalized_wt.extend(feature_ranking.normalized_weight_of_tree(oob_ls))
    return tree_ls

def predict_tree(tree, x_test):
    feature_idx = tree['feature_idx']
    split_point = tree['split_point']
    feature_value = x_test[feature_idx]

    if isinstance(split_point, (int, float)):
        if feature_value <= split_point:
            branch = tree['left_split']
        else:
            branch = tree['right_split']
    else:
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

while v >= f:
    print()
    print("-------------///////////////////////////////////////////iteration start////////////////////////////////////------------------------------------")

    X = df.drop(columns=[label])
    y = df[label]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    max_features = f
    max_depth = None
    min_samples_split = 2

    model = random_forest(X_train, y_train, n_estimators, max_features, max_depth, min_samples_split)
    global_wt = feature_ranking.global_wt(each_tree_ft_wt, each_tree_normalized_wt)
    updated_features, u, v_new, del_u, del_v = compute(global_wt)
    v=v_new
    r = compute_r(u, v, f)
    q = compute_q(u, v, f)
    strength = compute_strength(q, nav, n_estimators)
    correlation, rho = compute_correlation(u, v, f, nav, n_estimators)

    accuracy = compute_accuracy(strength, correlation)
    change_in_u, change_in_v = compute_qu_qv(u, v, f)
    nu = compute_nu(q, rho, nav, n_estimators)
    l = compute_l(q, nav, n_estimators)
    delta_b = compute_deltaB(change_in_u, change_in_v, del_u, del_v, l, nu)

    preds = predict_rf(model, X_test)
    acc = sum(preds == y_test.values) / len(y_test)

    # 1) Build the filtered dict by index
    all_cols = list(dataset.keys())
    last_idx = len(all_cols) - 1        # index of PlayTennis
    keep_idxs = set(updated_features.keys())

    new_dataset = {
        col: vals
        for idx, (col, vals) in enumerate(dataset.items())
        if idx in keep_idxs or idx == last_idx
    }
     
    dataset=new_dataset
    
    # 2) And as a DataFrame:
    df = pd.DataFrame(dataset)
    
    print()
    print("Testing accuracy: {}".format(np.round(acc, 3)))
    n_estimators += delta_b
    print("f", f)
    print("v", v)
    print("del_b", delta_b)
    print("n_estimators---------", n_estimators)
    print("-------------///////////////////////////////////////////iteration end////////////////////////////////////------------------------------------")
    print()