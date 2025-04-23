import math
import numpy as np

# Generate a longer random feature list
global_feature_list = {
    'feature0': 0.3842, 'feature1': 0.2457, 'feature2': 0.8247, 'feature3': 0.6475,
    'feature4': 0.8985, 'feature5': 0.5111}

# Generate updated feature values
updated_global_feature_list = {'feature0': 0.7092, 'feature1': 0.8273, 'feature2': 0.346, 'feature3': 0.2032,
                               'feature4': 0.6878, 'feature5': 0.1739, 'feature6': 0.8437}
g_list = [global_feature_list, updated_global_feature_list]

# For tracking previous lengths
del_u = 0
del_v = 0
# Initialize important and unimportant feature lists
important_features = {}
unimportant_features = {}
common_imp_unimp = {}
updated_global_list = {}
first_time = True


def compute(global_list):
    global important_features, unimportant_features, del_u, del_v, common_imp_unimp, updated_global_list,first_time
    if first_time:
        sorted_feature_list = dict(sorted(global_list.items(), key=lambda item: item[1], reverse=True))
        n_important = math.isqrt(len(sorted_feature_list))
        items = list(sorted_feature_list.items())
        important_features = dict(items[:n_important])
        unimportant_features = dict(items[n_important:])
        first_time = False
    else:
        for feat in global_list.keys():
            if feat in important_features:
                important_features[feat] = global_list[feat]
            else:
                unimportant_features[feat] = global_list[feat]

    unimp_feature_size_before = len(unimportant_features)
    imp_feature_size_before = len(important_features)
    print("Imp feature", important_features)
    print("Unimp feature", unimportant_features)
    remove_low_unimportant()
    promote_features()
    del_u = len(important_features) - imp_feature_size_before
    del_v = len(unimportant_features) - unimp_feature_size_before
    print("Imp feature", important_features)
    print("Unimp feature", unimportant_features)
    print("del-u", del_u)
    print("del-v", del_v)

def remove_low_unimportant():
    global unimportant_features
    if not unimportant_features:
        return

    values = np.array(list(unimportant_features.values()))
    mean = np.mean(values)
    std = np.std(values)
    threshold = mean - (2 * std)
    # Filter out features below threshold
    unimportant_features = {k: v for k, v in unimportant_features.items() if v >= threshold}

def promote_features():
    global important_features, unimportant_features
    if not unimportant_features:
        return

    min_important_value = min(important_features.values()) if important_features else -float('inf')
    promoted = {k: v for k, v in unimportant_features.items() if v >= min_important_value}
    for k in promoted:
        important_features[k] = unimportant_features.pop(k)