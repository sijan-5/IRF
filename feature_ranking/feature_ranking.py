
# Tree data (without comments for proper parsing)

# Organize collected data
from collections import defaultdict


class LocalGlobalWt:
    # Function to traverse the tree and collect (feature_idx, quality_of_split
    def __init__(self, number_of_features):
        self.feature_quality_list = []
        self.N = 0
        self.number_of_features = number_of_features

    def collect_features(self, tree, feature_quality_list):
        if isinstance(tree, dict):
            if "feature_idx" in tree and "quality_of_split" in tree:
                feature_quality_list.append((tree["feature_idx"], tree["quality_of_split"]))
            # Traverse left and right
            if "left_split" in tree:
                self.collect_features(tree["left_split"], feature_quality_list)
            if "right_split" in tree:
                self.collect_features(tree["right_split"], feature_quality_list)

    def find_local_weight_feature(self, tree):
        feature_to_qualities = defaultdict(list)
        self.collect_features(tree, self.feature_quality_list)
        for feature_idx, quality in self.feature_quality_list:
            feature_to_qualities[feature_idx].append(quality)

        for feature in range(self.number_of_features):
            if feature not in feature_to_qualities:
                feature_to_qualities[feature] = [0]
        feature_to_qualities = dict(sorted(feature_to_qualities.items()))
        # Number of nodes considered (only nodes where splits happen)
        self.N = len(self.feature_quality_list)

        # Calculate sums and local weights
        feature_sums = {feature: sum(qualities) for feature, qualities in feature_to_qualities.items()}
        local_weights = {feature: total / self.N for feature, total in feature_sums.items()}
        return list(local_weights.values())

    def normalized_weight_of_tree(self, oob_list):
        inverse_list = [1 / oob for oob in oob_list]
        highest = max(inverse_list)
        normalized_weight = [oob / highest for oob in inverse_list]
        return normalized_weight

    def global_wt(self, feature_wt_ls, normalized_tree_wt_ls):
        num_features = len(feature_wt_ls[0])
        feature_sums = []
        for i in range(num_features):
            sum_feature = feature_wt_ls[0][i] * normalized_tree_wt_ls[0] + feature_wt_ls[1][i] * normalized_tree_wt_ls[
                1] + feature_wt_ls[2][i] * normalized_tree_wt_ls[2]
            feature_sums.append(sum_feature)
        # Printing nicely
        for idx, val in enumerate(feature_sums):
            print(f"sum_feature{idx} : {val}")
        largest = max(feature_sums)
        return [round(x/largest, 4) for x in feature_sums]

