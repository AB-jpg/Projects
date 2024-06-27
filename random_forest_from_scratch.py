# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.
# python pacman.py --pacman ClassifierAgent
# python pacman.py --p TraceAgent



import numpy as np
import random
# The chosen classifier is a random forest. 
# This classifier is written step by step (i.e. doesn’t just use a library)
# I chose information gain for best split over Gini_score and have used the formulas for entropy and information gain from the lectures
# The design and explanation of this code follow a top-down approach.
# filled in skeleton code with predict from most common result in a fitted forest, by looping and checking individual result of every decision tree
# to get a individual result if every tree, we recursively traverse each tree in the function single_decision_tree_result for an instance
# to have fitted trees to traverse in a random forest, we fit the forest in the function random_forest_fitting
# perform bagging by creating different subsets of the original dataset by sampling with replacement in random_forest_fitting
# train each decision tree on this bagged data by recursively splitting the data based on features selected through random subspace method 
# find the best split (max info_gain) by calculating information gain of every combination of features selected through random subspace method and selecting the best one
# pick the best feature to split on with the max information gain = entropy(parent) – [average entropy(children)] 
# uses plurality classification when reaches termination point due to instances or features running out




class Classifier:
  
    def __init__(self):
        pass

    def reset(self):
        # reset all the features and variables of the Classifier class to their initial states. 
        # This includes deleting all the learned data from the `fit` method. 
        self.target_column = None
        self.data_array = None
        self.random_forest = None
    
    def fit(self, data, target):
        # want to make the dataset one again so we can easily bootstrap randomly
        self.target_column = np.reshape(target, (-1,1))
        self.data_array = np.concatenate((np.array(data), self.target_column), axis=1)
        # Calling the `random_forest` function with the following parameters 
        # data: self.data_array  
        # num_trees: 6,  
        # ratio_bagged: 0.2, 
        # smallest_leaf_size: 10, 
        # num_features: 6, 
        # max_depth: 5
        self.random_forest = self.random_forest_fitting(self.data_array, 6, 0.2, 10, 6, 5)

    # my code starts here
    def predict(self, features, legal=None):

        # Create an empty list to store predictions from each decision tree
        predictions = []

        # Loop through each decision tree in the forest
        for tree in self.random_forest:
            # Use the tree to classify the features and store the result
            predictions.append(self.single_decision_tree_result(features, decision_tree=tree))

        # Find the most common prediction among all trees
        most_common_prediction = max(predictions, key=predictions.count)
        # debugging
        print('most_common_prediction = ', most_common_prediction)
        # Return the most common prediction as the final result
        return most_common_prediction
                
    def single_decision_tree_result(self, instance, decision_tree):

        # The decision_tree is class target if we reached a leaf
        if isinstance(decision_tree, str):
            return decision_tree

        # If it's not, it's a dictionary with a split decision based on one feature.

        # 1. Get the split decision rule.
        decision_rule = list(decision_tree.keys())[0]
        # The rule is in the form "index operator value". Split these components.
        rule_parts = decision_rule.split(" ")

        # 2. Get the feature index and split value
        feature_index = int(rule_parts[0])
        split_value = int(rule_parts[2]) 

        # 3. Get the feature value for the instance
        instance_value = instance[feature_index]

        # 4. Depending on the feature's value, choose left or right sub-tree.
        if instance_value < split_value:
            sub_tree = decision_tree[decision_rule][0]
        else:
            sub_tree = decision_tree[decision_rule][1]

        # 5. If the chosen sub-tree is a dictionary, recursively call the function again
        if isinstance(sub_tree, dict):
            return self.single_decision_tree_result(instance, sub_tree)

        # If it's not a dict, it's a leaf node so we have a classification.
        return sub_tree
        
    def random_forest_fitting(self, data, num_trees, ratio_bagged, smallest_leaf_size, num_features, max_depth):
        # Initialise list to store decision trees
        random_forest = []

        # Loop over the specified number of trees
        for _ in range(num_trees):
            # Perform bagging on dataset
            # Sample dataset with replacement
            # The ratio of bagged dataset size to total dataset size is ratio_bagged
            bagged_instances = np.random.randint(low=0, high=int(len(data)), size=int(round(ratio_bagged*len(data))))

            # Create bagged dataset using bagged_instances
            bagged_dataset = data[bagged_instances]

            # Train a decision tree on bagged dataset
            # Add tree to the random forest list
            decision_tree = self.decision_trees_fitting(bagged_dataset, smallest_leaf_size, max_depth, num_features)
            random_forest.append(decision_tree)

        # Return the random forest after training all trees
        return random_forest

    def decision_trees_fitting(self, data, depth_count=0, min_leaf_size=15, max_tree_depth=5, num_features=None):
                
        # Termination conditions: if data is pure, too few instances left, or max depth reached 
        if self.leaf_is_one_class(data) or len(data) < min_leaf_size or depth_count == max_tree_depth:
            # Return the most common target value as classification
            return self.plurality_classification(data)
        
        depth_count += 1  # Increase depth as we go down the tree

        # Select a random subset of features for consideration.
        features = self.random_subspace_method(data, num_features)

        # Find the best feature and its splitting value
        best_feature, decision_value = self.get_max_info_gain_split(data, features)

        # Split the data into two subsets: left and right
        left_data, right_data = self.get_split(data, best_feature, decision_value)

        # If either subset is empty, return the most common target value as classification
        if len(left_data) == 0 or len(right_data) == 0:
            return self.plurality_classification(data)

        # Represent the decision made at this node
        decision_detail = f'{best_feature} < {decision_value}'

        # Build subtrees in a recursive way
        left_tree = self.decision_trees_fitting(left_data, depth_count, min_leaf_size, max_tree_depth, num_features)
        right_tree = self.decision_trees_fitting(right_data, depth_count, min_leaf_size, max_tree_depth, num_features)

        # If left and right branches are the same, store just one.
        if left_tree == right_tree:
            return left_tree

        # Return a dict with decision detail as key and left and right branches as its value
        return {decision_detail: [left_tree, right_tree]}
        
    def get_max_info_gain_split(self, data, sampled_features):

        # Initialize the maximum information gain to negative infinity
        info_gain_max = float('-inf')

        for feature_index in sampled_features:
            for decision_values in sampled_features[feature_index]:  
                # Split the data into two nodes
                left_node, right_node = self.get_split(data, feature_index, decision_values)
                
                # Calculate the information gain
                info_gain = self.information_gain(data, left_node, right_node)

                # Check if this split gives a higher information gain than the current maximum
                if info_gain_max < info_gain:
                    info_gain_max = info_gain
                    split_on, split_feature_value = feature_index, decision_values

        return split_on, split_feature_value
    
    def get_split(self, data, split_feature, feature_value):
        
        # Extract the relevant feature column from the data
        column_values = data[:, split_feature]

        # Split the data all instances with the feature value less than the given feature value go to left node, and the rest go right
        data_left_node = data[column_values < feature_value]
        data_right_node = data[column_values >= feature_value]
        
        return data_left_node, data_right_node
    
    def plurality_classification(self, data):
        # The function finds the most frequent target (class) in the data

        # Extract the targets from the data
        targets = data[:, -1]

        # Find unique targets and their frequencies
        # np.unique returns unique elements and their counts
        unique_targets, target_counts = np.unique(targets, return_counts=True)

        # Find most common target
        # np.argmax gives the index of the max count, use this index to find the corresponding target
        max_count_index = target_counts.argmax()
        plurality_classification = unique_targets[max_count_index]

        return plurality_classification


    def random_subspace_method(self, data, k):
        
        # Get the number of features subtract 1 because the last column is target variable
        num_features = data.shape[1] - 1

        # Check if 'k' is valid. If not, select all features.
        if not k or k > num_features:
            selected_features = range(num_features)
        else:
            # Select 'k' feature indices at random, without replacement
            selected_features = random.sample(range(num_features), k)

        # Create a dictionary to store the selected_features_unique_values 
        selected_features_unique_values = {}
        
        for feature_index in selected_features:
            # Get unique values from the data for this feature_index
            unique_values = np.unique(data[:, feature_index])
            selected_features_unique_values[feature_index] = unique_values
            
        return selected_features_unique_values
        
    def entropy(self, data):
        
        # The last column of the data array is selected using data[:, -1]
        targets = data[:, -1]

        # np.unique identifies unique elements in 'targets' array and returns two values: unique elements (ignored here by using _),
        # and their corresponding frequency counts in 'target_frequency'.
        _, target_frequency = np.unique(targets, return_counts=True)

        # Convert counts to floats for accurate division in the next step.
        target_frequency = target_frequency.astype(float)

        # The probability of each target is calculated by dividing its frequency by the total frequency.
        probabilities = target_frequency / target_frequency.sum()

        # The entropy formula is -p*log2(p) where 'p' are the probabilities. 
        # It's summed over all unique targets. 
        # The more balanced the data (i.e., equal probabilities i.e. more disorder), the higher the entropy.
        entropy = sum(probabilities * -np.log2(probabilities))
        return entropy
    
    def information_gain(self, parent_data, data_left_node, data_right_node):
        # Number of instances in parent, left child, and right child nodes
        num_parent = float(len(data_left_node)+len(data_right_node))
        num_left = float(len(data_left_node))
        num_right = float(len(data_right_node))

        # Proportion of instances in left and right child nodes to parent node
        p_left = num_left / num_parent
        p_right = num_right / num_parent

        # Entropy of parent node
        parent_entropy = self.entropy(parent_data)

        # Entropy of left and right child nodes
        left_entropy = self.entropy(data_left_node)
        right_entropy = self.entropy(data_right_node)

        # Weighted average entropy of children nodes
        weighted_children_entropy = (p_left * left_entropy) + (p_right * right_entropy)

        # Information Gain = Entropy(parent) - Weighted_Average_Entropy(Children)
        info_gain = parent_entropy - weighted_children_entropy

        return info_gain
    
    def leaf_is_one_class(self, data):

        # Obtain the target values
        targets = data[:, -1]

        # Check and return whether all targets belong to the same class 
        return len(np.unique(targets)) == 1

    

    

    
    
