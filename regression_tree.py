import numpy as np


class Node(object):
    def __init__(self, parent, data_index, predict_value):
        self.parent = parent
        self.data_index = data_index
        self.predict_value = predict_value
        self.left = None
        self.right = None
        self.split_feature_index = None
        self.split_feature_value = None


class RegressionTree(object):
    def __init__(self, min_data_in_leaf):
        self.min_data_in_leaf = min_data_in_leaf
        self.tree = None

    def fit(self, X, ys):
        tree = Node(None, np.arange(X.shape[0]), np.mean(ys))
        cand_leaves = [tree]

        while len(cand_leaves) != 0:
            # 要求叶子分开
            min_squared_error = np.inf
            for leaf in cand_leaves:
                X_target = X[leaf.data_index]
                ys_target = ys[leaf.data_index]

                # 确定要划分的特征量
                for d in range(X_target.shape[1]):
                    # 按第d个特征数量排序数据
                    argsort = np.argsort(X_target[:, d])

                    # 查找具有最小平方误差的分部
                    for split in range(1, argsort.shape[0]):
                        # [0, split), [split, N_target)分割
                        tmp_left_data_index = argsort[:split]
                        tmp_right_data_index = argsort[split:]

                        left_predict = np.mean(ys_target[tmp_left_data_index])
                        left_squared_error = np.sum((ys_target[tmp_left_data_index] - left_predict) ** 2)
                        right_predict = np.mean(ys_target[tmp_right_data_index])
                        right_squared_error = np.sum((ys_target[tmp_right_data_index] - right_predict) ** 2)

                        squared_error = left_squared_error + right_squared_error
                        if squared_error < min_squared_error:
                            min_squared_error = squared_error
                            target_leaf = leaf
                            left_data_index = leaf.data_index[tmp_left_data_index]
                            right_data_index = leaf.data_index[tmp_right_data_index]
                            split_feature_index = d
                            split_feature_value = X_target[:, d][tmp_right_data_index[0]]

            # 当没有获得要分割的叶子时，它结束
            if min_squared_error == np.inf:
                break

           # 从候选集中删除要分割的叶子
            cand_leaves.remove(target_leaf)

           # 如果在划分之后分配给子节点的元素的数量少于min_data_in_leaf，则不将其分割为叶子
            if left_data_index.shape[0] < self.min_data_in_leaf or right_data_index.shape[0] < self.min_data_in_leaf:
                continue

           
            left_node = Node(target_leaf, np.sort(left_data_index), np.mean(ys[left_data_index]))
            right_node = Node(target_leaf, np.sort(right_data_index), np.mean(ys[right_data_index]))
            target_leaf.split_feature_index = split_feature_index
            target_leaf.split_feature_value = split_feature_value
            target_leaf.left = left_node
            target_leaf.right = right_node

            # 添加新叶子以分割候选叶子列表
            if left_node.data_index.shape[0] > 1:
                cand_leaves.append(left_node)
            if right_node.data_index.shape[0] > 1:
                cand_leaves.append(right_node)

        self.tree = tree

    def predict(self, X):
        ys_predict = []
        for xs in X:
            node = self.tree
            # 移动到达叶子
            while node.left is not None and node.right is not None:
                if xs[node.split_feature_index] < node.split_feature_value:
                    node = node.left
                else:
                    node = node.right
            ys_predict.append(node.predict_value)
        return np.array(ys_predict)
