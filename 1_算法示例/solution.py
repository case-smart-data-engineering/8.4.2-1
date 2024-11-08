#!/usr/bin/env python3


import math
import pickle
import numpy as np
import scipy.optimize
import scipy.sparse as sp

class NeuralTensorNetwork(object):
    """
    网络初始化
    """
    def __init__(self, program_parameters):
        """
        从字典中提取程序参数
        """
        self.num_words           = program_parameters['num_words']
        self.embedding_size      = program_parameters['embedding_size']
        self.num_entities        = program_parameters['num_entities']
        self.num_relations       = program_parameters['num_relations']
        self.batch_size          = program_parameters['batch_size']
        self.slice_size          = program_parameters['slice_size']
        self.word_indices        = program_parameters['word_indices']
        self.activation_function = program_parameters['activation_function']
        self.lamda               = program_parameters['lamda']

        # 随机初始化词向量至[-r, r]
        r = 0.0001
        word_vectors = np.random.random((self.embedding_size, self.num_words)) * 2 * r - r
        r = 1 / math.sqrt(2 * self.embedding_size)

        # 初始化参数字典
        W = {}  # 张量网络的权重
        V = {}  # 线重性层的权
        b = {}  # 偏置项
        U = {}  # 输出层的权重

        for i in range(self.num_relations):
            # 初始化网络参数
            W[i] = np.random.random((self.embedding_size, self.embedding_size, self.slice_size)) * 2 * r - r
            V[i] = np.zeros((2 * self.embedding_size, self.slice_size))
            b[i] = np.zeros((1, self.slice_size))
            U[i] = np.ones((self.slice_size, 1))

        # 将参数展开成一个向量
        self.theta, self.decode_info = self.stackToParams(W, V, b, U, word_vectors)

    def stackToParams(self, *arguments):
        """
                将传入的参数展开成一个向量。

                参数:
                *arguments: 不定数量的参数，可以是字典或数组。

                返回:
                tuple: 包含参数向量和解码信息的元组。
                """
        theta       = []
        decode_info = {}

        for i in range(len(arguments)):
            # 提取第i个参数
            argument = arguments[i]
            if isinstance(argument, dict):
                # 如果参数是字典，则将配置存储为字典
                decode_cell = {}
                for j in range(len(argument)):
                    # 存储配置并连接到展开的向量
                    decode_cell[j] = argument[j].shape
                    theta          = np.concatenate((theta, argument[j].flatten()))

                # 存储参数的配置字典
                decode_info[i] = decode_cell

            else:
                # 若非=字典，存储配置并展开
                decode_info[i] = argument.shape
                theta          = np.concatenate((theta, argument.flatten()))

        return theta, decode_info


    # 使用展开的向量返回参数堆栈
    def paramsToStack(self, theta):
        """
            使用展开的参数向量返回参数堆栈。
            参数:
            theta (numpy.ndarray): 包含展开参数的向量。
            返回:
            list: 包含重新构建的参数的列表。
        """
        stack = []
        index = 0

        for i in range(len(self.decode_info)):
            # 获取第i个配置参数
            decode_cell = self.decode_info[i]

            if isinstance(decode_cell, dict):
                param_dict = {}
                for j in range(len(decode_cell)):
                    # 从展开向量中提取参数矩阵
                    param_dict[j] = theta[index : index + np.prod(decode_cell[j])].reshape(decode_cell[j])
                    index        += np.prod(decode_cell[j])
                stack.append(param_dict)

            else:
                stack.append(theta[index : index + np.prod(decode_cell)].reshape(decode_cell))
                index += np.prod(decode_cell)

        return stack


    # 对传递的矩阵应用激活函数
    def activationFunction(self, x):
        if self.activation_function == 0:
            return np.tanh(x)

        elif self.activation_function == 1:
            return (1 / (1 + np.exp(-x)))


    # 返回传递矩阵的激活函数的微分
    def activationDifferential(self, x):
        if self.activation_function == 0:
            return (1 - np.power(x, 2))

        elif self.activation_function == 1:
            return (x * (1 - x))

    def neuralTensorNetworkCost(self, theta, data_batch, flip):
        """
        计算神经张量网络的成本和梯度。

        参数:
        theta (numpy.ndarray): 展开的参数向量。
        data_batch (dict): 包含关系、实体及其负样本的批次数据。
        flip (bool): 指示是否反转实体向量用于负样本生成。

        返回:
        float: 计算的成本。
        """
        # 将参数向量重新转化为参数矩阵
        W, V, b, U, word_vectors = self.paramsToStack(theta)

        # 初始化实体向量和梯度
        entity_vectors = np.zeros((self.embedding_size, self.num_entities))
        entity_vector_grad = np.zeros((self.embedding_size, self.num_entities))

        # 计算每个实体的向量表示，取其单词向量的平均值
        for entity in range(self.num_entities):
            entity_vectors[:, entity] = np.mean(word_vectors[:, self.word_indices[entity]], axis = 1)

        cost = 0

        # 初始化梯度字典
        W_grad = {}; V_grad = {}; b_grad = {}; U_grad = {}

        for i in range(self.num_relations):
            # 获取当前关系的示例列表
            rel_i_list = (data_batch['rel'] == i)
            num_rel_i = np.sum(rel_i_list)

            e1 = data_batch['e1'][rel_i_list]
            e2 = data_batch['e2'][rel_i_list]
            e3 = data_batch['e3'][rel_i_list]

            entity_vectors_e1 = entity_vectors[:, np.array(e1, dtype=int)]
            entity_vectors_e2 = entity_vectors[:, np.array(e2, dtype=int)]
            entity_vectors_e3 = entity_vectors[:, np.array(e3, dtype=int)]

            # 处理负样本的实体向量
            if flip:
                entity_vectors_e1_neg = entity_vectors_e1
                entity_vectors_e2_neg = entity_vectors_e3
                e1_neg = e1
                e2_neg = e3

            else:
                entity_vectors_e1_neg = entity_vectors_e3
                entity_vectors_e2_neg = entity_vectors_e2
                e1_neg = e3
                e2_neg = e2

            # 添加偏置和线性变换
            preactivation_pos = np.zeros((self.slice_size, num_rel_i))
            preactivation_neg = np.zeros((self.slice_size, num_rel_i))

            for slice in range(self.slice_size):

                preactivation_pos[slice, :] = np.sum(entity_vectors_e1 *
                    np.dot(W[i][:, :, slice], entity_vectors_e2), axis = 0)
                preactivation_neg[slice, :] = np.sum(entity_vectors_e1_neg *
                    np.dot(W[i][:, :, slice], entity_vectors_e2_neg), axis = 0)

            preactivation_pos += b[i].T + np.dot(V[i].T, np.vstack((entity_vectors_e1, entity_vectors_e2)))
            preactivation_neg += b[i].T + np.dot(V[i].T, np.vstack((entity_vectors_e1_neg, entity_vectors_e2_neg)))

            activation_pos = self.activationFunction(preactivation_pos)
            activation_neg = self.activationFunction(preactivation_neg)

            score_pos = np.dot(U[i].T, activation_pos)
            score_neg = np.dot(U[i].T, activation_neg)

            wrong_filter = (score_pos + 1 > score_neg)[0]

            # 计算成本
            cost += np.sum(wrong_filter * (score_pos - score_neg + 1)[0])

            W_grad[i] = np.zeros(W[i].shape)
            V_grad[i] = np.zeros(V[i].shape)


            num_wrong = np.sum(wrong_filter)

            # 过滤错误分类的实例
            activation_pos            = activation_pos[:, wrong_filter]
            activation_neg            = activation_neg[:, wrong_filter]
            entity_vectors_e1_rel     = entity_vectors_e1[:, wrong_filter]
            entity_vectors_e2_rel     = entity_vectors_e2[:, wrong_filter]
            entity_vectors_e1_rel_neg = entity_vectors_e1_neg[:, wrong_filter]
            entity_vectors_e2_rel_neg = entity_vectors_e2_neg[:, wrong_filter]

            e1     = e1[wrong_filter]
            e2     = e2[wrong_filter]
            e1_neg = e1_neg[wrong_filter]
            e2_neg = e2_neg[wrong_filter]

            U_grad[i] = np.sum(activation_pos - activation_neg, axis = 1).reshape(self.slice_size, 1)

            temp_pos_all = U[i] * self.activationDifferential(activation_pos)
            temp_neg_all = - U[i] * self.activationDifferential(activation_neg)


            b_grad[i] = np.sum(temp_pos_all + temp_neg_all, axis = 1).reshape(1, self.slice_size)

            values = np.ones(num_wrong)
            rows   = np.arange(num_wrong + 1)

            e1_sparse     = sp.csr_matrix((values, e1, rows), shape = (num_wrong, self.num_entities))
            e2_sparse     = sp.csr_matrix((values, e2, rows), shape = (num_wrong, self.num_entities))
            e1_neg_sparse = sp.csr_matrix((values, e1_neg, rows), shape = (num_wrong, self.num_entities))
            e2_neg_sparse = sp.csr_matrix((values, e2_neg, rows), shape = (num_wrong, self.num_entities))

            for k in range(self.slice_size):
                # 计算 W 的梯度
                temp_pos = temp_pos_all[k, :].reshape(1, num_wrong)
                temp_neg = temp_neg_all[k, :].reshape(1, num_wrong)

                W_grad[i][:, :, k] = np.dot(entity_vectors_e1_rel * temp_pos, entity_vectors_e2_rel.T) \
                    + np.dot(entity_vectors_e1_rel_neg * temp_neg, entity_vectors_e2_rel_neg.T)
                # 计算 V 的梯度
                V_grad[i][:, k] = np.sum(np.vstack((entity_vectors_e1_rel, entity_vectors_e2_rel)) * temp_pos
                    + np.vstack((entity_vectors_e1_rel_neg, entity_vectors_e2_rel_neg)) * temp_neg, axis = 1)

                V_pos = V[i][:, k].reshape(2*self.embedding_size, 1) * temp_pos
                V_neg = V[i][:, k].reshape(2*self.embedding_size, 1) * temp_neg

                # 计算实体向量的梯度
                entity_vector_grad += V_pos[:self.embedding_size, :] * e1_sparse + V_pos[self.embedding_size:, :] * e2_sparse \
                    + V_neg[:self.embedding_size, :] * e1_neg_sparse + V_neg[self.embedding_size:, :] * e2_neg_sparse

                entity_vector_grad += (np.dot(W[i][:, :, k], entity_vectors[:, np.array(e2, dtype=int)]) * temp_pos) * e1_sparse \
                    + (np.dot(W[i][:, :, k].T, entity_vectors[:, np.array(e1, dtype=int)]) * temp_pos) * e2_sparse \
                    + (np.dot(W[i][:, :, k], entity_vectors[:, np.array(e2, dtype=int)]) * temp_neg) * e1_neg_sparse \
                    + (np.dot(W[i][:, :, k].T, entity_vectors[:, np.array(e1_neg, dtype=int)]) * temp_neg) * e2_neg_sparse

            # 平均梯度
            W_grad[i] /= self.batch_size
            V_grad[i] /= self.batch_size
            b_grad[i] /= self.batch_size
            U_grad[i] /= self.batch_size

        # 初始化单词向量的梯度
        word_vector_grad = np.zeros(word_vectors.shape)

        for entity in range(self.num_entities):

            entity_len = len(self.word_indices[entity])
            word_vector_grad[:, self.word_indices[entity]] += \
                np.tile(entity_vector_grad[:, entity].reshape(self.embedding_size, 1) / entity_len, (1, entity_len))


        word_vector_grad /= self.batch_size
        cost             /= self.batch_size

        theta_grad, d_t = self.stackToParams(W_grad, V_grad, b_grad, U_grad, word_vector_grad)

        cost       += 0.5 * self.lamda * np.sum(theta * theta)
        theta_grad += self.lamda * theta

        return cost, theta_grad


    def computeBestThresholds(self, dev_data, dev_labels):
        # 从模型参数theta中获取所需的权重和向量
        W, V, b, U, word_vectors = self.paramsToStack(self.theta)
        # 初始化实体向量矩阵，大小为(embedding_size, num_entities)
        entity_vectors = np.zeros((self.embedding_size, self.num_entities))
        # 遍历所有实体，计算每个实体的向量（取该实体对应单词向量的平均值）
        for entity in range(self.num_entities):
            entity_vectors[:, entity] = np.mean(word_vectors[:, self.word_indices[entity]], axis = 1)
        # 初始化开发集得分矩阵，大小与dev_labels相同
        dev_scores = np.zeros(dev_labels.shape)

        for i in range(dev_data.shape[0]):
            # 获取关系索引
            rel = dev_data[i, 1]
            # 获取两个实体的向量，并重塑为列向量
            entity_vector_e1 = entity_vectors[:, dev_data[i, 0]].reshape(self.embedding_size, 1)
            entity_vector_e2 = entity_vectors[:, dev_data[i, 2]].reshape(self.embedding_size, 1)
            # 将两个实体向量垂直堆叠
            entity_stack = np.vstack((entity_vector_e1, entity_vector_e2))

            # 计算该数据点的得分（根据模型的具体公式）
            for k in range(self.slice_size):

                dev_scores[i, 0] += U[rel][k, 0] * \
                   (np.dot(entity_vector_e1.T, np.dot(W[rel][:, :, k], entity_vector_e2)) +
                    np.dot(V[rel][:, k].T, entity_stack) + b[rel][0, k])

        # 获取开发集得分的最小值和最大值
        score_min = np.min(dev_scores)
        score_max = np.max(dev_scores)

        # 初始化最佳阈值和最佳准确率矩阵
        best_thresholds = np.empty((self.num_relations, 1))
        best_accuracies = np.empty((self.num_relations, 1))

        # 将最佳阈值和最佳准确率初始化为默认值
        for i in range(self.num_relations):
            best_thresholds[i, :] = score_min
            best_accuracies[i, :] = -1

        # 设置阈值搜索的起始值和步长
        score_temp = score_min
        interval   = 0.01

        # 遍历可能的阈值范围，找到每个关系的最佳阈值和准确率
        while(score_temp <= score_max):
            for i in range(self.num_relations):
                # 获取当前关系的数据点索引
                rel_i_list    = (dev_data[:, 1] == i)
                # 根据当前阈值进行预测（得分小于等于阈值为-1，否则为1）
                predictions   = (dev_scores[rel_i_list, 0] <= score_temp) * 2 - 1
                # 计算当前阈值下的准确率
                temp_accuracy = np.mean((predictions == dev_labels[rel_i_list, 0]))

                # 如果当前准确率高于之前记录的最佳准确率，则更新最佳准确率和阈值
                if(temp_accuracy > best_accuracies[i, 0]):
                    best_accuracies[i, 0] = temp_accuracy
                    best_thresholds[i, 0] = score_temp

            # 增加阈值，继续搜索
            score_temp += interval

        # 将计算得到的最佳阈值保存到类的属性中
        self.best_thresholds = best_thresholds

    def getPredictions(self, test_data):
        # 从模型参数theta中获取所需的权重和向量
        W, V, b, U, word_vectors = self.paramsToStack(self.theta)
        # 初始化实体向量矩阵，大小为(embedding_size, num_entities)
        entity_vectors = np.zeros((self.embedding_size, self.num_entities))

        # 遍历所有实体，计算每个实体的向量（取该实体对应单词向量的平均值）
        for entity in range(self.num_entities):
            entity_vectors[:, entity] = np.mean(word_vectors[:, self.word_indices[entity]], axis = 1)

        # 初始化预测结果矩阵，大小为(test_data.shape[0], 1)
        predictions = np.empty((test_data.shape[0], 1))

        for i in range(test_data.shape[0]):
            # 获取关系索引
            rel = test_data[i, 1]
            # 获取两个实体的向量，并重塑为列向量
            entity_vector_e1  = entity_vectors[:, test_data[i, 0]].reshape(self.embedding_size, 1)
            entity_vector_e2  = entity_vectors[:, test_data[i, 2]].reshape(self.embedding_size, 1)

            # 将两个实体向量垂直堆叠
            entity_stack = np.vstack((entity_vector_e1, entity_vector_e2))
            test_score   = 0

            for k in range(self.slice_size):

                test_score += U[rel][k, 0] * \
                   (np.dot(entity_vector_e1.T, np.dot(W[rel][:, :, k], entity_vector_e2)) +
                    np.dot(V[rel][:, k].T, entity_stack) + b[rel][0, k])

            # 根据测试得分和最佳阈值进行预测
            # 如果测试得分小于等于阈值，则预测关系存在（标记为1）
            # 否则，预测关系不存在（标记为-1）
            if(test_score <= self.best_thresholds[rel, 0]):
                predictions[i, 0] = 1
            else:
                predictions[i, 0] = -1

        return predictions

def getTestData(file_name, entity_dictionary, relation_dictionary):
    file_object = open(file_name, 'r')
    data        = file_object.read().splitlines()

    # 获取数据的条目数
    num_entries = len(data)

    # 初始化测试数据矩阵和标签矩阵
    # 测试数据矩阵大小为(num_entries, 3)，用于存储实体和关系的索引
    # 标签矩阵大小为(num_entries, 1)，用于存储标签（1或-1）
    test_data   = np.empty((num_entries, 3))
    labels      = np.empty((num_entries, 1))

    index = 0

    for line in data:
        # 将行拆分为实体1、关系、实体2和标签
        entity1, relation, entity2, label = line.split()
        # 使用字典将实体和关系转换为索引，并存储到测试数据矩阵中
        test_data[index, 0] = entity_dictionary[entity1]
        test_data[index, 1] = relation_dictionary[relation]
        test_data[index, 2] = entity_dictionary[entity2]

        # 将标签转换为数值（1或-1）并存储到标签矩阵中
        if label == '1':
            labels[index, 0] = 1
        else:
            labels[index, 0] = -1

        index += 1

    return test_data, labels

def getWordIndices(file_name):
    """
        从文件中获取单词索引和单词数量。

        参数:
        file_name (str): 包含单词索引字典的文件路径。

        返回:
        tuple: 包含单词索引和单词数量的元组。
    """
    import pickle

    with open(file_name, 'rb') as file:
        word_dictionary = pickle.load(file, encoding='latin1')

    num_words = word_dictionary['num_words']
    word_indices = word_dictionary['word_indices']

    return word_indices, num_words

def getTrainingData(file_name, entity_dictionary, relation_dictionary):
    """
        从文件中获取训练数据并将其转换为索引形式。

        参数:
        file_name (str): 包含训练数据的文件路径。
        entity_dictionary (dict): 实体字典，将实体名称映射到索引。
        relation_dictionary (dict): 关系字典，将关系名称映射到索引。

        返回:
        tuple: 包含训练数据和示例数量的元组。
    """
    file_object = open(file_name, 'r')
    data = file_object.read().splitlines()

    num_examples = len(data)
    training_data = np.empty((num_examples, 3))

    index = 0

    for line in data:
        entity1, relation, entity2 = line.split()

        training_data[index, 0] = entity_dictionary[entity1]
        training_data[index, 1] = relation_dictionary[relation]
        training_data[index, 2] = entity_dictionary[entity2]

        index += 1

    return training_data, num_examples


def getDictionary(file_name):
    """
        读取文件并构建字典，将每个实体映射到一个唯一的索引。

        参数:
        file_name (str): 包含实体名称的文件路径。

        返回:
        tuple: 包含字典和条目数量的元组。
    """
    file_object = open(file_name, 'r')
    data = file_object.read().splitlines()

    dictionary = {}
    index = 0

    # 遍历每个实体，添加到字典并分配索引
    for entity in data:
        dictionary[entity] = index
        index += 1
    num_entries = index

    return dictionary, num_entries


def getProgramParameters():
    program_parameters = {}
    # 设置参数
    program_parameters['embedding_size']      = 128    # size of a single word vector
    program_parameters['slice_size']          = 3      # number of slices in tensor
    program_parameters['num_iterations']      = 50    # number of optimization iterations
    program_parameters['batch_size']          = 32  # training batch size
    program_parameters['corrupt_size']        = 10     # corruption size
    program_parameters['activation_function'] = 0      # 0 - tanh, 1 - sigmoid
    program_parameters['lamda']               = 0.0001 # regulariization parameter
    program_parameters['batch_iterations']    = 5      # optimization iterations for each batch

    return program_parameters

def neuralTensorNetwork():
    # 获取程序参数
    program_parameters = getProgramParameters()

    # 从程序参数中提取各个参数
    num_iterations = program_parameters['num_iterations']
    batch_size = program_parameters['batch_size']
    corrupt_size = program_parameters['corrupt_size']
    batch_iterations = program_parameters['batch_iterations']
    print("====完成程序参数初始化====")

    # 获取实体和关系的字典及其数量
    entity_dictionary, num_entities = getDictionary('entities.txt')
    relation_dictionary, num_relations = getDictionary('relations.txt')

    # 获取训练数据及其数量
    training_data, num_examples = getTrainingData('train.txt', entity_dictionary, relation_dictionary)

    # 获取单词索引及其数量
    word_indices, num_words = getWordIndices('wordIndices.p')

    # 更新程序参数
    program_parameters['num_entities'] = num_entities
    program_parameters['num_relations'] = num_relations
    program_parameters['num_examples'] = num_examples
    program_parameters['num_words'] = num_words
    program_parameters['word_indices'] = word_indices
    print("====开始训练网络====")

    # 初始化神经张量网络
    network = NeuralTensorNetwork(program_parameters)
    print("====完成网络初始化====")

    for i in range(num_iterations):
        # 随机选择一个批次的训练数据
        batch_indices = np.random.randint(num_examples, size = batch_size)
        data = {}
        data['rel'] = np.tile(training_data[batch_indices, 1], (1, corrupt_size)).T
        data['e1'] = np.tile(training_data[batch_indices, 0], (1, corrupt_size)).T
        data['e2'] = np.tile(training_data[batch_indices, 2], (1, corrupt_size)).T
        data['e3'] = np.random.randint(num_entities, size = (batch_size * corrupt_size, 1))

        # 根据随机值选择优化参数
        if np.random.random() < 0.5:
            opt_solution = scipy.optimize.minimize(network.neuralTensorNetworkCost, network.theta,
                args = (data, 0,), method = 'L-BFGS-B', jac = True, options = {'maxiter': batch_iterations})
        else:
            opt_solution = scipy.optimize.minimize(network.neuralTensorNetworkCost, network.theta,
                args = (data, 1,), method = 'L-BFGS-B', jac = True, options = {'maxiter': batch_iterations})

        # 更新网络参数
        network.theta = opt_solution.x
        # 打印训练批次进度
        progress = (i + 1) / num_iterations * 100
        print(f'Iteration {i + 1}/{num_iterations} - Progress: {progress:.2f}%')

    print("====训练结束====")

    dev_data, dev_labels = getTestData('dev.txt', entity_dictionary, relation_dictionary)
    test_data, test_labels = getTestData('test.txt', entity_dictionary, relation_dictionary)

    dev_data = np.array(dev_data, dtype=int)
    test_data = np.array(test_data, dtype=int)

    network.computeBestThresholds(dev_data, dev_labels)
    predictions = network.getPredictions(test_data)

    print("Accuracy:", np.mean((predictions == test_labels)))

if __name__ == '__main__':
    neuralTensorNetwork()
