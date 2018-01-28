'''
File Name:     kNN
Description: 
Author:        jwj
Date:          2018/1/18
'''
__author__ = 'jwj'

import numpy as np
import operator     # 运算符模块
from os import listdir    # listdir列出给定目录中的文件名


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# 分类器实现，参数：输入测试数据、训练集、训练标签、k值
def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]                      # 数据集大小，即训练样本数量
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # 计算两矩阵元素级别上的差
    sqDiffMat = diffMat ** 2                            # 所有元素求平方
    sqDistances = sqDiffMat.sum(axis=1)                 # 横轴上所有元素求和   [ 2.21  2.    0.    0.01]
    # print(sqDistances)
    distances = sqDistances ** 0.5                      # 开根号  [ 1.48660687  1.41421356  0.          0.1       ]
    sortedDistIndicies = distances.argsort()            # 从小到大排序后返回其索引 [2 3 1 0]
    classCount = {}                                     # 空字典，存储前k个数据的标签及其计数
    for i in range(k):
        votelabel = labels[sortedDistIndicies[i]]       # 依次获取该数据的标签
        classCount[votelabel] = classCount.get(votelabel, 0) + 1  # 若字典中存在该标签，则直接加1；若不存在，则先初始化为0，再加1
    # print(classCount)  # {'B': 2, '-A': 1}
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # print(classCount.items())   # dict_items([('B', 2), ('A', 1)])
    # print(sortedClassCount)   # [('B', 2), ('A', 1)]
    return sortedClassCount[0][0]  # B


def file2matrix(filename):
    with open(filename, 'r') as file:                        # 打开文件
        arrayLines = file.readlines()                        # 读取文件中所有行数据
        numberOfLines = len(arrayLines)                      # 文件行数
        returnMat = np.zeros((numberOfLines, 3))             # 创建返回的矩阵，初始化为0
        classLabelVector = []
        index = 0
        for line in arrayLines:                              # 遍历行
            line = line.strip()                              # 去除空格
            listFromLine = line.split('\t')                  # 根据\t划分为列表
            returnMat[index, :] = listFromLine[0:3]          # 获取该行的前3个数据
            classLabelVector.append(int(listFromLine[-1]))   # 获取该样本数据的标签值
            index += 1
    return returnMat, classLabelVector                       # 返回样本数据数组和标签


def autoNorm(dataSet):
    minValues = dataSet.min(0)                          # 返回数据集中每列上的最小值
    maxValues = dataSet.max(0)                          # 每列上的最大值
    ranges = maxValues - minValues                      # 求差，得数据的范围
    normDataSet = np.zeros(shape=np.shape(dataSet))     # 根据shape创建数组，全为0
    m = dataSet.shape[0]                                # 样本数量
    normDataSet = dataSet - np.tile(minValues, (m, 1))  # 公式
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minValues


def dataClassTest():
    testRatio = 0.20                                            # 定义数据集中为测试样本的比例
    dataSet, dataLabels = file2matrix("datingTestSet2.txt")   # 读取数据
    normMat, ranges, minValues = autoNorm(dataSet)            # 归一化处理
    m = normMat.shape[0]                                      # 样本数量
    numTestVecs = int(m * testRatio)                            # 确定测试样本的数量
    errorCount = 0.0                                          # 错误数量统计
    for i in range(numTestVecs):
        classifierResult = classify(normMat[i, :], normMat[numTestVecs:m, :], dataLabels[numTestVecs:m], 3)  # 传入测试数据、训练集、训练标签、k值
        print("classify: %d, read answer: %d" % (classifierResult, dataLabels[i]))
        if classifierResult != dataLabels[i]:                # 如果预测不正确，则统计加1
            errorCount += 1
    print("total error rate is: %f" % (errorCount / float(numTestVecs)))  # 打印错误率


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']              # 定义三种喜欢程度，对应数据集中标签 1,2,3
    ffMiles = float(input("frequent flier miles earned per year?"))              # 输入每年飞行里程数
    percentTats = float(input("percentage of time spent playing video games?"))  # 输入玩游戏所耗时间百分比
    iceCream = float(input("liters of ice cream consumed per week?"))            # 输入每周消费冰激凌公升数
    dataArray, dataLabels = file2matrix("datingTestSet2.txt")                    # 从txt中获取训练数据
    normMat, ranges, minVals = autoNorm(dataArray)                               # 归一化处理
    inArray = np.array([ffMiles, percentTats, iceCream])                         # 对测试数据处理，整合成数组
    normInArray = (inArray - minVals) / ranges                                   # 对数据做归一化处理
    classifyResult = classify(normInArray, normMat, dataLabels, 3)               # 分类，k=3
    print("you will probably like this person: ", resultList[classifyResult - 1])


# 将图像转为向量
def img2vector(filename):
    returnVector = np.zeros((1, 1024))                       # 初始化0数组，1行1024列
    with open(filename, 'r') as file:                        # 读取文件
        for i in range(32):                                  # 遍历行
            lineStr = file.readline()                        # 读取行
            for j in range(32):                              # 遍历列
                returnVector[0, 32 * i + j] = int(lineStr[j])# 将该行上第j个数据存进数组第i行第j列中
    return returnVector


def handwritingClassTest():
    hwLabels = []                                        # 列表，存放训练数据集标签
    trainingFileList = listdir("digits/trainingDigits")  # 列出给定目录中的文件名
    m = len(trainingFileList)                            # 训练样本数
    trainingMat = np.zeros((m, 1024))                    # 初始化全0矩阵 m行1024列
    for i in range(m):                                   # 遍历训练数据
        fileNameStr = trainingFileList[i]                # 获取文件名全称，如 3_107.txt
        fileStr = fileNameStr.split('.')[0]              # 根据 . 划分，获取文件名 如 3_107
        classNum = int(fileStr.split('_')[0])            # 根据 _ 划分，获取该文件表示的真实数字 如 3
        hwLabels.append(classNum)                        # 将该数字标签放入训练集标签列表中
        trainingMat[i, :] = img2vector('digits/trainingDigits/%s' % fileNameStr)  # 调用函数，将第i个文件内的内容转化为数组，并存储
    testFileList = listdir("digits/testDigits")     # 列出测试集目录中的文件名
    errorCount = 0.0                                # 错误统计
    mTest = len(testFileList)                       # 测试集大小
    for i in range(mTest):                          # 遍历测试集
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNum = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector("digits/testDigits/%s" % fileNameStr)
        classifyResult = classify(vectorUnderTest, trainingMat, hwLabels, 3)  # 调用函数，预测数字
        print("the classifier: %d, the real value: %d" % (classifyResult, classNum))
        if classifyResult != classNum:
            errorCount += 1.0
    print("total number of errors: %d" % errorCount)
    print("total error rate: %f" % (errorCount / float(mTest)))   # 错误率
