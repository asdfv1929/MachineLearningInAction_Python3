'''
File Name:    main
Description:  主函数，主要调用kNN.py中的函数
Author:       jwj
Date:         2018/1/18
'''
__author__ = 'jwj'

import kNN

if __name__ == '__main__':
    group, labels = kNN.createDataSet()
    label = kNN.classify([0, 0], group, labels, 3)
    print(label)

    dataArray, dataLabels = kNN.file2matrix("datingTestSet2.txt")
    kNN.autoNorm(dataArray)

    normMat, ranges, minVals = kNN.autoNorm(dataArray)
    # print(normMat)

    # kNN.dataClassTest()
    # kNN.classifyPerson()

    kNN.handwritingClassTest()