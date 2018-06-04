# feature matching
学会如何一个图片和其他进行特征匹配

## Brute-Force匹配概概念
Brute-Force匹配很简单。取第一个集合中一个特征的描述，使用一些距离计算和第二个集合其他特征进行匹配。最接近的一个被返回。

## FLANN based Matcher
FLANN代表一个最接近相邻的快库。包含一些列在大数据集合中和高尺寸特征寻找最接近相邻集合算法。比BFMatcher大数据检测更快。