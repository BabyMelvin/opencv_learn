# ORB
Oriented FAST and Rotated BRIEF

## 原理
作为一个OpenCV粉丝，最重要的事情是ORB是来自于“OpenCV Labs”.这个算法是在2011年，Ethan Rublee,Vincent Rabaud《ORB:An efficient alternative to SIRF or SURF》.简单说，以计算为代价，匹配性能，主要是专利，SIFT和SURF是很好的选择。是的SIRF和SURF是使用专利的，需要付费使用。但是ORB不是的。

ORB是对FAST关键点和BRIEF描述的融合，并且很多修改来提供性能。首先是使用FAST找到关键点，然后用Harris角检测找到他们中前N个点。也使用塔来算不同缩放的特征。唯一问题是FAST不能计算方向。那么方向不变怎么办？作者提出如下修改：

1.计算patch在角中心位置强度中心比重。向量方向从这个角点中心到中心位置给出的方向。提高旋转不变性，x,y应该在半径为r圆区域进行计算。

2。关于描述符，ORB使用BRIEF描述符。但是我们知道BRIEF对旋转很差。所以ORB做的是根据关键点方向。任何特征在(xi,yi)位置n二值，定义为2xn矩阵，S包含这些像素的坐标。然后使用path方向@，矩阵旋转被找到，旋转S来获得S@。

ORB离散角度，增量为2π/30（12°）并且构成一个预计算BRIEF表的检查表。只要方向@关键点是通过views一直是，正确的S@将会被要做计算描述。

BRIEF有一个重要特性是位特征好一个大的变量平均值大概是0.5.但是一旦旋转关键点方向，将会失去属性变成更加分散的。变化越大，特征越分散，虽然对不同输入有不同反应。另一个想要的属性是无关测试，因为每个测试将会对结果有影响。解决这些问题，ORB在大量测试搜索，找到高变量接近0.5，只要他们不相关。结果成为rBRIEF.

为了描述匹配，multi-probe LSH对传统的LSH提高，被采用。论文说ORB比SURF和SIFT更快，ORB描述符比SURF工作性能更好。ORB对低功耗设备如全景图拼接很好。