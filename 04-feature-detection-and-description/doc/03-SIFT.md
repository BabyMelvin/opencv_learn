# SIFT（scale-invariant-feature-tansform）尺寸不变特征转换

## 原理
Harris他们是转动不变量，意味着即使转动图片，我们仍然找到相同的转角。很显然，转动的图片的中的转角还是转角。但是什么比例（scale）？如果图片缩放后，一个转角可能不是转角了。如下图，一个转角在一个小的图片中用小的窗口来说相对是一个平面。所以Harris转角不是缩放不变的。

<image src="image/03-01.jpg"/>

在2004年D.Lowe哥伦比亚大学，提出一个新的算法，SIFT`<<Distintive Image Features from Scale-Invariant Keypoints>>`主要提取关键点并比较描述符。(论文很简单，最好的SIFT的材料)。主要有四步SIFT算法：

### 1.缩放空间极值检测
从上面的图片显然我们不能使用相同的窗口函数去检测关键点在不同的缩放中。在小的转角，表现挺好的，但是检测打的转角我们需要放大窗口。所以，`scale-space`过滤将被使用。其中，高斯的拉普拉斯在这图片中用变量`σ`。LoG作为大对象检测，能够检测blob(大对象)由于`σ`的变化。简单来说，`σ`作为缩放参数。比如：上面的图片，小的`σ`高斯内核对应小的转角，大的`σ`高斯内核对应打的转角。所以，我们能够找到本地极大值通过缩放和空间，将会得到一系列(x,y,σ)值，意味着一些潜在的关键点(x,y)在`σ`缩放倍数。

但是LoG有代价较高，所以SIFT算法用不同的高斯，接近LoG。不同的高斯通过，用两种不同`σ`模糊图像，假设`σ`和`kσ`。这个过程通过高斯金字塔的八度图，如下图:

<image src="image/03-02.jpg"/>
一旦DoG被发现，将会搜索本地极值通过缩放和空间。例如，图片中的一像素和它相邻8个像素点组成9像素图。并且前一个比例。如果是本地极值，将会是潜在的关键点。也就意味着，关键点是这个比例最佳呈现。

<image src="image/03-03.jpg"/>

对于参数的不同，论文给出了一些经验值：数量的八度number of octaves=4,缩放水平number of scale levels=5,initial σ=1.6，sqrt(2)最为最优解。

## 2.点定位
一旦潜在关键点被发现，我们将会细化获得更高的精度结果。使用泰勒展开式获得更好位置极值，如果强度极值小于阀值(0.03论文中)，将会被丢弃。这个阀值被称为`contrastThreshold`。
DoG很大代表是边缘，需要移除。这里与Harris类似的概念被使用。使用2x2Hessian矩阵(H)去计算主曲率。我们使用Harris角检测对边使用，一个特征值比另一个大。所以使用一个简单的函数。

如果比率比阀值大，称为`edgeThreshold`，观点点被丢弃，论文中10.

所以排除任何低对比度关键点和边缘点，剩下是强感兴趣点。

## 3.方向分配
现在一个方向被分配到每个关键点实现图片旋转的不变性。一个关键点周围相邻点根据比例获得，梯度强度和方向在这区域可以计算的。一个有36个箱子覆盖360度直方图被创建。比重通过梯度迁都和高斯强度圆窗口`σ=1.5`比例的关键点。直方图中的极值点被获得，极值的80%的范围数据用来计算方向。同样比例同样位置，但是不同的方向，这将有助于匹配的稳定性。

## 4.关键点描述

现在关键点描述被创建。一个关键点周围16x16区域被使用。被分为16个4x4子块。每个子块，8bin方向直方图被创建。所以公有128bin值被获得。被表示为一个向量形成关键点的描述。另外，采取一些措施抵制光变化，旋转获得鲁棒性（健壮性）。

## 5.关键点匹配

两个图片的关键点通过识别最近相邻进行匹配。但是有些情况，第二个可能距离第一个太近了。发生可能是噪音或者其他原因。这时候，最短和次最短的利率被采用。如果大于0.8，将会被丢弃。大约丢弃掉90%不匹配的只有5%已匹配但被丢弃的。

**注意**:这个算法是需专利授权的。