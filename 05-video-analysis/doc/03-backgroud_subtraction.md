# background subtraction
本章将会学习去除背景的方法

## 概要
去除背景是很多基本视觉应用的主要预处理步骤。例如：这些案例：如静态相机获得一些进出房间来数拜访者人数，或者交通相机获取车辆信息等。这些示例中，首先获取人或车辆的信息。技术上讲，你要从静态背景中获得移动的前景。

如果只有一个背景的图片，房间的图片没有拜访者，马路上车辆等，很容易的工作了。只需要取出新图片中的背景。你只获得了物体的前景。但是大多数情况下，没有这样的图片情形。所以需要从无论什么样的图片中获得背景。当车辆有阴影将会变得更复杂。既然阴影也是在移动，简单的去除将会标记阴影也为前景色。这是个复杂的事情。

几种算法来解决这个问题。OpenCV实现三种这类方法很好使用。

## backgroudSutractorMOG
一个高斯混合前景/背景部分算法。在2001年论文<<An improved adaptive backgroud mixture model for real-time tracking with shadow detection>>.使用一个方法通过K高斯混合分布(K=3~5)来对每个背景像素建模。混合比重代表时间比例，这些颜色保留在场景中。可能背景颜色是那些保留时间更长更静止的颜色。

## backgroundSubtractorMOG2
同样采用高斯混合基于背景/前景部分算法。基于两篇文章`<<improved adaptive Gausian mixture model for background subtration>>`和`<<Efficient Adaptive Density Estimation per Image Pixel for the Task of Background Subtration>>`这个算法重要特征是选择合适高斯分布对每个像素。对变化的场景如透明度改变有很好的自适应性。

## backgroundSubtractorGMG
这个算法将统计图片估值和每个像素贝叶斯分类结合起来。`<<Visual Tracking of Human Visitors under Variable-Lighting Conditions for a Responsive Audio Art Installation>>`2012年。

它使用一些帧作为背景模型(120帧)。用贝叶斯分类对前景物体分类识别可能的前景。估计值被采用，更新观测比旧的观测中的多，来适应亮度的变化。一些形态学的操作，关闭和开放来去除不想要的噪音。前几帧可能是黑屏。