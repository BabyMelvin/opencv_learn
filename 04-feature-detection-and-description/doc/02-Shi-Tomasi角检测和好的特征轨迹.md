## Shi-Tomasi 角检测& good feature to track
### 原理
之前看到了Harris角检测。最近在1994年，J.Shi和C.Tomasi对Harris检测进行小的修改，在他们论文`<<Good Features to Track>>`,结果表明有更好的效果。这前得分函数在Harris Corner检测是：

<image src="image/02-01.png"/>

替换为Shi-Tomasi提出的：

<image src="image/02-02.png"/>

如果比阀值大，被认为是转角。如果我们画出`λ1-λ2`空间像Harris Corner Detector的图片如下:

<image src="image/02-03.png"/>

只有当`λ1，λ2`大于最小值`λmin`，被认为是转角(绿色区域).
