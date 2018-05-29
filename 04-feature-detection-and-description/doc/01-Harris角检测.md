# Harris corner detection
## 原理
我们知道转角是图片中强度各个方向变化大的区域。早期由`Chris Harris&Mike Stephens`在他们论文中提供找到转角的方法。`<<A Combined corner and Edge Detector>>`发表在1988年，现在称为Harris Corner Detector.采用算术表达式来表达，主要在各个方向对位移(u,v)寻找强度的不同。表达式如下:

<image src="image/01-1.png"/>

* `window function`：一个`rectangle window`或 `gaussian window`，主要给出窗口下像素的比重。

为了转角检测，必须获得函数`E(u,v)`的最大值。也就是意味着使得第二个项最大。将泰特展开式带入到上式，最后能够得到下面表达式:

<image src="image/01-2.png"/>
其中：
<image src="image/01-3.png"/>

其中`Ix`和`Iy`代表x和y方向的导数。(通过`cv2.Sobel()`能够容易获得)。然后进入主要部分，创建一个得分，也就是一个公式来决定窗口中是否包含转角:

<image src="image/01-04.png"/>

其中:

* `det(M)=λ1*λ2`
* `trace(M)=λ1+λ2`
* `λ1`和`λ2`是M的特征值

多以特征值决定图形区域是角，边或者面。

* 当`λ1`,`λ2`,很小时，`|R|`也很小。图形区域为平面。
* 当`λ1>>λ2`或`λ1<<λ2`时，`R<0`，图形区域为边缘。
* 当`λ1`,`λ2`,很大时，`R`也很大，并且`λ1~λ2`.区域是角。

<image src="image/01-05.jpg"/>

Harris Corner Detection结果是这个灰色得分的图解。合适的阀值给你图片中的转角。