# depth map from stereo images
## 基础

上节看到了极线约束和其他相关术语。如果有同一场景的两张图片，我们能直观上获得图片的深度。下面图片一些直觉算术说明。

<image src="image/04-01.jpg"/>
上图包含了三角等式。写出等式，得到下面结果：

<image src="image/04-02.png"/>

x和x'图片两点的距离。先对应的是场景3D点和相机中点。B是连个相机的距离，f是相机聚焦长度。简单说上面等式场景中图像点深度和图片与相机中心点成反比。有这个信息，我们获得图片中所有像素点深度。

所以找到两个图片的匹配。我们已经找到如何极线使得操作更快和更精度。一旦匹配，将会找到不相等。

