# Hough圆形变换
##原理
圆的数学表示`(x-x0)^2+(x-y0)^2=r^2`其中，`(x0,y0)`是圆心，r是半径。与三个参数确定一个圆。需要3D累积，对Hough将会效率非常的低。OpenCV使用更复杂的方法，`Hough Gradiend Method`利用边缘梯度变换。

