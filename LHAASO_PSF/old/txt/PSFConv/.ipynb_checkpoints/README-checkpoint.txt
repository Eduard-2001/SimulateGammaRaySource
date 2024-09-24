首先确保psf.npy以及PSF.py在同一个目录里

然后在你的.py文件中 import PSF，并且需要定义如下三个变量：

1. src, 一个n*n的二维numpy数组，包含源（没有经过psf卷积）的数据
2. xx和yy,分别都是n*n的二维numpy数组，包含了src中每一个点的x和y坐标。
   xx和yy可以由np.meshgrid(x,y)生成，其中x,y为长度为n的一维数组，包含横纵轴的坐标。

接着在你的.py文件中添加psf卷积函数
PSFConv = PSF.csinterp(index)
这里index是一个整型变量，选取不同的index值取决于你想要用哪个能量段的PSF。
     index  对应文件
        0 : 1.4.txt
        1 : 1.6.txt
        2 : 2.8.txt
        3 : 2.0.txt
        4 : 2.2.txt
        5 : 2.4.txt
        6 : 2.6.txt
        7 : 2.8.txt
        8 : 3.0.txt
然后得到的PSFConv可用于对src进行卷积：
src_with_psf = PSFConv(src,xx,yy)

这样src_with_psf就是卷积psf后的天图了。