论文地址：https://www.mdpi.com/1099-4300/22/2/242?type=check_update&version=1

#### 2022年4月7日：
文章内容与作者提供的实际实现源代码存在出入，有些地方含糊不清；
由于存在模糊不清的地方，也许无法完全复现，可能会终止实验。

**问题**
> 1. 计算$H_{uv}$，那么这个$H_{uv}$是表示哪个点呢？u还是v?如果是u，那换一个邻居H的值就变了。看了源代码，作者说和做是两回事。
>
> 2. 像all_degree，没理解错的话是度总和，直接就简单的等于“图中所有顶点-1”，这在简单图上是成立的，但问题是这些数据集是简单图吗？
> 
> 3. 平均度的计算，也没指明是怎么计算的。看了源代码才知道是$\frac{edge\times 2} {node}$; E\<k\>的计算在文章中也写得云里雾里;