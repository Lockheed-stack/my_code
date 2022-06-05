## Start
> 大概的计划框架:
> 1. 看看相关论文，对其改进后以利用
> 2. 将该算法与经典算法比较，目前确定比较的算法有：link prediction、Facebook的EdgeRank、SNS算法。*（没时间的话就从中挑2个进行比较吧）*
> 3. 评价的方法：precision、recall、F1-measure、AUC、是否不同的算法都推荐了相同的好友、运行算法的时间等。

## 进度1
> 对数据集的初步分析:
> 1. 数据集scholat social network 有171个连通分量, 最大连通分量有9583个顶点，其直径为10; 剩余连通分量最大直径为4,大部分为1.
> 2. SCHOLAT LINK Prediction 有164个连通分量, 最大连通分量有9428个顶点,其直径为13; 剩余连通分量直径最大为5，多数直径为1。
> 3. 没有孤立的点
## 进度2
> 计算相似度的方法：
> 1. Jaccard distance/index:
Jaccard index是用来衡量两个不同集合中的共同和不同项目的相似性指数。  Jaccard指数的百分比越高，在不同集合中发现的项目的相似度就越高。该指数更适用于大型数据集。
Jaccard distance $(A,B)=\frac{|A\cap B|}{|A \cup B|}$，A、B是两个不同的集合。
> 2. Cosine distance:
余弦距离是A和B相交与A和B元素的点积的模。Cosine distance = $\frac{|A\cap B|}{|A|\cdot |B|}$。
>
>
> 相关算法:
> 1. pageRank
> 2. Shortest distance/path: 它涉及计算图中特定节点之间的最短距离/路径。
> 3. Adamic/Adar index: 用于社交网络中两点之间的链路预测。$A(x,y)=\sum_{u\in N(x)\cap N(y)} \frac{1}{\log|N(u)|}$，其中$N(u)$表示u点的邻居。A=0表示这两个点完全不相邻，而这个值越高表明越靠近。**就是看看x、y这两个点相同邻居的个数**。


## 进度3
基于相似度的算法
> 1. common neighbors index：最简单，两个用户有着很多相同邻居，就认为它们很有可能建立联系。$Score(x,y)=|N(x)\cap N(y)|$。通常也是最有效的方法。
> 2. Adamic Adar index:~~对于已经是好友的点就不再计算AA_index，因此推荐出的好友比较少(3个左右)，其他点的 AA_index 基本上是0(完全没关系或者已经是好友)。可能没有共同的邻居，但研究的兴趣是相关的。~~
速度快，实现简单(已实现)。不过仅考虑了边的关系，评价维度较少。
> 3. Jaccard cofficient index:已实现，但论文中提到该算法的效果一般不太好。

进一步对数据分析，发现SCHOLAT LINK Prediction数据集中的顶点度数分布很接近幂律分布。**(马丹，matplotlib画的图要保存为jpg,保存为png图片显示不全，折腾了半天，焯！！)**