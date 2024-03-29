### 2022.9.5
> LTD1DT 大体实现, minGreedy算法逻辑已经实现

### 2022.9.6
> 测试 LTD1DT + minGreedy 发现问题, 在运行 minGreedy 时需要保持 G 的属性不变,但运行每创建一个 LTD1DT对象,
self.G 的属性就会改变, 偏离 minGreedy 的原意, 需要改变逻辑.

### 2022.9.7
> 把 init 之后的 G 保留下来就行。\
> ~~*发现原论文的一个问题：假设 node A 达到了 influence threshold，其邻居都是 R-active，没有 T-active。但没有达到 decision threshold，即 node A 会接收 > T。但这合理吗？其周围没有 T-active 邻居，却能变成 node A?*~~\ **(2022.9.16: 后来想了一下，是没问题的。)**\
> **继续实现按论文所定义的模型** \
修改完成，待测试

### 2022.9.8
> LTD1DT修补完成，暂未发现问题。\
> ~~**没法按照论文的定义进行模拟扩散，太不符合逻辑了。** 稍微调整了一下，node A 周围必须有至少一个 T-active 邻居，才能变为 node A。否则继续保持 influenced 状态。~~\
> 优化了 LTD1DT 的实现逻辑,更方便后续操作.

### 2022.9.10
> 感觉 LTD1DT 得重构了,唉.

### 2022.9.12
> LTD1DT 改进成 model_V2. 自我感觉逻辑畅通多了，核心扩散功能基本正常；MinGreedy算法初步测试成功。<mark>想法：谣言遏制与真相最大化？让更多的inactive节点成为真相节点。<mark>

### 2022.9.17
> Pagerank 已经实现。假如后期效果不好，改用Networkx自带的PageRank。再不行就上pytorch。\
> ContrId也写完了，但还没测试。

### 2022.9.18
> 都测试过了，基本没问题。准备进行比较实验。\
> Proximately 版本的算法还没写。

### 2022.9.20
> 文章提到谣言种子节点分为随机选取和基于度选取。\
> 在模型阈值确定的情况下，随机选取谣言节点，测试100次扩散情况，实际上扩散不了多少，持平或低于文章的实验结果;
若基于度选取谣言节点，由于模型阈值已定，测试多少次都一样,结果也是低于或持平文章实验结果\
> 因此将进行以下测试：
> 1. 模型阈值确定+随机选谣言节点
> 2. 模型阈值随机+（随机、基于度）选谣言节点
>
> 对于情况1,上面已经说了。
>
> 对于情况2：（与原文给出的实验结果对比）
> * 在scale free、small world网络上，随机选取&基于度选取时,接近原文结果，略低于。
> * 在netScience、USpower网络上，随机选取时低于原文结果，有点差距；基于度选取时，基本一致。
> 
> 因此，最接近文章的效果是情况2.

### 2022.9.21
> 谣言信息往往与自身利益相关，容易失去对信息判断能力，就更倾向于相信谣言。

### 2022.9.22
> 随机选取谣言节点, 并不会造成大规模的扩散,因此真相节点也选不出啥.所以贪心和启发式算法的结果差不多.\
> 通过 line profiler 分析,贪心法99.7%的时间都用在 diffusion 上.看看能不能稍微优化一下.

### 2022.9.24
> 优化了 LTD1DT 模型，缩小了搜索范围，速度大约快了 3 倍。
> 对比如下：选取度最大的点作为谣言节点。
> 1. 原版：
> ```python
> [in]  model3= model_V0(G_scale_free,False,[],[1,2,4])
>       %time len(model3.diffusion()[2])
> [out] CPU times: user 20.6 ms, sys: 94 µs, total: 20.7 ms
>       Wall time: 18.6 ms
>       103
> ```
> 2. 新版（重新初始化）:
> ```python
> [in]  model4 = model_V2(G_scale_free,False,[],[1,2,4])
>       %time len(model4.diffusion()[2])
> [out] CPU times: user 5.75 ms, sys: 0 ns, total: 5.75 ms
>       Wall time: 5.54 ms
>       122
> ```
> 3. 新版（使用原版的初始化的图）:
> ```python
> [in]  model5 = model_V2(model3.G,True,[],[1,2,4])
>       %time len(model5.diffusion()[2])
> [out] CPU times: user 6.05 ms, sys: 1.16 ms, total: 7.21 ms
>       Wall time: 6.69 ms
>       103
> ```
> 然而对于MinGreedy，还是慢，只比原来快 50% 左右。算法速度的下限就那样了，加上python也是慢。
>
> 优化前：
> ```python
> [in]  %time algor.MinGreedy(model,[1,2,4],10)
> [out] CPU times: user 47.4 s, sys: 46.7 ms, total: 47.5 s
>       Wall time: 47.5 s
>       [3, 29, 35, 109, 157, 169, 17, 37, 124, 187]
> ```
> 优化后：（使用优化前初始化的图）
> ```python
> [in]  %time algor.MinGreedy(model_sf,[1,2,4],10)
> [out] CPU times: user 26.2 s, sys: 114 ms, total: 26.3 s
>       Wall time: 26.3 s
>       [3, 29, 35, 109, 157, 169, 17, 37, 124, 187]
> ```

### 2022.9.25
> 换了一个python解释器：pypy。原版的解释器应该是cpython。\
> 大部分情况是更快了，尤其循环多的情况。\
> 在控制台上测试：
> 1. 原版解释器：
> ```python
>   from LTD1DT import model_V2
>   from algorithm import MinGreedy
>   import networkx as nx
>   import time
>   G = nx.barabasi_albert_graph(500,1)
>   model = model_V2(G,False,[],[23,1,0])
>   def func():
>       t = time.time()
>       print(MinGreedy(model,[23,1,0],10))
>       print(time.time()-t)
> [in]  func()
> [out] [7, 2, 35, 61, 244, 6, 18, 25, 30, 56]
>       16.490251779556274
> ```
> 2. pypy解释器(重新初始化一个图)：
> ```python
>   def func():
>       t = time.time()
>       print(MinGreedy(model,[23,1,0],10))
>       print(time.time()-t)
> [in]  func()
> [out] [59, 178, 42, 105, 8, 54, 121, 144, 192, 212]
>       4.1216888427734375
> ```
> 3. pypy解释器(使用原始的图)：
> ```python
>   def func():
>       t = time.time()
>       print(MinGreedy(model,[23,1,0],10))
>       print(time.time()-t)
> [in]  func()
> [out] [7, 2, 35, 61, 244, 6, 18, 25, 30, 56]
>       4.612504005432129
> ```
### 2022.9.26
> 当图节点数量增多后, PageRank的速度也逐渐慢下来.\
> 但大头还是贪心法. 在 uspower 中,选一个点快 1 分钟.
> ```python
> [in] %time len(algor.MinGreedy(model_us,[2553,4458,831],1))
> [out] CPU times: user 52 s, sys: 170 ms, total: 52.2 s
>       Wall time: 52.2 s
> ```