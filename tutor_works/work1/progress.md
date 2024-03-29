# Spanning subgraph已实现
> 问题：8个点的完全图有28条边，$C_{28}^{14}=40116600$,
> 9个点的完全图36条边，$C_{36}^{18}=9075135300$
> 8个点求spanning subgraph已经很困难了，再往后基本无法得出结果，*内存不够 or 耗时很久*

# 计算 steiner distance
## 已实现，原理和逻辑还要继续研究
> 搜索了一下，发现是最小斯坦纳树问题: 一个图中，有若干个关键点，将这几个关键点连在一起的最小花费。直观的理解，就是带关键节点的最小生成树。

> 用状态压缩的动态规划来解决。
> 斯坦纳树的状态应该定为 $dp[i][state]$，表示以 i 为根，关键点在当前斯坦纳树中的连通状态为 state。
> 状态转移分成两部分：
> 1. 枚举连通状态的子集：
> $$dp[i][state]=min(dp[i][subset_1]+dp[i][subset_2])$$
> 2. 枚举树上边进行松弛：
> $$dp[i][state]=min(dp[i][state]+dp[j][state]+e[i][j])$$