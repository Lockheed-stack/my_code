### 2022.9.5
> LTD1DT 大体实现, minGreedy算法逻辑已经实现

### 2022.9.6
> 测试 LTD1DT + minGreedy 发现问题, 在运行 minGreedy 时需要保持 G 的属性不变,但运行每创建一个 LTD1DT对象,
self.G 的属性就会改变, 偏离 minGreedy 的原意, 需要改变逻辑.

### 2022.9.7
> 把 init 之后的 G 保留下来就行。\
> *发现原论文的一个问题：假设 node A 达到了 influence threshold，其邻居都是 R-active，没有 T-active。但没有达到 decision threshold，即 node A 会接收 > T。但这合理吗？其周围没有 T-active 邻居，却能变成 node A?* \
> **继续实现按论文所定义的模型** \
修改完成，待测试

### 2022.9.8
> LTD1DT修补完成，暂未发现问题。\
> **没法按照论文的定义进行模拟扩散，太不符合逻辑了。** 稍微调整了一下，node A 周围必须有至少一个 T-active 邻居，才能变为 node A。否则继续保持 influenced 状态。\
> 优化了 LTD1DT 的实现逻辑,更方便后续操作.