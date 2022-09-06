### 2022.9.5
LTD1DT 大体实现, minGreedy算法逻辑已经实现

### 2022.9.6
测试 LTD1DT + minGreedy 发现问题, 在运行 minGreedy 时需要保持 G 的属性不变,但运行每创建一个 LTD1DT对象,
self.G 的属性就会改变, 偏离 minGreedy 的原意, 需要改变逻辑.