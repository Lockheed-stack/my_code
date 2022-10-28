# 2022年10月17日

总结一下目前情况:
> * rumor 出现的情况：零星分布(个人)、少数有组织(degree 大)。都随机生成。
> * Truth 的出现情况：大部分有组织（权威、degree 大）。目前暂时认为是固定的。
> * 谣言的检测具有滞后性，需要检测到谣言后再进行行动(如散播真相去纠正)。
> * 点具有两个阈值：influenced & correction。值越低，越容易被影响或纠正。权威Truth node 会改变其 k 阶邻居的 correction 阈值，目前有 2 种计算公式：
>> 1. $P_{correction}=P_0-\frac{P_0}{1+e^{\beta x}}$ 。默认$\beta=1$。 $\beta$ 越小，说明权威Truth node 的纠正能力越强。x 表示第 k 阶邻居，下同。
>> 2. $P_{correction} = P_0-P_0(1-e^{- \frac{1-P_0}{x}})$ 。只与 $P_0$ 自身相关。

关于如何初始化 authoritative Truth & organized Rumor:
> * 试了一下使用 p-value 选取degree显著偏大的顶点。小一点的图，例如阈值设为 1e-7, 就还可以选出不错的结果；但图一旦变大，阈值设置到 $1e^{-100+}$ ，都会有很多点，效果不好。 *( 感觉与图的幂律分布有关 )*
> * 从高于 median/mean degree 的点中选取一部分。目前使用高于 **median degree** 的区分标准，效果还行。

# 2022年10月28日
> * 检测谣言具有滞后性，即谣言在传播一段时间后才会被检测到。处于一种"敌暗我明"的状态。

因此，除了相对固定的 authoritative Truth node，还需要预先设置一些 Monitoring T node，及早的检测到谣言，在检测到谣言后，才可以采取一些其他措施，比如封禁一些点，在某些地方设置 T node 等。

基于以上，问题初步定义为：
> 给定一个图G，n0 个 authoritative Truth nodes，n1 个（随机）的rumor nodes，在总预算 C 内，使得谣言的影响范围最小。*(如果能同时让真相的影响更大，那更好)*