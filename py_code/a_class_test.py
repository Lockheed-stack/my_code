# %%
''' example: 一个存在问题的参数'''
class Player:
    def __init__(self,name,items=[]) -> None:
        self.name = name
        self.items = items
        print('the id of items of '+self.name+': ',id(self.items))

p1 = Player("Alice")
p2 = Player('Bob')
p3 = Player("Charles",["sword"])

p1.items.append("armor")
p2.items.append("sword")
print(p1.items)
'''会发现 p1 既有 armor,也有sword,p1和p2的id是一样的
python官方文档里的解释:
Default parameter values are evaluated from left to right when then function definition is executed.
This means that the expression is evaluated once, when the function is defined, and that the same "pre-computed"
values is used for each call. This is especially important to understand when a default parameter value is a 
mutable object, such as a list or directory: if the function modifies the object, the default parameter value is 
in effect modified. 
This is generally not what was intended. A way around this is to use 'None' as the default, and explicitly test for
it in the body of the function.
'''
# %%
''' 也就是说参数中由'mutable'类型的变量，要设置为'None'。修改如下: '''
class Player:
    def __init__(self,name,items=None) -> None:
        self.name = name
        self.items = []
        print('the id of items of '+self.name+': ',id(self.items))

p1 = Player("Alice")
p2 = Player('Bob')
p3 = Player("Charles",["sword"])

p1.items.append("armor")
p2.items.append("sword")
print(p1.items)
# %%
'''迭代器iterator,是一个数据流对象, 可以用next从这个对象中获取数据'''
'''iterable, 可迭代对象,是一个对象'''
'''具体内容官方文档也写得很清楚'''
'''实现一个链表操作'''
class NodeIter:
    def __init__(self,node) -> None:
        self.curr_node = node
    
    def __next__(self):
        if self.curr_node is None:
            raise StopIteration #到头了,要给出这个异常
        node, self.curr_node = self.curr_node, self.curr_node.next
        return node
    
class Node:
    def __init__(self,name) -> None:
        self.name = name
        self.next = None
    def __iter__(self):
        return NodeIter(self)
    
node1 = Node("node1")
node2 = Node("node2")
node3 = Node('node3')

node1.next = node2
node2.next = node3
for node in node1:
    print(node.name)
'''这里NodeIter是一个迭代器,但没有实现__iter__, 因此它是不可迭代的。
如果想要它也能迭代, 只需要加两行代码即可:
def __iter__(self):
    return self
'''
# %%
'''generator生成器,是特殊的迭代器'''
'''调用生成器函数,返回一个生成器对象'''
''''''
def gen(num):#生成器函数
    while num>0:
        tmp = yield num#有yield，编译器不会把这个函数当成普通函数
        if tmp is not None:
            num = tmp
        num-=1
    return 
    #等价于raise StopIteration。不管有没有return值，都不会在调用next的时候返回，
    #只用yield的值会保存起来。想拿到return value，需要catch StopIteration这个exception
    
g = gen(5) # g是生成器对象，gen这个函数返回的不是具体的值，返回一个生成器对象
first = next(g)#使用next，此时才会运行函数本体
print("fitst:",first)
print("sned:",g.send(10))#sned就是在生成器yield之后，把yield的东西变成一个值，可继续赋给生成器中的其他变量
for i in g:#相当于每一次都call 一次 next.相当于g.send(None)
    print(i)
# %%
'''decorator装饰器'''
'''函数可以作为参数，也可以作为返回值'''
'''decorator本身是一个函数, @后面跟着是函数名。本质上是输入和输出都是函数的函数'''
'''即参数是函数, 返回值也是函数'''
def double(x):
    print(2*x) 
def triple(x):
    print(3*x)
def callback(func,x):
    return func(x)

def  get_multiple_fun(n):
    def mutiple(x):
        return n*x
    return mutiple

callback(double,4)#作为参数
a = get_multiple_fun(2)#作为返回值
print(a(5))

def dec(f):
    return 1

@dec
def penta(x):
    return x*5
#等价于penta = dec(penta)
print(penta)

#进阶例子
import time
def timeit(f):
    def wrapper(*args,**kwargs):#允许变长的函数参数
        start = time.time()
        ret = f(*args,**kwargs)
        print(time.time()-start)
        return ret
    return wrapper
@timeit
def my_func(x):
    time.sleep(x)
    str = "it's me, my_func"
    return str
    
print(my_func(1))

# %%
