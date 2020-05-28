# 第一章

## 神经网络计算过程及模型搭建

### 1.Tensor 表示张量，是多维数组、多维列表，用阶表示张量的维数。[有几个表示几阶

- 0阶表示标量 scalar
- 1阶表示向量 vector
- 2阶表示矩阵 matrix
- 3阶表示张量 tensor

### 2.数据类型

- tf.int 32  //  tf.float 32  //  tf.float 64  //  tf.bool `tf.constant([True,False])` //  tf.string `tf.constant("hello")`
  
### 3.创建Tensor

- **tf.constant(张量内容,dtype=数据类型(可选))**

    ``` python
    import tensorflow as tf 
    a=tf.constant([1,5],dtype=tf.int64)
    print(a)
    print(a.dtype)
    print(a.shape)
    ```

    > tf.Tensor([1 5], shape=(2,), dtype=int64)
    > <dtype: 'int64'>
    > (2,)

- **tf.convert_to_tensor(数据名,dtypr=数据类型(可选))**    将numpy转为Tensor

    ``` python
    import tensorflow as tf
    import numpy as np
    a = np.arange(0, 5)
    b = tf.convert_to_tensor( a, dtype=tf.int64 )
    print(a)
    print(b)
    ```

    > [0 1 2 3 4]
    > tf.Tensor([0 1 2 3 4], shape=(5,), dtype=int64)

- 一维 直接写个数

    二维 用[行,列]

    多维 用[n,m,j,k...]

    - **tf.zero(维度)** 创建全为0的张量
    - **tf.ones(维度)** 创建全为1的张量
    - **tf.fill(维度,指定值)** 创建指定值的张量

    ``` python
    a = tf.zeros([2, 3])
    b = tf.ones(4)
    c = tf.fill([2, 2], 9)
    print(a)
    print(b)
    print(c)
    ```

    > tf.Tensor([[0. 0. 0.] [0. 0. 0.]], shape=(2, 3), dtype=float32)
    >
    > tf.Tensor([1. 1. 1. 1.], shape=(4, ), dtype=float32)
    >
    > tf.Tensor([[9 9] [9 9]], shape=(2, 2), dtype=int32)

- **tf.random.normal (维度,mean=均值,stddev=标准差)** 生成正态分布的随机数，默认均值为0，标准差为1。

- **tf.random.truncated_normal (维度,mean=均值,stddev=标准差)** 生成截断式正态分布的随机数，能使生成的这些随机数更集中一些，如果随机生成数据的取值 在 (µ - 2σ，u + 2σ ) 之外则重新进行生成，保证了生成值在均值附近。

- **tf.random.uniform(维度,minval=最小值,maxval=最大值)** 生成指定维度的均匀分布随机数， minval 随机数的最小值，maxval 随机数的最大值。 最小、最大值是前闭后开区间。

    ```python
    d = tf.random.normal ([2, 2], mean=0.5, stddev=1)
    print(d)
    e = tf.random.truncated_normal ([2, 2], mean=0.5, stddev=1)
    print(e)
    f = tf.random.uniform([2, 2], minval=0, maxval=1)
    print(f)
    ```

    > tf.Tensor([[0.7925745 0.643315 ] [1.4752257 0.2533372]], shape=(2, 2), dtype=float32) tf.Tensor([[ 1.3688478 1.0125661 ] [ 0.17475659 -0.02224463]], shape=(2, 2), dtype=float32) tf.Tensor([[0.28219545 0.15581512] [0.77972126 0.47817433]], shape=(2, 2), dtype=float32)

### 4. 常用函数

- **tf.cast(张量名,dtype=数据类型)** 强制类型转换

- **tf.reduce_min(张量名)** 最小值

- **tf.reduce_max(张量名)** 最大值

- **axis** 指定操作的方向,axis=0表示对第一个维度进行操作，axis=1表示对第二个维度操作

    `对于二维张量,axis=0纵向操作，沿经度方向；axis=0，横向操作，沿纬度方向`

    ![image-20200520164616311](..\images\image-20200520164616311.png)

    -  **tf.reduce_mean (张量名,axis=操作轴)** 计算张量沿着指定维度的平均值 
    -  **tf.reduce_sum (张量名,axis=操作轴) **计算张量沿着指定维度的和
        - 不指定axis时对所有元素操作

    ``` python
    import tensorflow as tf
    
    x = tf.constant([[1, 2, 3], [2, 2, 3]])
    print("x:", x)
    print("mean of x:", tf.reduce_mean(x))  # 求x中所有数的均值
    print("sum of x:", tf.reduce_sum(x, axis=1))  # 求每一行的和
    print("sum of x,axis=0:",tf.reduce_sum(x,axis=0)) #求每一列的和
    ```

    > x: tf.Tensor(
    > [[1 2 3]
    >  [2 2 3]], shape=(2, 3), dtype=int32)
    > mean of x: tf.Tensor(2, shape=(), dtype=int32)
    > sum of x: tf.Tensor([6 7], shape=(2,), dtype=int32)
    > sum of x,axis=0: tf.Tensor([3 4 6], shape=(3,), dtype=int32)

- **tf.Variable(初始值)** 将变量标记为“可训练”,被标记的变量会在反向传播中记录梯度信息。

- **对应元素的四则运算** `只有维度相同的张量才可以做四则运算`：

    tf.add(张量1，张量2)

    tf.subtract(张量1，张量2)

    tf.multiply(张量1，张量2)

    tf.divide(张量1，张量2) 

- **平方、次方与开方**： tf.square(张量1，张量2)，tf.pow(张量1，张量2)，tf.sqrt(张量) 

- **矩阵乘**：tf.matmul(张量1，张量2)

-  **tf.data.Dataset.from_tensor_slices((输入特征, 标签))** 切分传入张量的第一维度，生成输入特征/标签对，构建数据集 `Tensor和numpy格式都行`

    ```python
    import tensorflow as tf
    
    features = tf.constant([12, 23, 10, 17])
    labels = tf.constant([0, 1, 1, 0])
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    for element in dataset:
        print(element)
    ```

    > (<tf.Tensor: shape=(), dtype=int32, numpy=12>, <tf.Tensor: shape=(), dtype=int32, numpy=0>)
    > (<tf.Tensor: shape=(), dtype=int32, numpy=23>, <tf.Tensor: shape=(), dtype=int32, numpy=1>)
    > (<tf.Tensor: shape=(), dtype=int32, numpy=10>, <tf.Tensor: shape=(), dtype=int32, numpy=1>)
    > (<tf.Tensor: shape=(), dtype=int32, numpy=17>, <tf.Tensor: shape=(), dtype=int32, numpy=0>)

- **tf.GradientTape( )** 可搭配with结构计算损失函数在某一张量处的梯度

    ```python
    with tf.GradientTape( ) as tape:
    	若干个计算过程
    grad=tape.gradient(函数，对谁求导)
    ```

    ```python
    import tensorflow as tf
    
    with tf.GradientTape() as tape:
        x = tf.Variable(tf.constant(3.0))
        y = tf.pow(x, 2)
    grad = tape.gradient(y, x)
    print(grad)
    ```

    > tf.Tensor(6.0, shape=(), dtype=float32)

- **enumerate(列表名)** 是python的内建函数

    ```python
    seq = ['one', 'two', 'three']
    for i, element in enumerate(seq):
    print(i, element)
    ```

    > 0 one 
    >
    > 1 two 
    >
    > 2 three

- 独热编码（one-hot encoding）：在分类问题中，常用独热码做标签， 标记类别：1表示是，0表示非。

    **tf.one_hot(待转换数据, depth=几分类)**

    ```python
    import tensorflow as tf
    
    classes = 3
    labels = tf.constant([1, 0, 2])  # 输入的元素值最小为0，最大为2
    output = tf.one_hot(labels, depth=classes)
    print("result of labels1:", output)
    print("\n")
    ```

    > result of labels1: tf.Tensor(
    > [[0. 1. 0.]
    >  [1. 0. 0.]
    >  [0. 0. 1.]], shape=(3, 3), dtype=float32)

- **tf.nn.softmax(x)** 使输出符合概率分布,使结果符合

    ```python
    tf.nn.softmax(
        logits, axis=None, name=None
    )
    #等价于tf.math.softmax
    ```

    ![image-20200520172752770](..\images\image-20200520172752770.png)

    ```python
    import tensorflow as tf
    
    y = tf.constant([1.01, 2.01, -0.66])
    y_pro = tf.nn.softmax(y)
    
    print("After softmax, y_pro is:", y_pro)  # y_pro 符合概率分布
    
    print("The sum of y_pro:", tf.reduce_sum(y_pro))  # 通过softmax后，所有概率加起来和为1
    ```

    ```java
    After softmax, y_pro is: tf.Tensor([0.25598174 0.69583046 0.04818781], shape=(3,), dtype=float32)
    The sum of y_pro: tf.Tensor(1.0, shape=(), dtype=float32)
    ```

    ![image-20200520172638689](..\images\image-20200520172638689.png)

- **w.assign_sub (w要自减的内容)**   自检操作，调用assign_sub前，先用 tf.Variable 定义变量 w 为可训练（可自更新）。

- **tf.argmax (张量名,axis=操作轴)**  返回张量沿指定维度最大值的**索引**

    ```python
    import numpy as np
    import tensorflow as tf
    
    test = np.array([[1, 2, 3], [2, 3, 4], [5, 4, 3], [8, 7, 2]])
    print("test:\n", test)
    print("每一列的最大值的索引：", tf.argmax(test, axis=0))  # 返回每一列最大值的索引
    print("每一行的最大值的索引", tf.argmax(test, axis=1))  # 返回每一行最大值的索引
    
    ```

    ```java
    test:
     [[1 2 3]
     [2 3 4]
     [5 4 3]
     [8 7 2]]
    每一列的最大值的索引： tf.Tensor([3 3 1], shape=(3,), dtype=int64)
    每一行的最大值的索引 tf.Tensor([2 2 0 0], shape=(4,), dtype=int64)
    ```

    

### 5.鸢尾花数据集（Iris）

- 从sklearn包 datasets 读入数据集，语法为：

    ``` python
    from sklearn.datasets import load_iris
    x_data = datasets.load_iris().data 返回iris数据集所有输入特征
    y_data = datasets.load_iris().target 返回iris数据集所有标签
    ```

    ```python
    from sklearn import datasets
    from pandas import DataFrame
    import pandas as pd
    
    x_data = datasets.load_iris().data  # .data返回iris数据集所有输入特征
    y_data = datasets.load_iris().target  # .target返回iris数据集所有标签
    print("x_data from datasets: \n", x_data)
    print("y_data from datasets: \n", y_data)
    
    x_data = DataFrame(x_data, columns=['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']) # 为每一列添加标签，为表格增加行索引（左侧）和列标签（上方）
    pd.set_option('display.unicode.east_asian_width', True)  # 设置列名对齐
    print("x_data add index: \n", x_data)
    
    x_data['类别'] = y_data  # 新加一列，列标签为‘类别’，数据为y_data
    print("x_data add a column: \n", x_data)
    
    #类型维度不确定时，建议用print函数打印出来确认效果
    ```

    ``` java
    x_data from datasets: 
     [[5.1 3.5 1.4 0.2]
     ...
     [5.9 3.  5.1 1.8]]
    y_data from datasets: 
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
     2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
     2 2]
    x_data add index: 
          花萼长度  花萼宽度  花瓣长度  花瓣宽度
    0         5.1       3.5       1.4       0.2
    1         4.9       3.0       1.4       0.2
    ..        ...       ...       ...       ...
    148       6.2       3.4       5.4       2.3
    149       5.9       3.0       5.1       1.8
    
    [150 rows x 4 columns]
    x_data add a column: 
          花萼长度  花萼宽度  花瓣长度  花瓣宽度  类别
    0         5.1       3.5       1.4       0.2     0
    1         4.9       3.0       1.4       0.2     0
    ..        ...       ...       ...       ...   ...
    148       6.2       3.4       5.4       2.3     2
    149       5.9       3.0       5.1       1.8     2
    
    [150 rows x 5 columns]
    
    ```

### 6.神经网络实现鸢尾花分类

1. 准备数据
    - 数据集读入 
    - 数据集乱序 
    - 生成训练集和测试集（即 x_train / y_train）
    -  配成（输入特征，标签）对，每次读入一小撮（batch）
2. 搭建网络
    - 定义神经网路中所有可训练参数 
3. 参数优化
    - 嵌套循环迭代，with结构更新参数，显示当前loss
4.  测试效果 
    -  计算当前参数前向传播后的准确率，显示当前acc 
5. acc / loss可视化