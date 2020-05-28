



# 第二章

## 神经网络优化

### 1.预备知识 

- **tf.where(条件语句，真返回A，假返回B)**

    **tf.greater(a, b)** 对a，b中元素依次比较，若a>b,返回真，否则返回假

    ```python
    import tensorflow as tf
    
    a = tf.constant([1, 2, 3, 1, 1])
    b = tf.constant([0, 1, 3, 4, 5])
    c = tf.where(tf.greater(a, b), a, b)  # 若a>b，返回a对应位置的元素，否则返回b对应位置的元素
    print("c：", c)
    ```

    ```java
    c： tf.Tensor([1 2 3 4 5], shape=(5,), dtype=int32)
    ```

- **np.random.RandomState.rand(维度)** 返回一个[0,1)之间的随机数

    ```python
    import numpy as np
    rdm=np.random.RandomState(seed=1) #seed=常数每次生成随机数相同
    a=rdm.rand() # 返回一个随机标量
    b=rdm.rand(2,3) # 返回维度为2行3列随机数矩阵
    print("a:",a)
    print("b:",b)
    ```

    ```java
    a: 0.417022004702574
    b: [[7.20324493e-01 1.14374817e-04 3.02332573e-01]
    [1.46755891e-01 9.23385948e-02 1.86260211e-01]]
    ```

- **np.vstack (数组 1，数组2)** 将两个数组按垂直方向叠加

    ```python
    import numpy as np
    a = np.array([1,2,3])
    b = np.array([4,5,6])
    c = np.vstack((a,b))
    print("c:\n",c)
    ```

    ```java
    c:
    [[1 2 3]
    [4 5 6]]
    ```

- 生成网格坐标点

    - **np.mgrid[ 起始值 : 结束值 : 步长 ，起始值 : 结束值 : 步长 , … ]**  [起始值 结束值)
    - **x.ravel( )** 将x变为一维数组，“把 . 前变量拉直”
    - **np.c_[ 数组1，数组2， … ]** 配对

    ```python
    import numpy as np
    
    # 生成等间隔数值点
    x, y = np.mgrid[1:3:1, 2:4:0.5]
    # 将x, y拉直，并合并配对为二维张量，生成二维坐标点
    grid = np.c_[x.ravel(), y.ravel()]
    print("x:\n", x)
    print("y:\n", y)
    print("x.ravel():\n", x.ravel())
    print("y.ravel():\n", y.ravel())
    print('grid:\n', grid)
    ```

    ```java
    x:
     [[1. 1. 1. 1.]
     [2. 2. 2. 2.]]
    y:
     [[2.  2.5 3.  3.5]
     [2.  2.5 3.  3.5]]
    x.ravel():
     [1. 1. 1. 1. 2. 2. 2. 2.]
    y.ravel():
     [2.  2.5 3.  3.5 2.  2.5 3.  3.5]
    grid:
     [[1.  2. ]
     [1.  2.5]
     [1.  3. ]
     [1.  3.5]
     [2.  2. ]
     [2.  2.5]
     [2.  3. ]
     [2.  3.5]]
    ```

    

### 2.神经网络复杂度

- 空间复杂度

    用层数 `不带输入层` 和 NN中待优化参数的个数 `总w和b`来表示

- 时间复杂度

    乘加运算的次数 `一组乘加算一次`

### 3.指数衰减学习率 

- 可以先用较大的学习率，快速得到较优解，然后逐步减小学习率，使

    模型在训练后期稳定。

- **tf.keras.optimizers.schedules.ExponentialDecay**

    ```python
    tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps, decay_rate, staircase=False, name=None
    )
    #等价于 tf.optimizers.schedules.ExponentialDecay
    '''
    initial_learning_rate: 初始学习率.
    decay_steps: 衰减步数, staircase为True时有效.
    decay_rate: 衰减率.staircase: Bool型变量.
    如果为True, 学习率呈现阶梯型下降趋势.
    返回计算得到的学习率
    '''
    ```

    

![image-20200521091543440](..\images\image-20200521091543440.png)

![image-20200521090919513](..\images\image-20200521090919513.png)

### 4.激活函数 

- 添加激活函数提升了模型表达力，使多层NN是x的线性组合，NN可随层数的增加提升表达力

- 优秀的激活函数：

    - 非线性：激活函数非线性时，多层神经网络可逼近所有函数。激活函数才不会被单层网络替代，使错层网络又意义。
    - 可微性：优化器大多用梯度下降更新参数
    - 单调性：当激活函数是单调的，能保证单层网络的损失函数是凸函数`方便找最值`
    - 近似恒等性：f(x)≈x当参数初始化为随机小值时，神经网络更稳定

- 激活函数输出值的范围：

    - 激活函数输出为有限值时，基于梯度的优化方法更稳定

    - 激活函数输出为无限值时，参数的初始值对模型的影响大，建议调小学习率

    ![image-20200521091735740](..\images\image-20200521091735740.png)

- **tf.nn.sigmoid(x)** 归一化，现在用的少 `等价于tf.sigmoid`

    ```python
    tf.nn.softmax(
        logits, axis=None, name=None
    )
    ```

    

    - 链式求导需要多层导数连续相乘 `多个0-0.25的值相乘`，结果趋于零，梯度消失 

    ![image-20200521095702127](..\images\image-20200521095702127.png)

- **tf.math.tanh(x)**  `等价于tf.tanh`

    ![image-20200521101733006](..\images\image-20200521101733006.png)

- **tf.nn.relu(x)**

    - Dead ReIU问题：送入激活函数的输入特征是负数时，激活函数的输出为零，反向传播得到的梯度为零，参数无法更新。`改进初始化，避免过多负数特征输入；设置更小lr，减少参数分布的变化，避免训练时产生负数特征进入relu函数`

    ![image-20200521100359106](..\images\image-20200521100359106.png)

- **tf.nn.leaky_relu(x)**

    ![image-20200521101225521](..\images\image-20200521101225521.png)

- 对于**初学者**

    ![image-20200521101351880](..\images\image-20200521101351880.png)

### 5.损失函数

- 损失函数（loss）：预测值（y）与已知答案（y_）的差距

    ![image-20200521101846606](..\images\image-20200521101846606.png)

    - 均方误差 **loss_mse = tf.reduce_mean(tf.square(y_ \- y))**

        ```python
    tf.keras.losses.MSE(y_true, y_pred) #等价
        ```

        

        ![image-20200521101924517](..\images\image-20200521101924517.png)

    - 自定义损失函数

        ![image-20200521110636567](..\images\image-20200521110636567.png)

    - **tf.losses.categorical_crossentropy(y_,y)**

        交叉熵损失函数CE (Cross Entropy)：表征两个概率分布之间的距离

        ![image-20200521111229785](..\images\image-20200521111229785.png)
    
        ```python
tf.keras.losses.categorical_crossentropy(
        y_true, y_pred, from_logits=False, label_smoothing=0
        )
        #等价
        ```

    
    ​    
    

    - **tf.nn.softmax_cross_entropy_with_logits(y_,y)** softmax与交叉熵结合
      
        输出先过softmax函数，再计算y与y_的交叉熵损失函数。
        
        ```python
        loss_ce2 = tf.nn.softmax_cross_entropy_with_logits(y_, y)
        ```
        
        等价于
        
        ```python
        y_pro= tf.nn.softmax(y)
        loss_ce1 = tf.losses.categorical_crossentropy(y_, y_pro)    
        ```

### 6.欠拟合与过拟合

![image-20200521112132435](..\images\image-20200521112132435.png)

- 欠拟合 不能有效表征数据点
    - 解决办法
        - 增加输入特征项 
        - 增加网络参数
        - 减少正则化参数
- 过拟合 泛化性弱
    - 解决办法
        - 数据清洗 减少噪声
        - 增大训练集
        - 采用正则化
        - 增大正则化参数

### 7.正则化减少过拟合 

- 正则化在损失函数中引入模型复杂度指标，利用给W加权值，弱化了训练数据的噪声（一般不正则化b）

- 正则化后loss变为

    ![image-20200521113249853](..\images\image-20200521113249853.png)

- 正则化选择
    1. **tf.nn.l1_loss(w)**L1正则化大概率会使很多参数变为零，因此该方法可通过稀疏参数，即减少参数的数量，降低复杂度。
    2. **tf.nn.l2_loss(w)**L2正则化会使参数很接近零但不为零，因此该方法可通过减小参数值的大小降低复杂度，缓解因噪声产生的过拟合。

### 8.优化器更新网络参数

`在计算梯度后，用于优化梯度更新`

![image-20200521173247909](..\images\image-20200521173247909.png)

**不同的优化器实质上只是定义了不同的一阶动量和二阶动量公式**

#### SDG随机梯度下降

- 无momentum，常用

![image-20200521173530124](..\images\image-20200521173530124.png)

![image-20200521180250602](..\images\image-20200521180250602.png)

#### SGDM

- 含momentum的SGD

- 在SGD基础上增加一阶动量

- m_t 代表各时刻梯度方向的指数滑动平均值

- β是个超参数，接近1，经验值0.9

![image-20200521174307307](..\images\image-20200521174307307.png)

![image-20200521180325795](..\images\image-20200521180325795.png)

#### Adagrad

- 在SGD基础上增加二阶动量，可对模型中的每个参数分配自适应学习率

![image-20200521175258455](..\images\image-20200521175258455.png)

![image-20200521180347898](..\images\image-20200521180347898.png)

#### RMSProp

- SGD基础上增加二阶动量

- v使用指数滑动平均值计算，表征过去一段时间的平均值

![image-20200521181534715](..\images\image-20200521181534715.png)

![image-20200521181518460](..\images\image-20200521181518460.png)

#### Adam 

- 同时结合SGDM一阶动量和RMSProp二阶动量并加入两个修正项目

    ![image-20200521181952106](..\images\image-20200521181952106.png)

    ![image-20200521182453860](..\images\image-20200521182453860.png)


```

```