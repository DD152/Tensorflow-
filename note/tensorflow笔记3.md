# 第三章

## 神经网络搭建八股

### 1. tf.keras搭建网络八股

- 六步法
    - **import**
    
    - **train,test**
    
    - **model=tf.keras.models.Sequential([网络结构])** 上层输入就是下层输出的顺序网络结构，走一遍前向传播
    
        or
    
        **class MyModel(model)=MyModel**
    
    - **model.compile()**  配置训练方法，选择哪个优化器、哪个损失函数，哪种评测指标
    
    - **model.fit()** 执行训练过程，告知训练集和测试集的输入特征和标签，每个batch时多少，迭代次数
    
    - **model.summary()** 打印网络结构和参数统计

####  model=tf.keras.model.Sequential([网络结构]) 描述各层网络

- 拉直层：tf.keras.layers.Flatten() 拉直层可以变换张量的尺寸，把输入特征拉直为一维数组，是不含计算参数的层。

- 全连接层：tf.keras.layers.Dense( 神经元个数, activation=”激活函数”, kernel_regularizer=”正则方式”) 
    - activation（字符串给出）可选relu、softmax、sigmoid、tanh等
    - kernel_regularizer可选：tf.keras.regularizers.l1()、tf.keras.regularizers.l2() 

- 卷积层：tf.keras.layers.Conv2D( filter = 卷积核个数, kernel_size = 卷积核尺寸, strides = 卷积步长, padding = “valid” or “same”) 

- LSTM层：tf.keras.layers.LSTM()

#### class MyModel(model)=MyModel

- ![image-20200522224859719](C:\Users\DDan\AppData\Roaming\Typora\typora-user-images\image-20200522224859719.png)

#### model.compile(optimizer = 优化器,loss = 损失函数,metrics = [“准确率”])

- Optimizer可选:

    ‘sgd’ or tf.keras.optimizers.SGD (lr=学习率,momentum=动量参数)

    ‘adagrad’ or tf.keras.optimizers.Adagrad (lr=学习率)

    ‘adadelta’ or tf.keras.optimizers.Adadelta (lr=学习率) 

    ‘adam’ or tf.keras.optimizers.Adam (lr=学习率, beta_1=0.9, beta_2=0.999) 

- loss可选: 

    ‘mse’ or tf.keras.losses.MeanSquaredError() 

    ‘sparse_categorical_crossentropy’ or tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False) **若经过概率分布(softmax)为False，直接输出为True**

- Metrics可选: 

    ‘accuracy’ ：y_和y都是数值，如y_=[1] y=[1] 

    ‘categorical_accuracy’ ：y\_和y都是独热码(概率分布)，如  y_=[0,1,0] y=[0.256,0.695,0.048] 

    ‘sparse_categorical_accuracy’ ：y\_是数值，y是独热码(概率分布),如y_=[1] y=[0.256,0.695,0.048]

#### model.fit

**model.fit (**

​						**训练集的输入特征, **

​						**训练集的标签, **

​						**batch_size= , **

​						**epochs= , **

​						**validation_data=(测试集的输入特征，测试集的标签), **

​						**validation_split=从训练集划分多少比例给测试集， **

​						**validation_freq = 多少次epoch测试一次**

**)**

- validation_data, validation_split二选一
- 问题：咱们体现validation_freq

#### model.summary()

![image-20200521221110603](C:\Users\DDan\AppData\Roaming\Typora\typora-user-images\image-20200521221110603.png)

