# 第四章

## 神经网络八股功能扩展

### 1.自制数据集，解决本领域应用 

```python
def generateds(path, txt):
    f = open(txt, 'r')  # 以只读形式打开txt文件
    contents = f.readlines()  # 读取文件中所有行
    f.close()  # 关闭txt文件
    x, y_ = [], []  # 建立空列表
    for content in contents:  # 逐行取出
        value = content.split()  # 以空格分开，图片路径为value[0] , 标签为value[1] , 存入列表
        img_path = path + value[0]  # 拼出图片路径和文件名
        img = Image.open(img_path)  # 读入图片
        img = np.array(img.convert('L'))  # 图片变为8位宽灰度值的np.array格式
        img = img / 255.  # 数据归一化 （实现预处理）
        x.append(img)  # 归一化后的数据，贴到列表x
        y_.append(value[1])  # 标签贴到列表y_
        print('loading : ' + content)  # 打印状态提示

    x = np.array(x)  # 变为np.array格式
    y_ = np.array(y_)  # 变为np.array格式
    y_ = y_.astype(np.int64)  # 变为64位整型
    return x, y_  # 返回输入特征x，返回标签y_
```



![image-20200526090515552](..\images\image-20200526090515552.png)

### 2.数据增强，扩充数据集

![image-20200524214007052](..\images\image-20200524214007052.png)

```python
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)  # 给数据增加一个维度,从(60000, 28, 28)reshape为(60000, 28, 28, 1)

image_gen_train = ImageDataGenerator(
    rescale=1. / 1.,  # 如为图像，分母为255时，可归至0～1
    rotation_range=45,  # 随机45度旋转
    width_shift_range=.15,  # 宽度偏移
    height_shift_range=.15,  # 高度偏移
    horizontal_flip=False,  # 水平翻转
    zoom_range=0.5  # 将图像随机缩放阈量50％
)
image_gen_train.fit(x_train)

model = tf.keras.models.Sequential

model.compile

model.fit(image_gen_train.flow(x_train, y_train, batch_size=32), epochs=5, validation_data=(x_test, y_test),
          validation_freq=1)
model.summary()
```



- 这里ImageDataGenerator.fit需要四维数据
- model.fit()需要ImageDataGenerator()作为输入
- 数据增强可以在小数据量上可以增加模型泛化性

### 3.断点续训，存取模型

- **load_weights(路径文件名)** 读取model

    ```python
    cp_callback=tf.keras.callbacks.ModelCheckpoint( 
    
    	filepath=路径文件名, 
        
        monitor='val_loss',
    
    	save_weights_only=True/False, #是否只保留模型参数
    
    	save_best_only=True/False	#是否最优模型
    )
    ```
    
    ```python
    history = model.fit(x_train, y_train, batch_size=32, epochs=5,
    validation_data=(x_test, y_test), validation_freq=1,
    callbacks=[cp_callback])
    ```
    
- monitor 配合 save_best_only 可以保存最优模型，包括：训练损失最小模型、测试损失最小模型、训练准确率最高模型、测试准确率最高模型等。

### 4.参数提取，把参数存入文本 

- **model.trainable_variables** 模型中可训练的参数
- 设置函数**np.set_printoptions**(precision=小数点后按四舍五入保留几位,threshold=数组元素数量少于或等于门槛值，打印全部元素；否则打印门槛值+1个元素，中间用省略号补充）
    - precision=np.inf 打印完整小数位；threshold=np.nan 打印全部数组元素

### 5.acc/loss可视化，查看训练效

- history： 

    训练集loss： loss 

    测试集loss： val_loss 

    训练集准确率： sparse_categorical_accuracy 

    测试集准确率： val_sparse_categorical_accuracy

- 用history.history["参数"]提取

### 6.应用程序，给图识物

- **predict(输入特征, batch_size=整数)** 返回前向传播计算结果

    ```python
    model = tf.keras.models.Sequential([])#复现模型（前向传播）
    model.load_weights(model_save_path)#加载参数
    result = model.predict(x_predict)#预测结果
    ```

- 预处理,使输入满足NN model对输入风格的要求

    ```python
     for i in range(28):         #提高对比度，变为只有黑白
            for j in range(28):
                if img_arr[i][j] < 200:		#200为阈值，调整可修改效果
                    img_arr[i][j] = 255
                else:
                    img_arr[i][j] = 0
    ```
	或者
    ```python
    img_arr = 255 - img_arr#白底黑字变为黑底白字
    ```

    

