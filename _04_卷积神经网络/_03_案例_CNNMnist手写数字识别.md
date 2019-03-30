# 4.3. 案例：CNNMnist手写数字识别

*   目标
    *   应用tf.nn.conv2d实现卷积计算
    *   应用tf.nn.relu实现激活函数计算
    *   应用tf.nn.max_pool实现池化层的计算
    *   应用卷积神经网路实现图像分类识别
*   应用
    *   CNN-Mnist手写数字识别
*   内容预览
    *   4.3.1 网络设计
        *   1 网络结构
        *   2 具体参数
    *   4.3.2 案例：CNN识别Mnist手写数字
        *   1 手写字识别分析
        *   2 代码
        *   3 学习率过大问题
    *   4.3.3 拓展-Tensorflow高级API实现结构

4.3.1 网络设计
----------

自己设计一个简单的卷积神经网络做图像识别。

由于神经网络的黑盒子特性，如果想自己设计复杂网络通常还是比较困难的，可以使用一些现有的网络结构如之前的GoogleNet、VGG等等

### 1 网络结构

![卷积网络设计](../images/卷积网络设计.png)

### 2 具体参数

*   第一层
    *   卷积：32个filter、大小5*5、strides=1、padding="SAME"
    *   激活：Relu
    *   池化：大小2x2、strides2
*   第一层
    *   卷积：64个filter、大小5*5、strides=1、padding="SAME"
    *   激活：Relu
    *   池化：大小2x2、strides2
*   全连接层

**经过每一层图片数据大小的变化需要确定，Mnist输入的每批次若干图片数据大小为\[None, 784\]，如果要经过卷积计算，需要变成\[None, 28, 28, 1\]**

*   第一层
    *   卷积：\[None, 28, 28, 1\]———>\[None, 28, 28, 32\]
        *   权重数量：\[5, 5, 1 ,32\]
        *   偏置数量：\[32\]
    *   激活：\[None, 28, 28, 32\]———>\[None, 28, 28, 32\]
    *   池化：\[None, 28, 28, 32\]———>\[None, 14, 14, 32\]
*   第二层
    *   卷积：\[None, 14, 14, 32\]———>\[None, 14, 14, 64\]
        *   权重数量：\[5, 5, 32 ,64\]
        *   偏置数量：\[64\]
    *   激活：\[None, 14, 14, 64\]———>\[None, 14, 14, 64\]
    *   池化：\[None, 14, 14, 64\]———>\[None, 7, 7, 64\]
*   全连接层
    *   \[None, 7, 7, 64\]———>\[None, 7 _7_ 64\]
    *   权重数量：\[7 _7_ 64, 10\]，由分类别数而定
    *   偏置数量：\[10\]，由分类别数而定

4.3.2 案例：CNN识别Mnist手写数字
-----------------------

### 1 手写字识别分析

1.  准备手写数字数据, 可以通过tensorflow
2.  实现前面设计的网络结构: 卷积、激活、池化（两层）
3.  全连接层得到输出类别预测
4.  计算损失值并优化
5.  计算准确率

### 2 代码

* 网络结构实现

  def conv_model():
  ​    """
  ​    自定义的卷积网络结构
  ​    :return: x, y_true, y_predict
  ​    """
  ​    # 1、准备数据占位符
  ​    # x [None, 784]  y_true [None, 10]
  ​    with tf.variable_scope("data"):

          x = tf.placeholder(tf.float32, [None, 784])
      
          y_true = tf.placeholder(tf.int32, [None, 10])
      
      # 2、卷积层一 32个filter, 大小5*5,strides=1, padding=“SAME”
      
      with tf.variable_scope("conv1"):
          # 随机初始化这一层卷积权重 [5, 5, 1, 32], 偏置[32]
          w_conv1 = weight_variables([5, 5, 1, 32])
      
          b_conv1 = bias_variables([32])
      
          # 首先进行卷积计算
          # x [None, 784]--->[None, 28, 28, 1]  x_conv1 -->[None, 28, 28, 32]
          x_conv1_reshape = tf.reshape(x, [-1, 28, 28, 1])
          # input-->4D
          x_conv1 = tf.nn.conv2d(x_conv1_reshape, w_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1
      
          # 进行激活函数计算
          #  x_relu1 -->[None, 28, 28, 32]
          x_relu1 = tf.nn.relu(x_conv1)
      
          # 进行池化层计算
          # 2*2, strides 2
          #  [None, 28, 28, 32]------>[None, 14, 14, 32]
          x_pool1 = tf.nn.max_pool(x_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
      
      # 3、卷积层二 64个filter, 大小5*5,strides=1,padding=“SAME”
      # 输入：[None, 14, 14, 32]
      with tf.variable_scope("conv2"):
          # 每个filter带32张5*5的观察权重，一共有64个filter去观察
          # 随机初始化这一层卷积权重 [5, 5, 32, 64], 偏置[64]
          w_conv2 = weight_variables([5, 5, 32, 64])
      
          b_conv2 = bias_variables([64])
      
          # 首先进行卷积计算
          # x [None, 14, 14, 32]  x_conv2 -->[None, 14, 14, 64]
          # input-->4D
          x_conv2 = tf.nn.conv2d(x_pool1, w_conv2, strides=[1, 1, 1, 1], padding="SAME") + b_conv2
      
          # 进行激活函数计算
          #  x_relu1 -->[None, 28, 28, 32]
          x_relu2 = tf.nn.relu(x_conv2)
      
          # 进行池化层计算
          # 2*2, strides 2
          #  [None, 14, 14, 64]------>x_pool2[None, 7, 7, 64]
          x_pool2 = tf.nn.max_pool(x_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
      
      # 4、全连接层输出
      # 每个样本输出类别的个数10个结果
      # 输入：x_poll2 = [None, 7, 7, 64]
      # 矩阵运算： [None, 7 * 7 * 64] * [7 * 7 * 64, 10] +[10] = [None, 10]
      with tf.variable_scope("fc"):
          # 确定全连接层权重和偏置
          w_fc = weight_variables([7 * 7 * 64, 10])
      
          b_fc = bias_variables([10])
      
          # 对上一层的输出结果的形状进行处理成2维形状
          x_fc = tf.reshape(x_pool2, [-1, 7 * 7 * 64])
      
          # 进行全连接层运算
          y_predict = tf.matmul(x_fc, w_fc) + b_fc
      
      return x, y_true, y_predict

* 损失计算优化、准确率计算

      # 1、准备数据API
      mnist = input_data.read_data_sets("./data/mnist/input_data/", one_hot=True)
      
      # 2、定义模型,两个卷积层、一个全连接层
      x, y_true, y_predict = conv_model()
      
      # 3、softmax计算和损失计算
      with tf.variable_scope("softmax_loss"):
      
          # labels:真实值 [None, 10]  one_hot
          # logits:全脸层的输出[None,10]
          # 返回每个样本的损失组成的列表
          loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,
                                                                        logits=y_predict))
      # 4、梯度下降损失优化
      with tf.variable_scope("optimizer"):
          # 学习率
          train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
          # train_op = tf.train.AdamOptimizer(0.1).minimize(loss)
      
      # 5、准确率计算
      with tf.variable_scope("accuracy"):
      
          equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
      
          accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))
      
      # 初始化变量op
      init_op = tf.global_variables_initializer()

* 会话运行

  # 会话运行
      with tf.Session() as sess:
      
          # 初始化变量
          sess.run(init_op)
      
          # 循环去训练模型
          for i in range(2000):
      
              # 获取数据，实时提供
              # 每步提供50个样本训练
              mnist_x, mnist_y = mnist.train.next_batch(50)
      
              sess.run(train_op, feed_dict={x:mnist_x, y_true: mnist_y})
      
              # 打印准确率大小
              print("第%d步训练的准确率为:--%f" % (i,
                                          sess.run(accuracy, feed_dict={x:mnist_x, y_true: mnist_y})
                                          ))


完整代码：

    import tensorflow as tf
    import os
    from tensorflow.examples.tutorials.mnist import input_data
    
    mnist = input_data.read_data_sets("./mnist_data", one_hot=True)
    # 利用数据，在训练的时候实时提供数据
    # mnist手写数字数据在运行时候实时提供给给占位符
    tf.app.flags.DEFINE_integer("is_train", 1, "指定是否是训练模型，还是拿数据去预测")
    FLAGS = tf.app.flags.FLAGS
    
    # 随机初始化权重的统一API
    def create_weights(shape):
    
        return tf.Variable(initial_value=tf.random_normal(shape=shape))


​    
    def conv_model(x):
        """
        实现卷积神经网络模型
        :param x: [None, 784]
        :return:
        """
        # 第一个卷积大层
        with tf.variable_scope("conv1"):
            # 调整图片形状--> [None, 28, 28, 1]
            # 卷积层：
            # 32
            # 个filter，大小5 * 5，步长1
            # filter：进行随机初始化权重，形状[5, 5, 1, 32]
            # bias: [32]
            # strides = [1, 1, 1, 1]
            # padding = "SAME"
            # 原图的形状改变为：[None, 28, 28, 32]
            x_reshape = tf.reshape(x, [-1, 28, 28, 1])
            filter1_weights = create_weights([5, 5, 1, 32])
            filter1_bias = create_weights([32])
    
            # 卷积层
            x_conv1 = tf.nn.conv2d(x_reshape, filter1_weights, strides=[1, 1, 1, 1], padding="SAME") + filter1_bias
    
            # 激活函数
            x_relu1 = tf.nn.relu(x_conv1)
    
            # 池化层
            # 池化层：
            # 输入形状为：[None, 28, 28, 32]
            # ksize = [1, 2, 2, 1]
            # strides = [1, 2, 2, 1]
            # 原图的形状改变为：[None, 14, 14, 32]
            x_pool1 = tf.nn.max_pool(x_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        # 第二个卷积大层
        with tf.variable_scope("conv2"):
            # 初始化filter的权重和偏置
            filter2_weights = create_weights([5, 5, 32, 64])
            filter2_bias = create_weights([64])
            # 卷积层：
            # 64个filter, 大小5 * 5， 步长1
            # filter: 进行随机初始化权重，形状[5, 5, 32, 64]
            # bias: [64]
            # strides = [1, 1, 1, 1]
            # padding = "SAME"
            # 原图的形状改变为：[None, 14, 14, 64]
            x_conv2 = tf.nn.conv2d(x_pool1, filter2_weights, strides=[1, 1, 1, 1], padding="SAME") + filter2_bias
    
            # 激活函数
            x_relu2 = tf.nn.relu(x_conv2)
    
            # 池化层
            # 池化层：
            # 输入形状为：[None, 14, 14, 64]
            # ksize = [1, 2, 2, 1]
            # strides = [1, 2, 2, 1]
            # 原图的形状改变为：[None, 7, 7, 64]
            x_pool2 = tf.nn.max_pool(x_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    
        # 全连接层
        with tf.variable_scope("full_connection"):
            # 形状改变：
            # [None, 7 * 7 * 64] * [7 * 7 * 64, 10] = [None, 10]
            # weights: [7 * 7 * 64, 10]
            # bias: [10]
            x_reshape2 = tf.reshape(x_pool2, [-1, 7 * 7 * 64])
    
            # 初始化全连接层的权重和偏置
            fc_weights = create_weights([7 * 7 * 64, 10])
            fc_bias = create_weights([10])
    
            # 全连接
            y_predict = tf.matmul(x_reshape2, fc_weights) + fc_bias
    
        return y_predict


​    
    def mnist_recognize():
        """
        识别手写数字图片
        特征值：[None, 784]
        目标值：one_hot编码 [None, 10]
        :return:
        """
        # 1、准备数据
        # x [None, 784] y_true [None, 10]
        with tf.variable_scope("prepare_data"):
            x = tf.placeholder(tf.float32, [None, 784])
            y_true = tf.placeholder(tf.float32, [None, 10])
        # 2、卷积神经网络模型
        y_predict = conv_model(x)
        # 3、softmax回归以及交叉熵损失计算
        with tf.variable_scope("softmax_crossentropy"):
            # labels:真实值 [None, 10]  one_hot
            # logits:全脸层的输出[None,10]
            # 返回每个样本的损失组成的列表
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))
        # 4、梯度下降损失优化
        with tf.variable_scope("optimizer"):
            # 学习率
            train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
        # 5、得出每次训练的准确率（通过真实值和预测值进行位置比较，每个样本都比较）
        with tf.variable_scope("accuracy"):
            equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
            accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))
        # （2）收集要显示的变量
        # 先收集损失和准确率
        tf.summary.scalar("losses", loss)
        tf.summary.scalar("acc", accuracy)
        # # 收集权重和偏置
        # tf.summary.histogram("weights", weight)
        # tf.summary.histogram("bias", bias)
        # 初始化变量op
        init_op = tf.global_variables_initializer()
        # （3）合并所有变量op
        merged = tf.summary.merge_all()
        # 创建模型保存和加载
        saver = tf.train.Saver()
        # 开启会话去训练
        with tf.Session() as sess:
            # 初始化变量
            sess.run(init_op)
            # （1）创建一个events文件实例
            file_writer = tf.summary.FileWriter("./tmp/summary/", graph=sess.graph)
            # 加载模型
            if os.path.exists("./tmp/modelckpt/checkpoint"):
                saver.restore(sess, "./tmp/modelckpt/fc_nn_model")
            if FLAGS.is_train == 1:
                # 循环步数去训练
                for i in range(3000):
                    # 获取数据，实时提供
                    # 每步提供50个样本训练
                    mnist_x, mnist_y = mnist.train.next_batch(50)
                    # 运行训练op
                    sess.run(train_op, feed_dict={x: mnist_x, y_true: mnist_y})
                    print("训练第%d步的准确率为：%f, 损失为：%f " % (i+1,
                                         sess.run(accuracy, feed_dict={x: mnist_x, y_true: mnist_y}),
                                         sess.run(loss, feed_dict={x: mnist_x, y_true: mnist_y})
                                         )
                      )
    
                    # 4）运行合变量op，写入事件文件当中
                    summary = sess.run(merged, feed_dict={x: mnist_x, y_true: mnist_y})
                    file_writer.add_summary(summary, i)
                    if i % 100 == 0:
                        saver.save(sess, "./tmp/modelckpt/fc_nn_model")
            else:
                # 如果不是训练，我们就去进行预测测试集数据
                for i in range(100):
                    # 每次拿一个样本预测
                    mnist_x, mnist_y = mnist.test.next_batch(1)
                    print("第%d个样本的真实值为：%d, 模型预测结果为：%d" % (
                                                          i+1,
                                                          tf.argmax(sess.run(y_true, feed_dict={x: mnist_x, y_true: mnist_y}), 1).eval(),
                                                          tf.argmax(sess.run(y_predict, feed_dict={x: mnist_x, y_true: mnist_y}), 1).eval()
                                                          )
                                                          )
    
        return None
    if __name__ == "__main__":
        mnist_recognize()


### 3 学习率过大问题

**发现当我们设置0.1的学习率之后，准确率一直上不去，并且打印参数发现已经变为NaN，这个地方是不是与之前在做线性回归的时候似曾相识。对于卷积网络来说，更容易发生梯度爆炸现象，只能通过调节学习来避免。**

4.3.3 拓展-Tensorflow高级API实现结构
----------------------------

高级API可以更快的构建模型，但是对神经网络的运行流程了解清晰还是需要去使用底层API去构建模型，更好理解网络的原理

[https://www.tensorflow.org/tutorials/layers](https://www.tensorflow.org/tutorials/layers)

    def cnn_model_fn(features, labels, mode):
      """Model function for CNN."""
      # Input Layer
      input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    
      # Convolutional Layer #1
      conv1 = tf.layers.conv2d(
          inputs=input_layer,
          filters=32,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)
    
      # Pooling Layer #1
      pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    
      # Convolutional Layer #2 and Pooling Layer #2
      conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=64,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)
      pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    
      # 全连接层计算+dropout
      pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
      dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
      dropout = tf.layers.dropout(
          inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    
      # 得出预测结果
      logits = tf.layers.dense(inputs=dropout, units=10)
    
      predictions = {
          # Generate predictions (for PREDICT and EVAL mode)
          "classes": tf.argmax(input=logits, axis=1),
          # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
          # `logging_hook`.
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
      }
    
      if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
      # 计算损失
      loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    
      # 配置train_op
      if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
      # 评估模型
      eval_metric_ops = {
          "accuracy": tf.metrics.accuracy(
              labels=labels, predictions=predictions["classes"])}
      return tf.estimator.EstimatorSpec(
          mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
