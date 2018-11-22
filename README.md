
* 代码参考自[wzyonggege](https://github.com/wzyonggege/RNN_poetry_generator)和[Determined22](https://github.com/Determined22/zh-NER-TF/),向两位作者表示感谢。

* 以以上两份代码为参考自己完成的代码，当作对tensorflow的联系和NLP相关的练习。只是基本代码，不能真正生成可读性很好的诗句来。

* 笔记本性能有限，只跑了一个epoch,但代码运行没有问题。

### 一些问题

* 注意计算loss时用到的**seq2seq.sequence_loss_by_example**

* 关于生成的原理还是没有很明白，尤其to_word(weights)函数为什么要这样设计

* 最开始自己的代码有个问题: 当数据总量不能被batch_size整除时就会报维度不匹配错误，原因在于对self.initial_state的初始化维度无法匹配。因此后续删除了对initial_state的初始化，通过tensorflow自动初始。

    * 关于init_state,根据[源码](https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/ops/rnn.py#L629-L638),在没有初始化的情况下，只要指明dtype,tensorflow即可自动初始化出init_state。
在作者的原代码中，使用self.initial_state = self.cell.zero_state(self.batch_size, tf.float32)进行了初始化，self.initial_state与self.x_tf的第一维度一定要是一致的，因此当最后一个batch不满足batch_size的容量时，就会报错(报错在于feed给self.x_tf的数据容量小于batch_size,与initial_staet无法对应)。
因此原作者的代码中crate_batch的方法会丢弃掉最后一个不满足batch_size的batch,因此是可以运行的。

    * 而根据tensorflow源码，可以不对self.initial_state进行初始化，这样就可以避免以上问题，只要feed时维度对应即可。