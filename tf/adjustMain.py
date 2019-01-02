import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

hyperparams = {}
hyperparams_set = []

# 加载模型,并返回指定的tensor
def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # 加载和恢复模型
    model = tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    # 获取当前默认计算图
    graph = tf.get_default_graph()

    # 返回给定名称的tensor
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return image_input, keep_prob, layer3, layer4, layer7
    
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # https://www.cnblogs.com/guqiangjs/p/7807852.html
    # 规则化可以帮助防止过度配合，提高模型的适用性。（让模型无法完美匹配所有的训练项。）（使用规则来使用尽量少的变量去拟合数据）
    # 规则化就是说给需要训练的目标函数加上一些规则（限制），让他们不要自我膨胀。
    kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3)
    # https://blog.csdn.net/liyaohhh/article/details/77165483
    # 参数初始化
    kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d()
    
    # https://blog.csdn.net/gqixf/article/details/80519912
    pool3_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, kernel_size=1, 
                                 padding='same', 
                                 kernel_initializer=kernel_initializer, 
                                 kernel_regularizer=kernel_regularizer)
    
    pool4_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, kernel_size=1, 
                                 padding='same', 
                                 kernel_initializer=kernel_initializer, 
                                 kernel_regularizer=kernel_regularizer)
    
    conv7_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, kernel_size=1, 
                                 padding='same', 
                                 kernel_initializer=kernel_initializer, 
                                 kernel_regularizer=kernel_regularizer)
    
    # make prediction of segmentation
    # 反卷积操作
    deconv7 = tf.layers.conv2d_transpose(conv7_1x1, num_classes, kernel_size=4, strides=2, padding='same',
                                         kernel_initializer=kernel_initializer, 
                                         kernel_regularizer=kernel_regularizer)
    
    fuse1 = tf.add(deconv7, pool4_1x1)
    deconv_fuse1 = tf.layers.conv2d_transpose(fuse1, num_classes, kernel_size=4, strides=2, padding='same',
                                              kernel_initializer=kernel_initializer,
                                              kernel_regularizer=kernel_regularizer)
    
    fuse2 = tf.add(deconv_fuse1, pool3_1x1)
    
    out = tf.layers.conv2d_transpose(fuse2, num_classes, kernel_size=16, strides=8, padding='same',
                                     kernel_initializer=kernel_initializer, 
                                     kernel_regularizer=kernel_regularizer)
    
    return out

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    # tf.nn.softmax_cross_entropy_with_logits:求交叉熵的函数,计算labels和logits之间的交叉熵
    # 交叉熵（Cross Entropy）是Shannon信息论中一个重要概念，主要用于度量两个概率分布间的差异性信息
    # tf.reduce_mean:计算tensor（图像）的平均值
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
#     train_op = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(cross_entropy_loss)
    # 此函数是Adam优化算法：是一个寻找全局最优点的优化算法，引入了二次方梯度校正。
    # 相比于基础SGD算法，1.不容易陷于局部优点。2.速度更快
    # Adam：一种随机优化方法:https://blog.csdn.net/zj360202/article/details/70262874
    # https://blog.csdn.net/shenxiaoming77/article/details/77169756
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss)
    
    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    # 初始化 计算图
    sess.run(tf.global_variables_initializer())
    loss_per_epoch = []
    for epoch in range(epochs):
        losses, i = [], 0
        for images, labels in get_batches_fn(batch_size):
            i += 1
            feed_dict = {input_image: images, 
                         correct_label: labels, 
                         keep_prob: KEEP_PROB, 
                         learning_rate: LEARNING_RATE}
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)
            losses.append(loss)
        
        training_loss = sum(losses) / len(losses)
        loss_per_epoch.append(training_loss)
        print(" [-] epoch: %d/%d, loss: %.5f" % (epoch+1, epochs, training_loss))
    return loss_per_epoch
    # pass
tests.test_train_nn(train_nn)


def run():
    # HYPERPARAMETERS
    for i in range(20):
        LEARNING_RATE = np.random.uniform(1e-5, 5*1e-4)
        BATCH_SIZE = np.random.randint(2, 6)
        KEEP_PROB = np.random.uniform(0.2, 0.7)
        print("BATCH_SIZE: ", BATCH_SIZE)
        print("LEARNING_RATE: ", LEARNING_RATE)
        print("KEEP_PROB", KEEP_PROB)

        def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
                    correct_label, keep_prob, learning_rate):
            """
            Train neural network and print out the loss during training.
            :param sess: TF Session
            :param epochs: Number of epochs
            :param batch_size: Batch size
            :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
            :param train_op: TF Operation to train the neural network
            :param cross_entropy_loss: TF Tensor for the amount of loss
            :param input_image: TF Placeholder for input images
            :param correct_label: TF Placeholder for label images
            :param keep_prob: TF Placeholder for dropout keep probability
            :param learning_rate: TF Placeholder for learning rate
            """
            # TODO: Implement function
            sess.run(tf.global_variables_initializer())
            loss_per_epoch = []
            for epoch in range(epochs):
                losses, i = [], 0
                for images, labels in get_batches_fn(batch_size):
                    i += 1
                    feed_dict = {input_image: images, 
                                correct_label: labels, 
                                keep_prob: KEEP_PROB, 
                                learning_rate: LEARNING_RATE}
                    _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)
                    losses.append(loss)

                training_loss = sum(losses) / len(losses)
                loss_per_epoch.append(training_loss)
                if (epoch+1) % 5 == 0:
                    print(" [-] epoch: %d/%d, loss: %.5f" % (epoch+1, epochs, training_loss))
            return loss_per_epoch

        num_classes = 2
        image_shape = (224, 224)
        data_dir = './data'
        
        tf.reset_default_graph()

        # Download pretrained vgg model
        helper.maybe_download_pretrained_vgg(data_dir)

        with tf.Session() as sess:
            # Path to vgg model
            vgg_path = os.path.join(data_dir, 'vgg')
            # Create function to get batches
            get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'training'), image_shape)

            image_input, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
            out = layers(layer3, layer4, layer7, num_classes)

            correct_label = tf.placeholder(tf.int32)
            learning_rate = tf.placeholder(tf.float32)
            logits, train_op, cross_entropy_loss = optimize(out, correct_label, learning_rate, num_classes)

            epochs = 20
            batch_size = BATCH_SIZE
            loss_per_epoch = train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, image_input, correct_label, keep_prob, learning_rate)

            # record results
            hyperparams = {'learning_rate': LEARNING_RATE, 
                        'BATCH_SIZE': BATCH_SIZE, 
                        'KEEP_PROB': KEEP_PROB, 
                        'LOSS_PER_EPOCH': loss_per_epoch}

            hyperparams_set.append(hyperparams)

            fp = open("result.txt",'w')
            fp.write(hyperparams_set)
            fp.close()


if __name__ == '__main__':
    run()
