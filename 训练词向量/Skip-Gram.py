# 1.导入所依赖的库
import sys
import time
import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib
import pprint
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 2.准备数据集

url = "http://mattmahoney.net/dc/"


def maybe_download(filename, expected_bytes):
    """
    判断文件是否已经下载，如果没有，则下载数据集
    """

    def _progress(block_num, block_size, total_size):
        '''回调函数
           @block_num: 已经下载的数据块
           @block_size: 数据块的大小
           @total_size: 远程文件的大小
        '''
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(block_num * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
    if not os.path.exists(filename):
        # 数据集不存在，开始下载
        filename, _ = urllib.request.urlretrieve(url + filename, filename, _progress)
        # 有关函数urllib.request.urlretrieve的详解博客https://blog.csdn.net/pursuit_zhangyu/article/details/80556275
    # 核对文件尺寸
    stateinfo = os.stat(filename)
    # 有关系统stat调用的详细解释链接https://www.cnblogs.com/fmgao-technology/p/9056425.html
    if stateinfo.st_size == expected_bytes:
        print("数据集已存在，且文件尺寸合格！", filename)
    else:
        print(stateinfo.st_size)
        raise Exception(
            "文件尺寸不对 !请重新下载，下载地址为：" + url
        )
    return filename


"""
测试文件是否存在
"""
filename = maybe_download("text8.zip", 31344016)


# 3.解压文件
def read_data(filename):
    # 有关zipfile模块的详细使用介绍博客https://www.cnblogs.com/ManyQian/p/9193199.html
    with zipfile.ZipFile(filename) as f:
        # 有关python中split()函数的详细解释博客https://www.runoob.com/python/att-string-split.html
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
        '''
        使用 zipfile.ZipFile（）来提取压缩文件，然后我们可以使用 
        zipfile 模块中的读取器功能。首先，namelist（）函数检索该
        档案中的所有成员——在本例中只有一个成员，所以我们可以使
        用 0 索引对其进行访问。然后，我们使用 read（）函数读取文
        件中的所有文本，并传递给 TensorFlow 的 as_str 函数，以确
        保文本保存为字符串数据类型。最后，我们使用 split（）函数
        创建一个列表，该列表包含文本文件中所有的单词，并用空格字
        符分隔'''
    return data


words = read_data(filename)
print("总的单词个数：", len(words))

# 4.构建词汇表，并统计每个单词出现的频数，同时用字典的形式进行存储，取频数排名前50000的单词
vocabulary_size = 50000


def build_dataset(words):
    count = [["unkown", -1]]
    # collections.Counter()返回的是形如[["unkown",-1],("the",4),("physics",2)]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    # 有关collection模块中counter类的详细解释链接http://www.pythoner.com/205.html
    # most_common()函数用来实现Top n 功能 即截取counter结果的前多少个子项
    # 对于列表的一些常见基本操作详细链接https://blog.csdn.net/ywx1832990/article/details/78928238
    dictionary = {}
    # 将全部单词转为编号（以频数排序的编号），我们只关注top50000的单词，以外的认为是unknown的，编号为0，同时统计一下这类词汇的数量
    for word, _ in count:
        dictionary[word] = len(dictionary)
        # 形如：{"the"：1，"UNK"：0，"a"：12}
    data = []
    unk_count = 0  # 准备统计top50000以外的单词的个数
    for word in words:
        # 对于其中每一个单词，首先判断是否出现在字典当中
        if word in dictionary:
            # 如果已经出现在字典中，则转为其编号
            index = dictionary[word]
        else:
            # 如果不在字典，则转为编号0
            index = 0
            unk_count += 1
        data.append(index)  # 此时单词已经转变成对应的编号
    """
    print(data[:10]) 
    [5234, 3081, 12, 6, 195, 2, 3134, 46, 59, 156]
    """
    count[0][1] = unk_count  # 将统计好的unknown的单词数，填入count中
    # 将字典进行翻转,形如：{3:"the,4:"an"}
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


# 为了节省内存，将原始单词列表进行删除
data, count, dictionary, reverse_dictionary = build_dataset(words)
del words
# 将部分结果展示出来
# print("出现频率最高的单词（包括未知类别的）：",count[:10])
# 将已经转换为编号的数据进行输出，从data中输出频数，从翻转字典中输出编号对应的单词
# print("样本数据(排名)：",data[:10],"\n对应的单词",[reverse_dictionary[i] for i in data[:10]])

# 5.生成Word2Vec的训练样本，使用skip-gram模式
data_index = 0


def generate_batch(batch_size, num_skips, skip_window):
    """

    :param batch_size: 每个训练批次的数据量
    :param num_skips: 每个单词生成的样本数量，不能超过skip_window的两倍，并且必须是batch_size的整数倍
    :param skip_window: 单词最远可以联系的距离，设置为1则表示当前单词只考虑前后两个单词之间的关系，也称为滑窗的大小
    :return:返回每个批次的样本以及对应的标签
    """
    global data_index  # 声明为全局变量，方便后期多次使用
    # 使用Python中的断言函数，提前对输入的参数进行判别，防止后期出bug而难以寻找原因
    assert batch_size % num_skips == 0
    assert num_skips <= skip_window * 2

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)  # 创建一个batch_size大小的数组，数据类型为int32类型，数值随机
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)  # 数据维度为[batch_size,1]
    span = 2 * skip_window + 1  # 入队的长度
    buffer = collections.deque(maxlen=span)  # 创建双向队列。最大长度为span
    """
    print(batch,"\n",labels) 
    batch :[0 ,-805306368  ,405222565 ,1610614781 ,-2106392574 ,2721-2106373584 ,163793]
    labels: [[         0]
            [-805306368]
            [ 407791039]
            [ 536872957]
            [         2]
            [         0]
            [         0]
            [    131072]]
    """
    # 对双向队列填入初始值
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    """
    print(buffer,"\n",data_index)  输出：
                                    deque([5234, 3081, 12], maxlen=3) 
                                    3
    """
    # 进入第一层循环，i表示第几次入双向队列
    for i in range(batch_size // num_skips):
        target = skip_window  # 定义buffer中第skip_window个单词是目标
        targets_avoid = [skip_window]  # 定义生成样本时需要避免的单词，因为我们要预测的是语境单词，不包括目标单词本身，因此列表开始包括第skip_window个单词
        for j in range(num_skips):
            """第二层循环，每次循环对一个语境单词生成样本，先产生随机数，直到不在需要避免的单词中，也即需要找到可以使用的语境词语"""
            while target in targets_avoid:
                target = random.randint(0, span - 1)
            targets_avoid.append(target)  # 因为该语境单词已经被使用过了，因此将其添加到需要避免的单词库中
            batch[i * num_skips + j] = buffer[skip_window]  # 目标词汇
            labels[i * num_skips + j, 0] = buffer[target]  # 语境词汇
        # 此时buffer已经填满，后续的数据会覆盖掉前面的数据
        # print(batch,labels)
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


batch, labels = generate_batch(8, 2, 1)
"""
for i in range(8):
    print("目标单词："+reverse_dictionary[batch[i]]+"对应编号为：".center(20)+str(batch[i])+"   对应的语境单词为: ".ljust(20)+reverse_dictionary[labels[i,0]]+"    编号为",labels[i,0])
测试结果：
目标单词：originated         对应编号为：   3081    对应的语境单词为:  as           编号为 12
目标单词：originated         对应编号为：   3081    对应的语境单词为:  anarchism    编号为 5234
目标单词：as                 对应编号为：   12      对应的语境单词为:  originated   编号为 3081
目标单词：as                 对应编号为：   12      对应的语境单词为:  a            编号为 6
目标单词：a              对应编号为：   6       对应的语境单词为:  as           编号为 12
目标单词：a              对应编号为：   6       对应的语境单词为:  term         编号为 195
目标单词：term           对应编号为：   195     对应的语境单词为:  of           编号为 2
目标单词：term           对应编号为：   95      对应的语境单词为:  a            编号为 6
"""

# 6.定义训练数据的一些参数
batch_size = 128  # 训练样本的批次大小
embedding_size = 128  # 单词转化为稠密词向量的维度
skip_window = 1  # 单词可以联系到的最远距离
num_skips = 1  # 每个目标单词提取的样本数

# 7.定义验证数据的一些参数
valid_size = 16  # 验证的单词数
valid_window = 100  # 指验证单词只从频数最高的前100个单词中进行抽取
valid_examples = np.random.choice(valid_window, valid_size, replace=False)  # 进行随机抽取
num_sampled = 64  # 训练时用来做负样本的噪声单词的数量

# 8.开始定义Skip-Gram Word2Vec模型的网络结构
# 8.1创建一个graph作为默认的计算图，同时为输入数据和标签申请占位符，并将验证样例的随机数保存成TensorFlow的常数
graph = tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, [batch_size])
    train_labels = tf.placeholder(tf.int32, [batch_size, 1])
    valid_dataset = tf.constant(valid_examples, tf.int32)

    # 选择运行的device为CPU
    with tf.device("/cpu:0"):
        # 单词大小为50000，向量维度为128，随机采样在（-1，1）之间的浮点数
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        # 使用tf.nn.embedding_lookup()函数查找train_inputs对应的向量embed
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        ###############运算慢##################
        # # 使用截断正太函数初始化权重,偏重初始化为0
        # weights = tf.Variable(
        #     tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        # biases = tf.Variable(tf.zeros([vocabulary_size]))
        # # 隐藏层实现
        # hidden_out = tf.matmul(embed, tf.transpose(weights)) + biases
        # # 将标签使用one-hot方式表示，便于在softmax的时候进行判断生成是否准确
        # train_one_hot = tf.one_hot(train_labels, vocabulary_size)
        # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hidden_out, labels=train_one_hot))
        # # 优化选择随机梯度下降
        # optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(cross_entropy)
        # # 为了方便进行验证，采用余弦定理验证相似性  链接https://blog.csdn.net/u012160689/article/details/15341303
        # # 归一化
        # norm = tf.sqrt(tf.reduce_sum(tf.square(weights), 1, keep_dims=True))
        # normalized_embeddings = weights / norm
        ################运算慢#####################

        # 优化目标选择NCE loss
        # 使用截断正太函数初始化NCE损失的权重,偏重初始化为0
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
        # 计算学习出的embedding在训练数据集上的loss，并使用tf.reduce_mean()函数进行汇总
        loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weights,
                                            biases = nce_biases,
                                            labels = train_labels,
                                            inputs = embed,
                                            num_sampled = num_sampled,
                                            num_classes = vocabulary_size
                                            ))
        # 定义优化器为SGD，且学习率设置为1.0.然后计算嵌入向量embeddings的L2范数norm，并计算出标准化后的normalized_embeddings
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
        norm = tf.sqrt(tf.reduce_sum(tf.square(nce_weights), 1, keep_dims=True))  # 嵌入向量的L2范数
        normalized_embeddings = nce_weights / norm  # 标准哈embeddings

        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)  # 查询验证单词的嵌入向量
        # 计算验证单词的嵌入向量与词汇表中所有单词的相似性
        similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True
        )
        init = tf.global_variables_initializer()  # 定义参数的初始化

##9.启动训练
num_steps = 20001  # 进行15W次的迭代计算
t0 = time.time()
# 创建一个回话并设置为默认
with tf.Session(graph=graph) as session:
    init.run()  # 启动参数的初始化
    print("初始化完成！")
    average_loss = 0  # 计算误差
    # 开始迭代训练
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)  # 调用生成训练数据函数生成一组batch和label
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}  # 待填充的数据
        # 启动回话，运行优化器optimizer和损失计算函数，并填充数据
        # optimizer_trained, loss_val = session.run([optimizer, cross_entropy], feed_dict=feed_dict)
        optimizer_trained, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val  # 统计NCE损失

        # 为了方便，每2000次计算一下损失并显示出来
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            print('第%d轮迭代用时：%s' % (step, time.time() - t0))
            t0 = time.time()
            print("第{}轮迭代后的损失为：{}".format(step, average_loss))
            average_loss = 0

        # 每10000次迭代，计算一次验证单词与全部单词的相似度，并将于验证单词最相似的前8个单词呈现出来
        if step % 10000 == 0:
            sim = similarity.eval()  # 计算向量
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]  # 得到对应的验证单词
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]  # 计算每一个验证单词相似度最接近的前8个单词
                log_str = "与单词 {} 最相似的： ".format(str(valid_word))

                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]  # 相似度高的单词
                    log_str = "%s %s, " % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()


# 10.可视化Word2Vec效果
def plot_with_labels(low_dim_embs, labels, filename="tsne.png"):
    assert low_dim_embs.shape[0] >= len(labels), "标签数超过了嵌入向量的个数！！"

    plt.figure(figsize=(20, 20))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(5, 2),
            textcoords="offset points",
            ha="right",
            va="bottom"
        )
    plt.savefig(filename)


from sklearn.manifold import TSNE

tsne = TSNE(perplexity=30, n_components=2, init="pca", n_iter=5000)
plot_only = 100
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
Labels = [reverse_dictionary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, Labels)
