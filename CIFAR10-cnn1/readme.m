% 卷积神经网络训练CIFAR10数据集：
% 准确率约为50%
% 卷积所用的时间特别长。
% 
% 1.从训练数据中随机选取100000个8*8的小块（50000个训练数据，每个数据中随机选取其中的两个小块），作为线性自编码的训练数据。
% 训练得到编码器的参数，保存在CIFAR10Features.mat中。
% 特征个数为400。
% 
% 2.对训练数据和测试数据进行卷积，池化，将池化后的数据保存在cnnPooledFeatures.mat中。
% 卷积小块的size为8*8，池化的size为5*5,。
% 池化后，每个数据大小为400*5*5。
% 
% 3.使用softmax分类器对池化后的数据进行分类。