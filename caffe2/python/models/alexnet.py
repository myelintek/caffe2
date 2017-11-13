from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from caffe2.python import brew
'''
Utility for creating ResNets
See "Deep Residual Learning for Image Recognition" by He, Zhang et. al. 2015
'''


def create_alexnet(
    model,
    data,
    num_input_channels,
    num_labels,
    label=None,
    is_test=False,
    no_loss=False,
    no_bias=0,
    conv1_kernel=7,
    conv1_stride=2,
    final_avg_kernel=7,
):
    conv1 = brew.conv(
        model,
        data,
        "conv1",
        num_input_channels,
        96,
        11, ('XavierFill', {}), ('ConstantFill', {}),
        stride=4,
        pad=2
    )
    relu1 = brew.relu(model, conv1, "relu1")
    norm1 = brew.lrn(model, relu1, "norm1", size=5, alpha=0.0001, beta=0.75)
    pool1 = brew.max_pool(model, norm1, "pool1", kernel=3, stride=2)
    conv2 = brew.group_conv(
        model,
        pool1,
        "conv2",
        96,
        256,
        5,
        ('XavierFill', {}),
        ('ConstantFill', {}),
        group=2,
        stride=1,
        pad=2
    )
    relu2 = brew.relu(model, conv2, "relu2")
    norm2 = brew.lrn(model, relu2, "norm2", size=5, alpha=0.0001, beta=0.75)
    pool2 = brew.max_pool(model, norm2, "pool2", kernel=3, stride=2)
    conv3 = brew.conv(
        model,
        pool2,
        "conv3",
        256,
        384,
        3,
        ('XavierFill', {}),
        ('ConstantFill', {}),
        pad=1
    )
    relu3 = brew.relu(model, conv3, "relu3")
    conv4 = brew.group_conv(
        model,
        relu3,
        "conv4",
        384,
        384,
        3,
        ('XavierFill', {}),
        ('ConstantFill', {}),
        group=2,
        pad=1
    )
    relu4 = brew.relu(model, conv4, "relu4")
    conv5 = brew.group_conv(
        model,
        relu4,
        "conv5",
        384,
        256,
        3,
        ('XavierFill', {}),
        ('ConstantFill', {}),
        group=2,
        pad=1
    )
    relu5 = brew.relu(model, conv5, "relu5")
    pool5 = brew.max_pool(model, relu5, "pool5", kernel=3, stride=2)
    fc6 = brew.fc(
        model,
        pool5, "fc6", 256 * 6 * 6, 4096, ('XavierFill', {}),
        ('ConstantFill', {})
    )
    relu6 = brew.relu(model, fc6, "fc6")
    drop6 = brew.dropout(model, relu6, "drop6", use_cudnn=True)
    fc7 = brew.fc(
        model, drop6, "fc7", 4096, 4096, ('XavierFill', {}), ('ConstantFill', {})
    )
    relu7 = brew.relu(model, fc7, "fc7")
    drop7 = brew.dropout(model, relu7, "drop7", use_cudnn=True)
    fc8 = brew.fc(
        model, drop7, "fc8", 4096, 1000, ('XavierFill', {}), ('ConstantFill', {})
    )
    softmax = brew.softmax(model, fc8, "softmax")
    return softmax
