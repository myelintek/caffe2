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
    # conv1 + maxpool
    conv1 = brew.conv(
        model,
        data,
        "conv1",
        num_input_channels,
        96,
        kernel=11,
        weight_init=('MSRAFill', {}),
        stride=4,
        pad=2
    )
    brew.relu(model, conv1, "conv1")
    norm1 = brew.lrn(model, conv1, "norm1", size=5, alpha=0.0001, beta=0.75)
    pool1 = brew.max_pool(model, norm1, "pool1", kernel=3, stride=2)

    conv2 = brew.group_conv(
        model,
        pool1,
        "conv2",
        96,
        256,
        kernel=5,
        weight_init=('MSRAFill', {}),
        group=2,
        stride=1,
        pad=2
    )
    brew.relu(model, conv2, "conv2")
    norm2 = brew.lrn(model, conv2, "norm2", size=5, alpha=0.0001, beta=0.75)
    pool2 = brew.max_pool(model, norm2, "pool2", kernel=3, stride=2)

    conv3 = brew.conv(
        model,
        pool2,
        "conv3",
        256,
        384,
        kernel=3,
        weight_init=('MSRAFill', {}),
        pad=1
    )
    brew.relu(model, conv3, "conv3")
    conv4 = brew.group_conv(
        model,
        conv3,
        "conv4",
        384,
        384,
        kernel=3,
        weight_init=('MSRAFill', {}),
        group=2,
        pad=1
    )
    brew.relu(model, conv4, "conv4")
    conv5 = brew.group_conv(
        model,
        conv4,
        "conv5",
        384,
        256,
        kernel=3,
        weight_init=('MSRAFill', {}),
        group=2,
        pad=1
    )
    brew.relu(model, conv5, "conv5")
    pool5 = brew.max_pool(model, conv5, "pool5", kernel=3, stride=2)
    fc6 = brew.fc(
        model,
        pool5,
        "fc6",
        256 * 6 * 6,
        4096
    )
    relu6 = brew.relu(model, fc6, "relu6")
    drop6 = brew.dropout(model, relu6, "drop6", use_cudnn=True, ratio=0.5, is_test=is_test)
    fc7 = brew.fc(
        model,
        drop6,
        "fc7",
        4096,
        4096
    )
    brew.relu(model, fc7, "fc7")
    drop7 = brew.dropout(model, fc7, "drop7", use_cudnn=True, ratio=0.5, is_test=is_test)
    last_out = brew.fc(
        model,
        drop7,
        'last_out_L{}'.format(num_labels),
        4096,
        num_labels
    )

    if no_loss:
        return last_out

    # If we create model for training, use softmax-with-loss
    if (label is not None):
        (softmax, loss) = model.SoftmaxWithLoss(
            [last_out, label],
            ["softmax", "loss"],
        )

        return (softmax, loss)
    else:
        # For inference, we just return softmax
        return brew.softmax(model, last_out, "softmax")
