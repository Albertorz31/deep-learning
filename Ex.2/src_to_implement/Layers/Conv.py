import numpy as np
from scipy import signal

from Base import BaseLayer


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        # convolution shape determines whether this object provides a 1D or a 2D
        # convolution layer. For 1D, it has the shape [c, m], whereas for 2D, it has the shape
        # [c, m, n], where c represents the number of input channels, and m, n represent the
        # spatial extent of the filter kernel.
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels

        self.weights = np.random.uniform(size=(num_kernels,) + convolution_shape)
        self.bias = np.random.uniform(size=num_kernels)

        self.optimizer = None
        self.bias_optimizer = None

        self.gradient_weights = np.random.uniform(size=(num_kernels,) + convolution_shape)
        self.gradient_bias = None

        self.prev_input_tensor = None  # to record prev tensor
        self.stride_row = self.stride_shape[0]
        if len(self.convolution_shape) == 2:
            self.dim = 1
        elif len(self.convolution_shape) == 3:
            self.dim = 2
            self.stride_col = self.stride_shape[1]

    def forward(self, input_tensor):
        self.prev_input_tensor = input_tensor
        batch_size = input_tensor.shape[0]

        if self.dim == 1:  # 1D
            output_tensor_size = (
                batch_size,
                self.num_kernels,
                int(np.ceil(input_tensor.shape[2] / self.stride_row))
            )
        elif self.dim == 2:  # 2D
            output_tensor_size = (
                batch_size, self.num_kernels,
                int(np.ceil(input_tensor.shape[2] / self.stride_row)),  # shape after stride in dim y
                int(np.ceil(input_tensor.shape[3] / self.stride_col))  # shape after stride in dim x
            )
        output_tensor = np.zeros(output_tensor_size)

        for batch in range(batch_size):
            # self.weights.shape[0] equals self.num_kernels
            for output_channel in range(self.num_kernels):  # all different kernels
                channel_conv_out = [signal.correlate(
                    input_tensor[batch, x],
                    self.weights[output_channel, x],
                    mode='same'
                ) for x in range(self.weights.shape[1])]

                iterate_regions = np.sum(np.stack(channel_conv_out, axis=0), axis=0)

                if self.dim == 1:  # 1D Case
                    iterate_regions = iterate_regions[::self.stride_row]
                else:  # 2D Case
                    iterate_regions = iterate_regions[::self.stride_row, ::self.stride_col]

                output_tensor[batch, output_channel] = iterate_regions + self.bias[output_channel]

        return output_tensor  # [b, h, y, x]

    def backward(self, error_tensor):
        out_error_tensor = np.zeros_like(self.prev_input_tensor)
        tmp_tensor = self.weights.copy()
        batch_size = self.prev_input_tensor.shape[0]
        output_channel = self.prev_input_tensor.shape[1]
        channel_conv_out = []
        iterate_regions_out = []

        if self.dim == 2:
            temp_gradient_weights = np.zeros((error_tensor.shape[0],) + self.weights.shape)
            for batch in range(batch_size):
                channel_conv_out.clear()
                for out_ch in range(output_channel):
                    channel_conv_out.append(
                        np.pad(self.prev_input_tensor[batch, out_ch],
                               ((self.convolution_shape[1] // 2, self.convolution_shape[1] // 2),
                                (self.convolution_shape[2] // 2, self.convolution_shape[2] // 2)))
                    )
                    if not self.convolution_shape[2] % 2:
                        channel_conv_out[out_ch] = channel_conv_out[out_ch][:, :-1]
                    if not self.convolution_shape[1] % 2:
                        channel_conv_out[out_ch] = channel_conv_out[out_ch][:-1, :]
                iterate_regions_out.append(np.stack(channel_conv_out))
            padding_in = np.stack(iterate_regions_out)

            for batch in range(error_tensor.shape[0]):
                for out_ch in range(error_tensor.shape[1]):

                    # up-sampling
                    up_sam = signal.resample(
                        error_tensor[batch, out_ch],
                        error_tensor[batch, out_ch].shape[0]
                        * self.stride_shape[0],
                        axis=0
                    )
                    up_sam = signal.resample(
                        up_sam, error_tensor[batch, out_ch].shape[1]
                                * self.stride_shape[1],
                        axis=1
                    )
                    # match the correct shape
                    up_sam = up_sam[:self.prev_input_tensor.shape[2], :self.prev_input_tensor.shape[3]]
                    # zero-interpolation
                    if self.stride_shape[1] > 1:
                        for i, row in enumerate(up_sam):
                            for j, element in enumerate(row):
                                if j % self.stride_shape[1] != 0:
                                    row[j] = 0
                    if self.stride_shape[0] > 1:
                        for i, row in enumerate(up_sam):
                            for j, element in enumerate(row):
                                if i % self.stride_shape[0] != 0:
                                    row[j] = 0

                    # loop over input channels
                    for ch in range(self.prev_input_tensor.shape[1]):
                        temp_gradient_weights[batch, out_ch, ch] = signal.correlate(
                            padding_in[batch, ch], up_sam,
                            mode='valid'
                        )
            self.gradient_weights = temp_gradient_weights.sum(axis=0)

        if len(self.convolution_shape) == 3:
            tmp_tensor = np.transpose(tmp_tensor, (1, 0, 2, 3))
        elif len(self.convolution_shape) == 2:
            tmp_tensor = np.transpose(tmp_tensor, (1, 0, 2))

        for batch in range(error_tensor.shape[0]):
            for out_ch in range(tmp_tensor.shape[0]):
                channel_conv_out = []
                for ch in range(tmp_tensor.shape[1]):
                    if self.dim == 2:
                        up_sam2 = signal.resample(
                            error_tensor[batch, ch],
                            error_tensor[batch, ch].shape[0]
                            * self.stride_row,
                            axis=0
                        )
                        up_sam2 = signal.resample(
                            up_sam2,
                            error_tensor[batch, ch].shape[1]
                            * self.stride_shape[1],
                            axis=1
                        )
                        up_sam2 = up_sam2[:self.prev_input_tensor.shape[2], :self.prev_input_tensor.shape[3]]
                        if self.stride_shape[1] > 1:
                            for i, row in enumerate(up_sam2):
                                for j, element in enumerate(row):
                                    if j % self.stride_shape[1] != 0:
                                        row[j] = 0
                        if self.stride_shape[0] > 1:
                            for i, row in enumerate(up_sam2):
                                for j, element in enumerate(row):
                                    if j % self.stride_shape[0] != 0:
                                        row[j] = 0

                    elif self.dim == 1:
                        up_sam2 = signal.resample(
                            error_tensor[batch, ch],
                            error_tensor[batch, ch].shape[0]
                            * self.stride_shape[0],
                            axis=0
                        )
                        up_sam2 = up_sam2[:self.prev_input_tensor.shape[2]]
                        if self.stride_shape[0] > 1:
                            for i, element in enumerate(up_sam2):
                                if i % self.stride_shape[0] != 0:
                                    up_sam2[i] = 0

                    channel_conv_out.append(
                        signal.convolve(up_sam2, tmp_tensor[out_ch, ch], mode='same', method='direct'))

                out_error_tensor[batch, out_ch] = np.sum(np.stack(channel_conv_out, axis=0), axis=0)
        if self.dim == 2:
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))
        else:
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2))

        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        return out_error_tensor

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = np.prod(self.weights.shape[1:4])  #
        fan_out = np.prod((self.num_kernels,) + self.weights.shape[2:4])
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, 1, self.weights.shape[0])

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value
