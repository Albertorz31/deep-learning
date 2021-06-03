import numpy as np
from Base import BaseLayer


class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.trainable = False
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.pooling_col = self.pooling_shape[0]
        self.pooling_row = self.pooling_shape[1]

    def forward(self, input_tensor):
        self.prev_input_tensor = input_tensor
        batch_size, channel, height, width = input_tensor.shape

        kernel_height = int(np.floor((height - self.pooling_col) / self.pooling_col) + 1)
        kernel_width = int(np.floor((width - self.pooling_row) / self.pooling_row[1]) + 1)

        self.pool_output = np.zeros((batch_size, channel, kernel_height, kernel_width))
        self.pool_idx = np.zeros((batch_size, channel, kernel_height, kernel_width))

        for batch in range(batch_size):
            for ch in range(channel):
                #  Pooling
                img = input_tensor[batch, ch, :]
                im_row, im_col = img.shape
                max_pool_row = im_row - self.pooling_col + 1
                max_pool_col = im_col - self.pooling_col + 1
                max_pool = np.zeros((max_pool_row, max_pool_col))
                max_pool_idx = np.zeros((max_pool_row, max_pool_col))

                for x in np.arange(max_pool_row):
                    for y in np.arange(max_pool_col):
                        pool_block = img[x:(x + self.pooling_col),
                                     y:(y + self.pooling_col)]
                        max_pool[x, y] = np.max(pool_block)

                        x, y = (np.argmax(pool_block) // self.pooling_col), \
                               (np.argmax(pool_block) % self.pooling_col)
                        img_row_index = x + x
                        img_col_index = y + y
                        max_pool_idx[x, y] = img_row_index * im_col + img_col_index

                # Striding
                pool_row_stride = int(np.ceil(max_pool_row / self.stride_shape[0]))
                pool_col_stride = int(np.ceil(max_pool_col / self.stride_shape[1]))
                max_pool_stride = np.zeros((pool_row_stride, pool_col_stride))
                max_index_stride = np.zeros((pool_row_stride, pool_col_stride))

                for r in range(pool_row_stride):
                    for c in range(pool_col_stride):
                        max_pool_stride[r, c] = max_pool[r * self.stride_shape[0], c * self.stride_shape[1]]
                        max_index_stride[r, c] = max_pool_idx[r * self.stride_shape[0], c * self.stride_shape[1]]

                self.pool_output[batch, ch, :] = max_pool_stride
                self.pool_idx[batch, ch, :] = max_index_stride
        return self.pool_output

    def backward(self, error_tensor):

        error_out = np.zeros(np.shape(self.prev_input_tensor))
        batch_size, channel, height, img_width = self.pool_output.shape

        self.error_tensor = error_tensor.reshape(self.pool_output.shape)

        for batch in range(batch_size):
            for ch in range(channel):
                for ht in range(height):
                    for wdt in range(img_width):
                        idx_x = int(np.floor(self.pool_idx[batch, ch, ht, wdt] / self.prev_input_tensor.shape[3]))
                        idx_y = int(np.mod(self.pool_idx[batch, ch, ht, wdt], self.prev_input_tensor.shape[3]))
                        error_out[batch, ch, idx_x, idx_y] += self.error_tensor[batch, ch, ht, wdt]
        return error_out
