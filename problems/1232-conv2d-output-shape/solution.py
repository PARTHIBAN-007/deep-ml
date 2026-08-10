def conv_out_shape(h, w, kernel, stride, padding):
    height = (h + 2*padding - kernel )/stride + 1
    width = (w+ 2*padding - kernel)/stride + 1
    return (int(height),int(width))