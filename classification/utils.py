def conv_calculator(image_s, conv_s, padding, stride):
    return int((image_s + 2 * padding - conv_s)/stride) + 1
