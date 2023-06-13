import dask.array as da
import numpy as np

# array shape related helper function for 2d data and dask array 
def _get_padding(x:da.Array, stride:int=32):
    # new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
    # new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
    h, w = x.shape[-2:]
    s = stride
    w_padding = int((s-w%s)/2) if w%s >0 else 0
    h_padding = int((s-h%s)/2) if h%s >0 else 0
    w_padding_offset = int((s-w%s) % 2)
    h_padding_offset = int((s-h%s) % 2)
    padding = [(0,0),] * x.ndim
    padding[-1] = (w_padding, w_padding+w_padding_offset)
    padding[-2] = (h_padding, h_padding+h_padding_offset)
    return padding

def convert_tensor_shape(x:da.Array):
    # make sure image has four dimentions (b,c,w,h)
    while len(x.shape) < 4:
        x = da.expand_dims(x, 0)

    # make sure that iteration axis is always at 0th pos
    # eg. if data is t,h,w then it will be t, c, h, w. where t is time
    # eg. if data is z,h,w then it will be z, c, h, w. where z is time or iteration axis
    return da.transpose(x, (1,0,2,3))

def revert_tensor_shape(x:da.Array):
    return da.squeeze(x)

def reshape_input_data(input_data:da.Array):
    # Unet requires the data to be of shape divisible by 32
    padding = _get_padding(input_data, 32)
    return da.pad(input_data, padding, 'constant')

def reshape_output_data(input_data:da.Array, ouput_data:da.Array):
    padding = _get_padding(input_data, 32)
    w_pad = padding[-1]
    h_pad = padding[-2]
    slices = [[0,0],] * ouput_data.ndim
    slices[-1] = [w_pad[0], -w_pad[1]]
    slices[-2] = [h_pad[0], -h_pad[1]]
    if ouput_data.ndim == 2:
        # h & w
        return ouput_data[slices[-2][0]:slices[-2][1], slices[-1][0]:slices[-1][1]]
    elif ouput_data.ndim == 3:
        # z/c, h & w
        return ouput_data[:, slices[-2][0]:slices[-2][1], slices[-1][0]:slices[-1][1]]
    else:
        # t/b, z/c, h & w
        return ouput_data[:, :, slices[-2][0]:slices[-2][1], slices[-1][0]:slices[-1][1]]


def test_shapes():
    img = np.ones((10, 138,181))
    print("img", img.shape)
    
    padded_img = reshape_input_data(img)
    print("padded_img", padded_img.shape)
    
    unpadded_img = reshape_output_data(img, padded_img)
    print("unpadded_img", unpadded_img.shape)

    tensor_shape_img = convert_tensor_shape(padded_img)
    print("tensore_shape_img", tensor_shape_img.shape)

    revert_tensor_shape = revert_tensor_shape(padded_img)
    print("revert_tensor_shape", revert_tensor_shape.shape)

if __name__ == "__main__":
    test()
