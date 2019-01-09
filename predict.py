import glob
import time
import os
import cv2
import numpy as np

from keras.applications import vgg16
from keras.models import Model,Sequential
from keras.layers import Conv2D, Conv2DTranspose, Input, Cropping2D, add, Dropout, Reshape, Activation

from keras.models import load_model

def FCN8_helper(nClasses,  input_height, input_width):

    img_input = Input(shape=(input_height, input_width, 3))

    '''
    include_top：是否保留顶层的3个全连接网络
    weights：None代表随机初始化，即不加载预训练权重。'imagenet'代表加载预训练权重
    input_tensor：可填入Keras tensor作为模型的图像输出tensor
    input_shape：可选，仅当include_top=False有效，应为长为3的tuple，指明输入图片的shape，图片的宽高必须大于48，如(200,200,3)
    '''
    model = vgg16.VGG16(
        include_top=False,
        weights='imagenet',input_tensor=img_input,
        pooling=None,
        classes=1000)
    assert isinstance(model,Model)

    o = Conv2D(filters=4096, kernel_size=(7, 7), padding="same", activation="relu", name="fc6")(model.output)
    o = Dropout(rate=0.5)(o)
    o = Conv2D(filters=4096, kernel_size=(1, 1), padding="same", activation="relu", name="fc7")(o)
    o = Dropout(rate=0.5)(o)

    o = Conv2D(filters=nClasses, kernel_size=(1, 1), padding="same", activation="relu", kernel_initializer="he_normal",
               name="score_fr")(o)

    o = Conv2DTranspose(filters=nClasses, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation=None,
                        name="score2")(o)

    fcn8 = Model(inputs=img_input, outputs=o)
    # mymodel.summary()
    return fcn8

def FCN8(nClasses, input_height, input_width):

    fcn8=FCN8_helper(nClasses, input_height, input_width)

    # Conv to be applied on Pool4
    skip_con1 = Conv2D(nClasses, kernel_size=(1, 1), padding="same", activation=None,kernel_initializer="he_normal",
                       name="score_pool4")( fcn8.get_layer("block4_pool").output)
    Summed = add(inputs=[skip_con1, fcn8.output])

    x = Conv2DTranspose(nClasses, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation=None,
                        name="score4")(Summed)

    ###
    skip_con2 = Conv2D(nClasses, kernel_size=(1, 1), padding="same", activation=None,kernel_initializer="he_normal",
                       name="score_pool3")( fcn8.get_layer("block3_pool").output)
    Summed2 = add(inputs=[skip_con2, x])

    #####
    Up = Conv2DTranspose(nClasses, kernel_size=(8, 8), strides=(8, 8),
                         padding="valid", activation=None, name="upsample")(Summed2)

    Up = Reshape((-1, nClasses))(Up)
    Up = Activation("softmax")(Up)

    mymodel=Model(inputs=fcn8.input,outputs=Up)

    return mymodel

def FCN32(nClasses, input_height, input_width):

    img_input = Input(shape=( input_height, input_width,3))

    model = vgg16.VGG16(
        include_top=False,
        weights='imagenet',input_tensor=img_input,
        pooling=None,
        classes=1000)
    assert isinstance(model,Model)

    o=Conv2D(filters=4096,kernel_size=(7,7),padding="same",activation="relu",name="fc6")(model.output)
    o=Dropout(rate=0.5)(o)

    o = Conv2D(filters=4096, kernel_size=(1, 1), padding="same", activation="relu", name="fc7")(o)
    o=Dropout(rate=0.5)(o)


    o = Conv2D(filters=nClasses, kernel_size=(1,1), padding="same",activation="relu",kernel_initializer="he_normal",
               name="score_fr")(o)

    o=Conv2DTranspose(filters=nClasses,kernel_size=(32,32),strides=(32,32),padding="valid",activation=None,
                      name="score2")(o)


    o=Reshape((-1,nClasses))(o)
    o=Activation("softmax")(o)

    fcn8=Model(inputs=img_input,outputs=o)
    # mymodel.summary()
    return fcn8

def mkdir_s(path: str):
    """Create directory in specified path, if not exists."""
    if not os.path.exists(path):
        os.makedirs(path)

def normalize_in(img: np.array) -> np.array:
    """"""
    img = img.astype(np.float32)
    img /= 256.0
    img -= 0.5
    return img


def normalize_gt(img: np.array) -> np.array:
    """"""
    img = img.astype(np.float32)
    img /= 255.0
    return img


def add_border(img: np.array, size_x: int = 224, size_y: int = 224) -> (np.array, int, int):
    """Add border to image, so it will divide window sizes: size_x and size_y"""
    max_y, max_x = img.shape[:2]
    print(str(max_y)+"&"+str(max_x))
    border_y = 0
    if max_y % size_y != 0:
        print('1.'+str(size_y - (max_y % size_y)))
        border_y = (size_y - (max_y % size_y) + 1) // 2
        print('border_y='+str(border_y))
        img = cv2.copyMakeBorder(img, border_y, border_y, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    border_x = 0
    if max_x % size_x != 0:
        print('2.'+str(size_x - (max_x % size_x)))
        border_x = (size_x - (max_x % size_x) + 1) // 2
        print('border_x=' + str(border_x))
        img = cv2.copyMakeBorder(img, 0, 0, border_x, border_x, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return img, border_y, border_x


def split_img(img: np.array, size_x: int = 224, size_y: int = 224) -> [np.array]:
    """Split image to parts (little images).

    Walk through the whole image by the window of size size_x * size_y without overlays and
    save all parts in list. Images sizes should divide window sizes.

    """
    max_y, max_x = img.shape[:2]
    print('split'+str(max_y)+"^^"+str(max_x))
    parts = []
    curr_y = 0
    # TODO: rewrite with generators.
    while (curr_y + size_y) <= max_y:
        curr_x = 0
        while (curr_x + size_x) <= max_x:
            parts.append(img[curr_y:curr_y + size_y, curr_x:curr_x + size_x])
            curr_x += size_x
        curr_y += size_y
    print(parts)
    return parts

def combine_imgs(imgs: [np.array], max_y: int, max_x: int) -> np.array:
    """Combine image parts to one big image.

    Walk through list of images and create from them one big image with sizes max_x * max_y.
    If border_x and border_y are non-zero, they will be removed from created image.
    The list of images should contain data in the following order:
    from left to right, from top to bottom.

    """
    img = np.zeros((max_y, max_x), np.float)
    size_y, size_x = imgs[0].shape
    curr_y = 0
    i = 0
    # TODO: rewrite with generators.
    while (curr_y + size_y) <= max_y:
        curr_x = 0
        while (curr_x + size_x) <= max_x:
            try:
                img[curr_y:curr_y + size_y, curr_x:curr_x + size_x] = imgs[i]
            except:
                i -= 1
            i += 1
            curr_x += size_x
        curr_y += size_y
    return img

def postprocess_img(img: np.array) -> np.array:
    """Apply Otsu threshold to image."""
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img

def preprocess_img(img: np.array) -> np.array:
    """Apply bilateral filter to image."""
    #img = cv2.bilateralFilter(img, 5, 50, 50) TODO: change parameters.
    return img


def process_unet_img(img: np.array, model, batchsize: int = 20) -> np.array:
    """Split image to 128x128 parts and run U-net for every part."""
    print(img.shape)
    img, border_y, border_x = add_border(img)
    print(img.shape)
    print(border_y)
    print(border_x)
    img = normalize_in(img)
    parts = split_img(img)
    print(len(parts))
    parts = np.array(parts)
    parts.shape = (parts.shape[0], parts.shape[1], parts.shape[2], 3)
    parts = model.predict(parts, batchsize)
    tmp = []
    for part in parts:
        part.shape = (224, 224)
        tmp.append(part)
    parts = tmp
    img = combine_imgs(parts, img.shape[0], img.shape[1])
    img = img[border_y:img.shape[0] - border_y, border_x:img.shape[1] - border_x]
    img = img * 255.0
    img = img.astype(np.uint8)
    return img

def binarize_img(img: np.array, model, batchsize: int = 20) -> np.array:
    """Binarize image, using U-net, Otsu, bottom-hat transform etc."""
    img = preprocess_img(img)
    print(img)
    img = process_unet_img(img, model, batchsize)
    img = postprocess_img(img)
    return img

def main():
    start_time = time.time()

    fnames_in = list(glob.iglob(os.path.join('test', '**', '*_in.*'), recursive=True))
    model = None
    output = 'output'
    if len(fnames_in) != 0:
        mkdir_s(output)
        model = FCN8(2,224, 224)
        model.compile(loss='categorical_crossentropy',optimizer="adadelta",metrics=['acc'])
        model = load_model('output/segnet_model.h5')
    for fname in fnames_in:
        print(fname)
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        print(img.shape)
        img = binarize_img(img, model, 20)
        cv2.imwrite(os.path.join('result', os.path.split(fname)[-1].replace('_in', '_out')), img)

    print("finished in {0:.2f} seconds".format(time.time() - start_time))


if __name__ == "__main__":
    main()