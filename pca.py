from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt

img_filename = 'lena.png'  # replace with the image you like
img_format = img_filename.split('.')[1]

img = plt.imread(img_filename)
img_height = img.shape[0]
img_width = img.shape[1]


def rgb_to_gray(img):
    # multiply rgb to gray matrix
    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])


gray_img = rgb_to_gray(img)


def pca_compress(k):
    pca = PCA(n_components=k)
    p = pca.fit_transform(gray_img)
    return pca.inverse_transform(p)


def add_image(fig, img, pos, k):
    ax = fig.add_subplot(2, 3, pos)  # add image to 2*3 figure
    if k == 512:
        ax.set_title('original, k = 512')
    else:
        ax.set_title(f'Compressed Image, k = {k}')

    if (img_format == 'jpg') or (img_format == 'jpeg'):
        # imshow only supports int 0-255 or float 0-1
        plt.imshow(img.astype('uint8'), cmap=plt.get_cmap('gray'))
    elif img_format == 'png':
        plt.imshow(img, cmap=plt.get_cmap('gray'))


fig = plt.figure(num='PCA', figsize=(10, 8))
add_image(fig, gray_img, 1, 512)

ks = [256, 128, 64, 32, 16]
for i, k in enumerate(ks):
    add_image(fig, pca_compress(k), i+2, k)
plt.show()
