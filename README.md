# computer_vision_algorithms
algorithms used in introduction to computer vision


## Smoothing Images

### 2d Covolution( Image Filtering )
* http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('opencv_logo.png')

kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(img,-1,kernel)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()
```

## Image Blurring(Image Smoothing)
* http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html

### Averaging
```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('opencv_logo.png')

blur = cv2.blur(img,(5,5))

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()
```

### Gaussian Filtering
```python
blur = cv2.GaussianBlur(img,(5,5),0)
```

### Median Filtering
```python
median = cv2.medianBlur(img,5)
```
### Bilateral Filtering
```python
blur = cv2.bilateralFilter(img,9,75,75)
```

## Python code for Gaussian noise
* http://www.magikcode.com/?p=240

```python
import cv2
import numpy as np

def show_image_and_wait(title, image):
    # Display the image in a window.  Window size fits image.
    cv2.imshow(title, image)

    # Wait for user input; click X to destroy window.
    cv2.waitKey(0)

    # Destroy window and return to caller.
    cv2.destroyAllWindows()

def add_gaussian_noise(image_in, noise_sigma):
    temp_image = np.float64(np.copy(image_in))

    h = temp_image.shape[0]
    w = temp_image.shape[1]
    noise = np.random.randn(h, w) * noise_sigma

    noisy_image = np.zeros(temp_image.shape, np.float64)
    if len(temp_image.shape) == 2:
        noisy_image = temp_image + noise
    else:
        noisy_image[:,:,0] = temp_image[:,:,0] + noise
        noisy_image[:,:,1] = temp_image[:,:,1] + noise
        noisy_image[:,:,2] = temp_image[:,:,2] + noise

    """
    print('min,max = ', np.min(noisy_image), np.max(noisy_image))
    print('type = ', type(noisy_image[0][0][0]))
    """

    return noisy_image

def convert_to_uint8(image_in):
    temp_image = np.float64(np.copy(image_in))
    cv2.normalize(temp_image, temp_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)

    return temp_image.astype(np.uint8)

def main():
    girl_face_filename = "girl_face_closeup.jpg"
    print('opening image: ', girl_face_filename)

    # cv2.IMREAD_COLOR - read in color images (BGR)
    # cv2.IMREAD_GRAYSCALE - convert image to grayscale
    girl_face_image = cv2.imread(girl_face_filename, cv2.IMREAD_UNCHANGED)
    girl_face_grayscale_image = cv2.cvtColor(girl_face_image, cv2.COLOR_BGR2GRAY)

    """
    Gaussian is a nice noise function.  Gaussian noise are values generated from the
    random normal distribution.  The mean of the distribution is 0 and the standard
    deviation is 1.  The standard deviation is a measure of how spread out the values
    are from the mean or 0.  randn() generates random numbers from this distribution.
    The Gaussian distribution is symmetric about the mean of the probability.

    Sigma determines the magnitude of the noise function.  For a small sigma, the noise
    function produces values very close to zero or a gray image since we want to map the
    pixel with a value of zero to gray.  The larger sigma spreads out the noise.
    Multiplying an image by a noise image generated from a Gaussian function effectively
    changes the standard deviation of the pixel values.  This is how far apart the pixel
    colors are in value.
    """
    noisy_sigma = 35
    noisy_image = add_gaussian_noise(girl_face_grayscale_image, noisy_sigma)

    print('noisy image shape: {0}, len of shape {1}'.format(\
        girl_face_image.shape, len(noisy_image.shape)))
    print('    WxH: {0}x{1}'.format(noisy_image.shape[1], noisy_image.shape[0]))
    print('    image size: {0} bytes'.format(noisy_image.size))

    show_image_and_wait(girl_face_filename, convert_to_uint8(noisy_image))
    noisy_filename = 'girl_face_noise_' + str(noisy_sigma) + '.jpg'
    cv2.imwrite(noisy_filename, noisy_image)

if __name__ == "__main__":
    main()
```

## applying a Median Filter to Remove Salt & Pepper Noise
```python
# imports
import numpy as np
import cv2
import matplotlib.pyplot as plt

# load in image and add Salt and pepper noise
moon = cv2.imread('images/moon.png', 0)

######################################################## ADD SALT & PEPPER NOISE
# salt and peppering manually (randomly assign coords as either white or black)
rows, cols = moon.shape
salt_vs_pepper_ratio = 0.5
amount = 0.01
moon_salted_and_peppered = moon.copy()
num_salt = np.ceil(amount * moon.size * salt_vs_pepper_ratio)
coords = [np.random.randint(0, i - 1, int(num_salt)) for i in moon.shape]
moon_salted_and_peppered[coords] = 255
num_pepper = np.ceil(amount * moon.size * (1 - salt_vs_pepper_ratio))
coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in moon.shape]
moon_salted_and_peppered[coords] = 0

############################################ APPLY MEDIAN FILTER TO REMOVE NOISE
# The second argument is the aperture linear size; it must be odd and greater
# than 1, for example: 3, 5, 7
moon_median = cv2.medianBlur(moon, 3)

# show all three images using Matplotlib
plt.figure(figsize=(15, 6))
plt.subplot(1, 3, 1)
plt.imshow(moon, cmap='gray'), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 2)
plt.imshow(moon_salted_and_peppered, cmap='gray')
plt.title('Salted & Peppered'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 3)
plt.imshow(moon_median, cmap='gray'), plt.title('Median Blur on S&P')
plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
```

## correlation filtering and its flipped version called convolution
```python
"""Apply crosscorrelation and convolution to an image."""
from __future__ import division, print_function
from scipy import signal
import numpy as np
import random
from skimage import data
import util
np.random.seed(42)
random.seed(42)

def main():
    """Initialize kernel, apply it to an image (via crosscorrelation, convolution)."""
    img = data.camera()
    kernel = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    cc_response = crosscorrelate(img, kernel)
    cc_gt = signal.correlate(img, kernel, mode="same")

    conv_response = convolve(img, kernel)
    conv_gt = signal.convolve(img, kernel, mode="same")

    util.plot_images_grayscale(
        [img, cc_response, cc_gt, conv_response, conv_gt],
        ["Image", "Cross-Correlation", "Cross-Correlation (Ground Truth)", "Convolution", "Convolution (Ground Truth)"]
    )

def crosscorrelate(img, kernel):
    """Apply a kernel/filter via crosscorrelation to an image.
    Args:
        img The image
        kernel The kernel/filter to apply
    Returns:
        New image
    """
    imheight, imwidth = img.shape
    kheight, kwidth = kernel.shape
    assert len(img.shape) == 2
    assert kheight == kwidth # only square matrices
    assert kheight % 2 != 0 # sizes must be odd
    ksize = int((kheight - 1) / 2)
    im_pad = np.pad(img, ((ksize, ksize), (ksize, ksize)), mode="constant")
    response = np.zeros(img.shape)
    for y in range(ksize, ksize+imheight):
        for x in range(ksize, ksize+imwidth):
            patch = im_pad[y-ksize:y+ksize+1, x-ksize:x-ksize+1]
            corr = np.sum(patch * kernel)
            response[y-ksize, x-ksize] = corr
    return response

def convolve(img, kernel):
    """Apply a kernel/filter via convolution to an image.
    Args:
        img The image
        kernel The kernel/filter to apply
    Returns:
        New image
    """
    return crosscorrelate(img, np.flipud(np.fliplr(kernel)))

if __name__ == "__main__":
    main()
```

## hough transform: lines
* http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
```python
import cv2
import numpy as np

img = cv2.imread('dave.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)

lines = cv2.HoughLines(edges,1,np.pi/180,200)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imwrite('houghlines3.jpg',img)
```

## Progressive Probabilistic Hough Transform 
* http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
```python
import cv2
import numpy as np

img = cv2.imread('dave.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
minLineLength = 100
maxLineGap = 10
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imwrite('houghlines5.jpg',img)
```

## hough transform
```python
"""Apply crosscorrelation and convolution to an image."""
from __future__ import division, print_function
from scipy import signal
import numpy as np
import random
from skimage import data
import util
np.random.seed(42)
random.seed(42)

def main():
    """Initialize kernel, apply it to an image (via crosscorrelation, convolution)."""
    img = data.camera()
    kernel = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    cc_response = crosscorrelate(img, kernel)
    cc_gt = signal.correlate(img, kernel, mode="same")

    conv_response = convolve(img, kernel)
    conv_gt = signal.convolve(img, kernel, mode="same")

    util.plot_images_grayscale(
        [img, cc_response, cc_gt, conv_response, conv_gt],
        ["Image", "Cross-Correlation", "Cross-Correlation (Ground Truth)", "Convolution", "Convolution (Ground Truth)"]
    )

def crosscorrelate(img, kernel):
    """Apply a kernel/filter via crosscorrelation to an image.
    Args:
        img The image
        kernel The kernel/filter to apply
    Returns:
        New image
    """
    imheight, imwidth = img.shape
    kheight, kwidth = kernel.shape
    assert len(img.shape) == 2
    assert kheight == kwidth # only square matrices
    assert kheight % 2 != 0 # sizes must be odd
    ksize = int((kheight - 1) / 2)
    im_pad = np.pad(img, ((ksize, ksize), (ksize, ksize)), mode="constant")
    response = np.zeros(img.shape)
    for y in range(ksize, ksize+imheight):
        for x in range(ksize, ksize+imwidth):
            patch = im_pad[y-ksize:y+ksize+1, x-ksize:x-ksize+1]
            corr = np.sum(patch * kernel)
            response[y-ksize, x-ksize] = corr
    return response

def convolve(img, kernel):
    """Apply a kernel/filter via convolution to an image.
    Args:
        img The image
        kernel The kernel/filter to apply
    Returns:
        New image
    """
    return crosscorrelate(img, np.flipud(np.fliplr(kernel)))

if __name__ == "__main__":
    main()
```

###  horizontal Sobel
* http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html
```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('box.png',0)

# Output dtype = cv2.CV_8U
sobelx8u = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)

# Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)

plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])

plt.show()
```
## Canny edge dector
* http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html
``` python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('messi5.jpg',0)
edges = cv2.Canny(img,100,200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()

```

## Fourier Transform in Numpy
* http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html
```python
 import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('messi5.jpg',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
```

## attached to Fouier Transform in numpy
* High Pass Filtering is an edge detection operation
```python
ows, cols = img.shape
crow,ccol = rows/2 , cols/2
fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

plt.subplot(131),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(img_back, cmap = 'gray')
plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(img_back)
plt.title('Result in JET'), plt.xticks([]), plt.yticks([])

plt.show()
```

## Fourier Transform in OpenCV
* http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html

```python
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('messi5.jpg',0)

dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
```

* how to remove high frequency contents in the image
```python
rows, cols = img.shape
crow,ccol = rows/2 , cols/2

# create a mask first, center square is 1, remaining all zeros
mask = np.zeros((rows,cols,2),np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1

# apply mask and inverse DFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
```

# Gaussians for antialiasing when downsampling images
* https://github.com/todddangerfarr/intro-to-computer-vision-with-python-opencv/blob/master/23_aliasing_in_images.py
```python
    # Aliasing in images occurs when the frequency is higher than the number of
# pixels we have to represent that frequency in the image.  This results in
# a misidentification of a signal frequency which introduces distortion or error

# In particular this Python file shows how Guassian filtering can be used to
# help solve this problem by removing the high frequencies prior to downsampling

# imports
import numpy as np
import cv2
import matplotlib.pyplot as plt

# read in and show the image
img = cv2.imread('images/van_gogh.png')
cv2.imshow('Van Gogh Original', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# First create a downsampling fucntion for the image
def downsample(img, reduction_step=2):
    height, width, channels = img.shape
    rows_keep = range(0, height, reduction_step)
    cols_keep = range(0, width, reduction_step)

    ds_img = np.zeros((len(rows_keep), len(cols_keep), channels), dtype=np.uint8)
    for i, row_value in enumerate(rows_keep):
        for j, col_value in enumerate(cols_keep):
            for k in range(channels):
                ds_img[i, j, k] = img[row_value, col_value, k]

    return ds_img

# show images 
vg_half_size = downsample(img, 2)
vg_quarter_size = downsample(img, 4)
cv2.imshow('1/2 Size', vg_half_size)
cv2.imshow('1/4 Size', vg_quarter_size)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

# From scipy/scipy/ndimage/filters.py
* https://github.com/scipy/scipy/blob/v0.15.1/scipy/ndimage/filters.py#L251

```python
from __future__ import division, print_function, absolute_import

import math
import numpy
from . import _ni_support
from . import _nd_image
from scipy.misc import doccer
from scipy.lib._version import NumpyVersion
```

##correlate1d
```python
def correlate1d(input, weights, axis=-1, output=None, mode="reflect",
                cval=0.0, origin=0):
    """Calculate a one-dimensional correlation along the given axis.
    The lines of the array along the given axis are correlated with the
    given weights.
    Parameters
    ----------
    %(input)s
    weights : array
        One-dimensional sequence of numbers.
    %(axis)s
    %(output)s
    %(mode)s
    %(cval)s
    %(origin)s
    """
    input = numpy.asarray(input)
    if numpy.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    output, return_value = _ni_support._get_output(output, input)
    weights = numpy.asarray(weights, dtype=numpy.float64)
    if weights.ndim != 1 or weights.shape[0] < 1:
        raise RuntimeError('no filter weights given')
    if not weights.flags.contiguous:
        weights = weights.copy()
    axis = _ni_support._check_axis(axis, input.ndim)
    if (len(weights) // 2 + origin < 0) or (len(weights) // 2 +
                                            origin > len(weights)):
        raise ValueError('invalid origin')
    mode = _ni_support._extend_mode_to_code(mode)
    _nd_image.correlate1d(input, weights, axis, output, mode, cval,
                          origin)
    return return_value
```

## convolve1d
```python
def convolve1d(input, weights, axis=-1, output=None, mode="reflect",
               cval=0.0, origin=0):
    """Calculate a one-dimensional convolution along the given axis.
    The lines of the array along the given axis are convolved with the
    given weights.
    Parameters
    ----------
    %(input)s
    weights : ndarray
        One-dimensional sequence of numbers.
    %(axis)s
    %(output)s
    %(mode)s
    %(cval)s
    %(origin)s
    Returns
    -------
    convolve1d : ndarray
        Convolved array with same shape as input
    """
    weights = weights[::-1]
    origin = -origin
    if not len(weights) & 1:
        origin -= 1
    return correlate1d(input, weights, axis, output, mode, cval, origin)
```

## gaussian_filter1d
```python
def gaussian_filter1d(input, sigma, axis=-1, order=0, output=None,
                      mode="reflect", cval=0.0, truncate=4.0):
    """One-dimensional Gaussian filter.
    Parameters
    ----------
    %(input)s
    sigma : scalar
        standard deviation for Gaussian kernel
    %(axis)s
    order : {0, 1, 2, 3}, optional
        An order of 0 corresponds to convolution with a Gaussian
        kernel. An order of 1, 2, or 3 corresponds to convolution with
        the first, second or third derivatives of a Gaussian. Higher
        order derivatives are not implemented
    %(output)s
    %(mode)s
    %(cval)s
    truncate : float
        Truncate the filter at this many standard deviations.
        Default is 4.0.
    Returns
    -------
    gaussian_filter1d : ndarray
    """
    if order not in range(4):
        raise ValueError('Order outside 0..3 not implemented')
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd = sd * sd
    # calculate the kernel:
    for ii in range(1, lw + 1):
        tmp = math.exp(-0.5 * float(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    # implement first, second and third order derivatives:
    if order == 1:  # first derivative
        weights[lw] = 0.0
        for ii in range(1, lw + 1):
            x = float(ii)
            tmp = -x / sd * weights[lw + ii]
            weights[lw + ii] = -tmp
            weights[lw - ii] = tmp
    elif order == 2:  # second derivative
        weights[lw] *= -1.0 / sd
        for ii in range(1, lw + 1):
            x = float(ii)
            tmp = (x * x / sd - 1.0) * weights[lw + ii] / sd
            weights[lw + ii] = tmp
            weights[lw - ii] = tmp
    elif order == 3:  # third derivative
        weights[lw] = 0.0
        sd2 = sd * sd
        for ii in range(1, lw + 1):
            x = float(ii)
            tmp = (3.0 - x * x / sd) * x * weights[lw + ii] / sd2
            weights[lw + ii] = -tmp
            weights[lw - ii] = tmp
    return correlate1d(input, weights, axis, output, mode, cval, 0)
```

## gaussian_filter
```python
def gaussian_filter(input, sigma, order=0, output=None,
                  mode="reflect", cval=0.0, truncate=4.0):
    """Multidimensional Gaussian filter.
    Parameters
    ----------
    %(input)s
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel. The standard
        deviations of the Gaussian filter are given for each axis as a
        sequence, or as a single number, in which case it is equal for
        all axes.
    order : {0, 1, 2, 3} or sequence from same set, optional
        The order of the filter along each axis is given as a sequence
        of integers, or as a single number.  An order of 0 corresponds
        to convolution with a Gaussian kernel. An order of 1, 2, or 3
        corresponds to convolution with the first, second or third
        derivatives of a Gaussian. Higher order derivatives are not
        implemented
    %(output)s
    %(mode)s
    %(cval)s
    truncate : float
        Truncate the filter at this many standard deviations.
        Default is 4.0.
    Returns
    -------
    gaussian_filter : ndarray
        Returned array of same shape as `input`.
    Notes
    -----
    The multidimensional filter is implemented as a sequence of
    one-dimensional convolution filters. The intermediate arrays are
    stored in the same data type as the output. Therefore, for output
    types with a limited precision, the results may be imprecise
    because intermediate results may be stored with insufficient
    precision.
    """
    input = numpy.asarray(input)
    output, return_value = _ni_support._get_output(output, input)
    orders = _ni_support._normalize_sequence(order, input.ndim)
    if not set(orders).issubset(set(range(4))):
        raise ValueError('Order outside 0..4 not implemented')
    sigmas = _ni_support._normalize_sequence(sigma, input.ndim)
    axes = list(range(input.ndim))
    axes = [(axes[ii], sigmas[ii], orders[ii])
                        for ii in range(len(axes)) if sigmas[ii] > 1e-15]
    if len(axes) > 0:
        for axis, sigma, order in axes:
            gaussian_filter1d(input, sigma, axis, order, output,
                              mode, cval, truncate)
            input = output
    else:
        output[...] = input[...]
    return return_value
    ```

## prewitt fliter
```python
def prewitt(input, axis=-1, output=None, mode="reflect", cval=0.0):
    """Calculate a Prewitt filter.
    Parameters
    ----------
    %(input)s
    %(axis)s
    %(output)s
    %(mode)s
    %(cval)s
    """
    input = numpy.asarray(input)
    axis = _ni_support._check_axis(axis, input.ndim)
    output, return_value = _ni_support._get_output(output, input)
    correlate1d(input, [-1, 0, 1], axis, output, mode, cval, 0)
    axes = [ii for ii in range(input.ndim) if ii != axis]
    for ii in axes:
        correlate1d(output, [1, 1, 1], ii, output, mode, cval, 0,)
    return return_value
```

## sobel filter
```python
def sobel(input, axis=-1, output=None, mode="reflect", cval=0.0):
    """Calculate a Sobel filter.
    Parameters
    ----------
    %(input)s
    %(axis)s
    %(output)s
    %(mode)s
    %(cval)s
    """
    input = numpy.asarray(input)
    axis = _ni_support._check_axis(axis, input.ndim)
    output, return_value = _ni_support._get_output(output, input)
    correlate1d(input, [-1, 0, 1], axis, output, mode, cval, 0)
    axes = [ii for ii in range(input.ndim) if ii != axis]
    for ii in axes:
        correlate1d(output, [1, 2, 1], ii, output, mode, cval, 0)
    return return_value
```

##b N-dimensional Laplace filter using a provided second derivative function
````python
def generic_laplace(input, derivative2, output=None, mode="reflect",
                    cval=0.0,
                    extra_arguments=(),
                    extra_keywords = None):
    Parameters
    ----------
    %(input)s
    derivative2 : callable
        Callable with the following signature::
            derivative2(input, axis, output, mode, cval,
                        *extra_arguments, **extra_keywords)
        See "extra_arguments, extra_keywords below.
    %(output)s
    %(mode)s
    %(cval)s
    %(extra_keywords)s
    %(extra_arguments)s
  
    if extra_keywords is None:
        extra_keywords = {}
    input = numpy.asarray(input)
    output, return_value = _ni_support._get_output(output, input)
    axes = list(range(input.ndim))
    if len(axes) > 0:
        derivative2(input, axes[0], output, mode, cval,
                    *extra_arguments, **extra_keywords)
        for ii in range(1, len(axes)):
            tmp = derivative2(input, axes[ii], output.dtype, mode, cval,
                              *extra_arguments, **extra_keywords)
            output += tmp
    else:
        output[...] = input[...]
    return return_value
```
    
## Nth lablance filter
```python
def laplace(input, output=None, mode="reflect", cval=0.0):
    """N-dimensional Laplace filter based on approximate second derivatives.
    Parameters
    ----------
    %(input)s
    %(output)s
    %(mode)s
    %(cval)s
    """
    def derivative2(input, axis, output, mode, cval):
        return correlate1d(input, [1, -2, 1], axis, output, mode, cval, 0)
    return generic_laplace(input, derivative2, output, mode, cval)
```

## Multidimensional Laplace filter using gaussian second derivatives
```python
def gaussian_laplace(input, sigma, output=None, mode="reflect",
                     cval=0.0, **kwargs).
    Parameters
    ----------
    %(input)s
    sigma : scalar or sequence of scalars
        The standard deviations of the Gaussian filter are given for
        each axis as a sequence, or as a single number, in which case
        it is equal for all axes.
    %(output)s
    %(mode)s
    %(cval)s
    Extra keyword arguments will be passed to gaussian_filter().
    """
    input = numpy.asarray(input)

    def derivative2(input, axis, output, mode, cval, sigma, **kwargs):
        order = [0] * input.ndim
        order[axis] = 2
        return gaussian_filter(input, sigma, order, output, mode, cval,
                               **kwargs)

    return generic_laplace(input, derivative2, output, mode, cval,
                           extra_arguments=(sigma,),
                           extra_keywords=kwargs)
```

## Gradient magnitude using a provided gradient function
```python
def generic_gradient_magnitude(input, derivative, output=None,
                mode="reflect", cval=0.0,
                extra_arguments=(), extra_keywords = None):
    """Gradient magnitude using a provided gradient function.
    Parameters
    ----------
    %(input)s
    derivative : callable
        Callable with the following signature::
            derivative(input, axis, output, mode, cval,
                       *extra_arguments, **extra_keywords)
        See `extra_arguments`, `extra_keywords` below.
        `derivative` can assume that `input` and `output` are ndarrays.
        Note that the output from `derivative` is modified inplace;
        be careful to copy important inputs before returning them.
    %(output)s
    %(mode)s
    %(cval)s
    %(extra_keywords)s
    %(extra_arguments)s
    """
    if extra_keywords is None:
        extra_keywords = {}
    input = numpy.asarray(input)
    output, return_value = _ni_support._get_output(output, input)
    axes = list(range(input.ndim))
    if len(axes) > 0:
        derivative(input, axes[0], output, mode, cval,
                   *extra_arguments, **extra_keywords)
        numpy.multiply(output, output, output)
        for ii in range(1, len(axes)):
            tmp = derivative(input, axes[ii], output.dtype, mode, cval,
                             *extra_arguments, **extra_keywords)
            numpy.multiply(tmp, tmp, tmp)
            output += tmp
        # This allows the sqrt to work with a different default casting
        if NumpyVersion(numpy.__version__) > '1.6.1':
            numpy.sqrt(output, output, casting='unsafe')
        else:
            numpy.sqrt(output, output)
    else:
        output[...] = input[...]
    return return_value
```

## Multidimensional gradient magnitude using Gaussian derivatives
```python
def gaussian_gradient_magnitude(input, sigma, output=None,
                mode="reflect", cval=0.0, **kwargs):
    """Multidimensional gradient magnitude using Gaussian derivatives.
    Parameters
    ----------
    %(input)s
    sigma : scalar or sequence of scalars
        The standard deviations of the Gaussian filter are given for
        each axis as a sequence, or as a single number, in which case
        it is equal for all axes..
    %(output)s
    %(mode)s
    %(cval)s
    Extra keyword arguments will be passed to gaussian_filter().
    """
    input = numpy.asarray(input)

    def derivative(input, axis, output, mode, cval, sigma, **kwargs):
        order = [0] * input.ndim
        order[axis] = 1
        return gaussian_filter(input, sigma, order, output, mode,
                               cval, **kwargs)

    return generic_gradient_magnitude(input, derivative, output, mode,
                                      cval, extra_arguments=(sigma,),
                                      extra_keywords=kwargs)
   ```                                


## correlate_or_convolve
```python
def _correlate_or_convolve(input, weights, output, mode, cval, origin,
                           convolution):
    input = nUmpy.asarray(input)
    if numpy.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    origins = _ni_support._normalize_sequence(origin, input.ndim)
    weights = numpy.asarray(weights, dtype=numpy.float64)
    wshape = [ii for ii in weights.shape if ii > 0]
    if len(wshape) != input.ndim:
        raise RuntimeError('filter weights array has incorrect shape.')
    if convolution:
        weights = weights[tuple([slice(None, None, -1)] * weights.ndim)]
        for ii in range(len(origins)):
            origins[ii] = -origins[ii]
            if not weights.shape[ii] & 1:
                origins[ii] -= 1
    for origin, lenw in zip(origins, wshape):
        if (lenw // 2 + origin < 0) or (lenw // 2 + origin > lenw):
            raise ValueError('invalid origin')
    if not weights.flags.contiguous:
        weights = weights.copy()
    output, return_value = _ni_support._get_output(output, input)
    mode = _ni_support._extend_mode_to_code(mode)
    _nd_image.correlate(input, weights, output, mode, cval, origins)
    return return_value
```

##  Multi-dimensional correlation.
```python
def correlate(input, weights, output=None, mode='reflect', cval=0.0,
              origin=0):
    """
    Multi-dimensional correlation.
    The array is correlated with the given kernel.
    Parameters
    ----------
    input : array-like
        input array to filter
    weights : ndarray
        array of weights, same number of dimensions as input
    output : array, optional
        The ``output`` parameter passes an array in which to store the
        filter output.
    mode : {'reflect','constant','nearest','mirror', 'wrap'}, optional
        The ``mode`` parameter determines how the array borders are
        handled, where ``cval`` is the value when mode is equal to
        'constant'. Default is 'reflect'
    cval : scalar, optional
        Value to fill past edges of input if ``mode`` is 'constant'. Default
        is 0.0
    origin : scalar, optional
        The ``origin`` parameter controls the placement of the filter.
        Default 0
    See Also
    --------
    convolve : Convolve an image with a kernel.
    """
    return _correlate_or_convolve(input, weights, output, mode, cval,
                                  origin, False)
```

## Multidimensional convolution.
```python
def convolve(input, weights, output=None, mode='reflect', cval=0.0,
             origin=0):
    """
    Multidimensional convolution.
    The array is convolved with the given kernel.
    Parameters
    ----------
    input : array_like
        Input array to filter.
    weights : array_like
        Array of weights, same number of dimensions as input
    output : ndarray, optional
        The `output` parameter passes an array in which to store the
        filter output.
    mode : {'reflect','constant','nearest','mirror', 'wrap'}, optional
        the `mode` parameter determines how the array borders are
        handled. For 'constant' mode, values beyond borders are set to be
        `cval`. Default is 'reflect'.
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0
    origin : array_like, optional
        The `origin` parameter controls the placement of the filter.
        Default is 0.
    Returns
    -------
    result : ndarray
        The result of convolution of `input` with `weights`.
    See Also
    --------
    correlate : Correlate an image with a kernel.
    Notes
    -----
    Each value in result is :math:`C_i = \\sum_j{I_{i+j-k} W_j}`, where
    W is the `weights` kernel,
    j is the n-D spatial index over :math:`W`,
    I is the `input` and k is the coordinate of the center of
    W, specified by `origin` in the input parameters.
    Examples
    --------
    Perhaps the simplest case to understand is ``mode='constant', cval=0.0``,
    because in this case borders (i.e. where the `weights` kernel, centered
    on any one value, extends beyond an edge of `input`.
    >>> a = np.array([[1, 2, 0, 0],
    ....    [5, 3, 0, 4],
    ....    [0, 0, 0, 7],
    ....    [9, 3, 0, 0]])
    >>> k = np.array([[1,1,1],[1,1,0],[1,0,0]])
    >>> from scipy import ndimage
    >>> ndimage.convolve(a, k, mode='constant', cval=0.0)
    array([[11, 10,  7,  4],
           [10,  3, 11, 11],
           [15, 12, 14,  7],
           [12,  3,  7,  0]])
    Setting ``cval=1.0`` is equivalent to padding the outer edge of `input`
    with 1.0's (and then extracting only the original region of the result).
    >>> ndimage.convolve(a, k, mode='constant', cval=1.0)
    array([[13, 11,  8,  7],
           [11,  3, 11, 14],
           [16, 12, 14, 10],
           [15,  6, 10,  5]])
    With ``mode='reflect'`` (the default), outer values are reflected at the
    edge of `input` to fill in missing values.
    >>> b = np.array([[2, 0, 0],
                      [1, 0, 0],
                      [0, 0, 0]])
    >>> k = np.array([[0,1,0],[0,1,0],[0,1,0]])
    >>> ndimage.convolve(b, k, mode='reflect')
    array([[5, 0, 0],
           [3, 0, 0],
           [1, 0, 0]])
    This includes diagonally at the corners.
    >>> k = np.array([[1,0,0],[0,1,0],[0,0,1]])
    >>> ndimage.convolve(b, k)
    array([[4, 2, 0],
           [3, 2, 0],
           [1, 1, 0]])
    With ``mode='nearest'``, the single nearest value in to an edge in
    `input` is repeated as many times as needed to match the overlapping
    `weights`.
    >>> c = np.array([[2, 0, 1],
                      [1, 0, 0],
                      [0, 0, 0]])
    >>> k = np.array([[0, 1, 0],
                      [0, 1, 0],
                      [0, 1, 0],
                      [0, 1, 0],
                      [0, 1, 0]])
    >>> ndimage.convolve(c, k, mode='nearest')
    array([[7, 0, 3],
           [5, 0, 2],
           [3, 0, 1]])
    """
    return _correlate_or_convolve(input, weights, output, mode, cval,
                                  origin, True)
    ```


## Calculate a one-dimensional uniform filter along the given axis.
```python
def uniform_filter1d(input, size, axis=-1, output=None,
                     mode="reflect", cval=0.0, origin=0):
    """
    Calculate a one-dimensional uniform filter along the given axis.
    The lines of the array along the given axis are filtered with a
    uniform filter of given size.
    Parameters
    ----------
    %(input)s
    size : integer
        length of uniform filter
    %(axis)s
    %(output)s
    %(mode)s
    %(cval)s
    %(origin)s
    """
    input = numpy.asarray(input)
    if numpy.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    axis = _ni_support._check_axis(axis, input.ndim)
    if size < 1:
        raise RuntimeError('incorrect filter size')
    output, return_value = _ni_support._get_output(output, input)
    if (size // 2 + origin < 0) or (size // 2 + origin >= size):
        raise ValueError('invalid origin')
    mode = _ni_support._extend_mode_to_code(mode)
    _nd_image.uniform_filter1d(input, size, axis, output, mode, cval,
                               origin)
    return return_value
```

## Multi-dimensional uniform filter
```python
def uniform_filter(input, size=3, output=None, mode="reflect",
                   cval=0.0, origin=0):
    """Multi-dimensional uniform filter.
    Parameters
    ----------
    %(input)s
    size : int or sequence of ints
        The sizes of the uniform filter are given for each axis as a
        sequence, or as a single number, in which case the size is
        equal for all axes.
    %(output)s
    %(mode)s
    %(cval)s
    %(origin)s
    Notes
    -----
    The multi-dimensional filter is implemented as a sequence of
    one-dimensional uniform filters. The intermediate arrays are stored
    in the same data type as the output. Therefore, for output types
    with a limited precision, the results may be imprecise because
    intermediate results may be stored with insufficient precision.
    """
    input = numpy.asarray(input)
    output, return_value = _ni_support._get_output(output, input)
    sizes = _ni_support._normalize_sequence(size, input.ndim)
    origins = _ni_support._normalize_sequence(origin, input.ndim)
    axes = list(range(input.ndim))
    axes = [(axes[ii], sizes[ii], origins[ii])
                           for ii in range(len(axes)) if sizes[ii] > 1]
    if len(axes) > 0:
        for axis, size, origin in axes:
            uniform_filter1d(input, int(size), axis, output, mode,
                             cval, origin)
            input = output
    else:
        output[...] = input[...]
    return return_value
```

## Calculate a one-dimensional minimum filter along the given axis
```python
def minimum_filter1d(input, size, axis=-1, output=None,
                     mode="reflect", cval=0.0, origin=0):
    """Calculate a one-dimensional minimum filter along the given axis.
    The lines of the array along the given axis are filtered with a
    minimum filter of given size.
    Parameters
    ----------
    %(input)s
    size : int
        length along which to calculate 1D minimum
    %(axis)s
    %(output)s
    %(mode)s
    %(cval)s
    %(origin)s
    Notes
    -----
    This function implements the MINLIST algorithm [1]_, as described by
    Richard Harter [2]_, and has a guaranteed O(n) performance, `n` being
    the `input` length, regardless of filter size.
    References
    ----------
    .. [1] http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.42.2777
    .. [2] http://www.richardhartersworld.com/cri/2001/slidingmin.html
    """
    input = numpy.asarray(input)
    if numpy.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    axis = _ni_support._check_axis(axis, input.ndim)
    if size < 1:
        raise RuntimeError('incorrect filter size')
    output, return_value = _ni_support._get_output(output, input)
    if (size // 2 + origin < 0) or (size // 2 + origin >= size):
        raise ValueError('invalid origin')
    mode = _ni_support._extend_mode_to_code(mode)
    _nd_image.min_or_max_filter1d(input, size, axis, output, mode, cval,
                                  origin, 1)
    return return_value
```

## Calculate a one-dimensional maximum filter along the given axis
```python
def maximum_filter1d(input, size, axis=-1, output=None,
                     mode="reflect", cval=0.0, origin=0):
    """Calculate a one-dimensional maximum filter along the given axis.
    The lines of the array along the given axis are filtered with a
    maximum filter of given size.
    Parameters
    ----------
    %(input)s
    size : int
        Length along which to calculate the 1-D maximum.
    %(axis)s
    %(output)s
    %(mode)s
    %(cval)s
    %(origin)s
    Returns
    -------
    maximum1d : ndarray, None
        Maximum-filtered array with same shape as input.
        None if `output` is not None
    Notes
    -----
    This function implements the MAXLIST algorithm [1]_, as described by
    Richard Harter [2]_, and has a guaranteed O(n) performance, `n` being
    the `input` length, regardless of filter size.
    References
    ----------
    .. [1] http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.42.2777
    .. [2] http://www.richardhartersworld.com/cri/2001/slidingmin.html
    """
    input = numpy.asarray(input)
    if numpy.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    axis = _ni_support._check_axis(axis, input.ndim)
    if size < 1:
        raise RuntimeError('incorrect filter size')
    output, return_value = _ni_support._get_output(output, input)
    if (size // 2 + origin < 0) or (size // 2 + origin >= size):
        raise ValueError('invalid origin')
    mode = _ni_support._extend_mode_to_code(mode)
    _nd_image.min_or_max_filter1d(input, size, axis, output, mode, cval,
                                  origin, 0)
    return return_value
```

# min or max filter
```python
def _min_or_max_filter(input, size, footprint, structure, output, mode,
                       cval, origin, minimum):
    if structure is None:
        if footprint is None:
            if size is None:
                raise RuntimeError("no footprint provided")
            separable = True
        else:
            footprint = numpy.asarray(footprint)
            footprint = footprint.astype(bool)
            if numpy.alltrue(numpy.ravel(footprint), axis=0):
                size = footprint.shape
                footprint = None
                separable = True
            else:
                separable = False
    else:
        structure = numpy.asarray(structure, dtype=numpy.float64)
        separable = False
        if footprint is None:
            footprint = numpy.ones(structure.shape, bool)
        else:
            footprint = numpy.asarray(footprint)
            footprint = footprint.astype(bool)
    input = numpy.asarray(input)
    if numpy.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    output, return_value = _ni_support._get_output(output, input)
    origins = _ni_support._normalize_sequence(origin, input.ndim)
    if separable:
        sizes = _ni_support._normalize_sequence(size, input.ndim)
        axes = list(range(input.ndim))
        axes = [(axes[ii], sizes[ii], origins[ii])
                               for ii in range(len(axes)) if sizes[ii] > 1]
        if minimum:
            filter_ = minimum_filter1d
        else:
            filter_ = maximum_filter1d
        if len(axes) > 0:
            for axis, size, origin in axes:
                filter_(input, int(size), axis, output, mode, cval, origin)
                input = output
        else:
            output[...] = input[...]
    else:
        fshape = [ii for ii in footprint.shape if ii > 0]
        if len(fshape) != input.ndim:
            raise RuntimeError('footprint array has incorrect shape.')
        for origin, lenf in zip(origins, fshape):
            if (lenf // 2 + origin < 0) or (lenf // 2 + origin >= lenf):
                raise ValueError('invalid origin')
        if not footprint.flags.contiguous:
            footprint = footprint.copy()
        if structure is not None:
            if len(structure.shape) != input.ndim:
                raise RuntimeError('structure array has incorrect shape')
            if not structure.flags.contiguous:
                structure = structure.copy()
        mode = _ni_support._extend_mode_to_code(mode)
        _nd_image.min_or_max_filter(input, footprint, structure, output,
                                    mode, cval, origins, minimum)
    return return_value
```

## Calculates a multi-dimensional minimum filter
```python
def minimum_filter(input, size=None, footprint=None, output=None,
      mode="reflect", cval=0.0, origin=0):
    """Calculates a multi-dimensional minimum filter.
    Parameters
    ----------
    %(input)s
    %(size_foot)s
    %(output)s
    %(mode)s
    %(cval)s
    %(origin)s
    """
    return _min_or_max_filter(input, size, footprint, None, output, mode,
                              cval, origin, 1)
```

## Calculates a multi-dimensional maximum filter
```python
def maximum_filter(input, size=None, footprint=None, output=None,
      mode="reflect", cval=0.0, origin=0):
    """Calculates a multi-dimensional maximum filter.
    Parameters
    ----------
    %(input)s
    %(size_foot)s
    %(output)s
    %(mode)s
    %(cval)s
    %(origin)s
    """
    return _min_or_max_filter(input, size, footprint, None, output, mode,
                              cval, origin, 0)
```

## Rank filter 
```python
def _rank_filter(input, rank, size=None, footprint=None, output=None,
     mode="reflect", cval=0.0, origin=0, operation='rank'):
    input = numpy.asarray(input)
    if numpy.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    origins = _ni_support._normalize_sequence(origin, input.ndim)
    if footprint is None:
        if size is None:
            raise RuntimeError("no footprint or filter size provided")
        sizes = _ni_support._normalize_sequence(size, input.ndim)
        footprint = numpy.ones(sizes, dtype=bool)
    else:
        footprint = numpy.asarray(footprint, dtype=bool)
    fshape = [ii for ii in footprint.shape if ii > 0]
    if len(fshape) != input.ndim:
        raise RuntimeError('filter footprint array has incorrect shape.')
    for origin, lenf in zip(origins, fshape):
        if (lenf // 2 + origin < 0) or (lenf // 2 + origin >= lenf):
            raise ValueError('invalid origin')
    if not footprint.flags.contiguous:
        footprint = footprint.copy()
    filter_size = numpy.where(footprint, 1, 0).sum()
    if operation == 'median':
        rank = filter_size // 2
    elif operation == 'percentile':
        percentile = rank
        if percentile < 0.0:
            percentile += 100.0
        if percentile < 0 or percentile > 100:
            raise RuntimeError('invalid percentile')
        if percentile == 100.0:
            rank = filter_size - 1
        else:
            rank = int(float(filter_size) * percentile / 100.0)
    if rank < 0:
        rank += filter_size
    if rank < 0 or rank >= filter_size:
        raise RuntimeError('rank not within filter footprint size')
    if rank == 0:
        return minimum_filter(input, None, footprint, output, mode, cval,
                              origins)
    elif rank == filter_size - 1:
        return maximum_filter(input, None, footprint, output, mode, cval,
                              origins)
    else:
        output, return_value = _ni_support._get_output(output, input)
        mode = _ni_support._extend_mode_to_code(mode)
        _nd_image.rank_filter(input, rank, footprint, output, mode, cval,
                              origins)
        return return_value
```

#3 Calculates a multi-dimensional rank filte
```python
def rank_filter(input, rank, size=None, footprint=None, output=None,
      mode="reflect", cval=0.0, origin=0):
    """Calculates a multi-dimensional rank filter.
    Parameters
    ----------
    %(input)s
    rank : integer
        The rank parameter may be less then zero, i.e., rank = -1
        indicates the largest element.
    %(size_foot)s
    %(output)s
    %(mode)s
    %(cval)s
    %(origin)s
    """
    return _rank_filter(input, rank, size, footprint, output, mode, cval,
                        origin, 'rank')
 ```

## Calculates a multidimensional median filter.
```python
def median_filter(input, size=None, footprint=None, output=None,
                  mode="reflect", cval=0.0, origin=0):
    """
    Calculates a multidimensional median filter.
    Parameters
    ----------
    %(input)s
    %(size_foot)s
    %(output)s
    %(mode)s
    %(cval)s
    %(origin)s
    Returns
    -------
    median_filter : ndarray
        Return of same shape as `input`.
    """
    return _rank_filter(input, 0, size, footprint, output, mode, cval,
                        origin, 'median')
    ```


## Calculates a multi-dimensional percentile filter
```python
def percentile_filter(input, percentile, size=None, footprint=None,
                 output=None, mode="reflect", cval=0.0, origin=0):
    """Calculates a multi-dimensional percentile filter.
    Parameters
    ----------
    %(input)s
    percentile : scalar
        The percentile parameter may be less then zero, i.e.,
        percentile = -20 equals percentile = 80
    %(size_foot)s
    %(output)s
    %(mode)s
    %(cval)s
    %(origin)s
    """
    return _rank_filter(input, percentile, size, footprint, output, mode,
                                   cval, origin, 'percentile')
```

## Calculate a one-dimensional filter along the given axi
```python
def generic_filter1d(input, function, filter_size, axis=-1,
                 output=None, mode="reflect", cval=0.0, origin=0,
                 extra_arguments=(), extra_keywords = None):
    """Calculate a one-dimensional filter along the given axis.
    `generic_filter1d` iterates over the lines of the array, calling the
    given function at each line. The arguments of the line are the
    input line, and the output line. The input and output lines are 1D
    double arrays.  The input line is extended appropriately according
    to the filter size and origin. The output line must be modified
    in-place with the result.
    Parameters
    ----------
    %(input)s
    function : callable
        Function to apply along given axis.
    filter_size : scalar
        Length of the filter.
    %(axis)s
    %(output)s
    %(mode)s
    %(cval)s
    %(origin)s
    %(extra_arguments)s
    %(extra_keywords)s
    """
    if extra_keywords is None:
        extra_keywords = {}
    input = numpy.asarray(input)
    if numpy.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    output, return_value = _ni_support._get_output(output, input)
    if filter_size < 1:
        raise RuntimeError('invalid filter size')
    axis = _ni_support._check_axis(axis, input.ndim)
    if (filter_size // 2 + origin < 0) or (filter_size // 2 + origin >=
                                           filter_size):
        raise ValueError('invalid origin')
    mode = _ni_support._extend_mode_to_code(mode)
    _nd_image.generic_filter1d(input, function, filter_size, axis, output,
                      mode, cval, origin, extra_arguments, extra_keywords)
    return return_value
```
