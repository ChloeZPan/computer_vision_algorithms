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
