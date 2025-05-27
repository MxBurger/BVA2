# Wiener Filter

## Core Implementation

The Wiener Filter is a procedure for image restoration, which were degraded by noise and blurring. The
basic idea is based on the following model of the image degradation:
$$
g(x,y) = f(x,y) * h(x,y) + n(x,y)
$$
where:
- $g(x,y)$ is the observed image
- $f(x,y)$ is the original image
- $h(x,y)$ is the blurring function
- $n(x,y)$ is the noise
- $*$ is the convolution operator


```mermaid
graph LR
A[original image] --> B[blur image]
B --> C[add noise]
```

The Wiener filter now aims to estimate the original image $f(x,y)$ from the observed image $g(x,y)$ by
inverting the degradation process. The Wiener filter is defined as:
$$
W(u,v) = \frac{H^*(u,v)}{|H(u,v)|^2 + K}
$$
where:
- $W(u,v)$ is the Wiener filter in the frequency domain
- $H(u,v)$ is the Fourier transform of the degradation function
- $H^*(u,v)$ is the complex conjugate of $H(u,v)$
- $K$ is a regularization parameter (often related to the noise level)

> The regularization parameter $K$ represents the noise-to-signal power ratio and controls the trade-off between noise
> suppression and detail preservation. When $K$ is small (approaching 0), the filter behaves more like an inverse filter,
>attempting to restore fine details but potentially amplifying noise. When $K$ is large, the filter becomes more aggressive,
>suppressing noise effectively but at the cost of losing high-frequency details and making the image appear smoother or blurred.

After computing the Wiener filter, it can be applied in the frequency domain to the degraded image:
$$
F'(u,v) = W(u,v) * G(u,v)
$$
where:
- $F'(u,v)$ is the restored image in the frequency domain
- $W(u,v)$ is the Wiener filter
- $G(u,v)$ is the Fourier transform of the degraded image


Finally, the inverse Fourier transform is applied to obtain the restored image in the spatial domain:
$$
f(x,y) = \mathcal{F}^{-1}(F'(u,v))
$$

### Implementation

#### Methods for degrading the image

##### Creating the Kernels `create_kernels`

This implemented method creates different blurring kernels.

###### Mean Filter (Box Filter)
```python
kernels['mean'] = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
```
Simple averaging filter, which replaces each pixel with the average of its neighbors.

###### Gaussian Filter
```python
sigma = kernel_size / 3.0
ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
xx, yy = np.meshgrid(ax, ax)
kernels['gaussian'] = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
```
Creates a Gaussian kernel with a standard deviation of `sigma`. The kernel is normalized to sum to 1.
For sigma 1/3 of the kernel size will be used.

###### Motion Blur
- **Horizontal**: Only the middle row of the kernel contains non-zero values.
```python
kernels['horizontal'] = np.zeros((kernel_size, kernel_size))
kernels['horizontal'][kernel_size // 2, :] = 1.0 / kernel_size
```
- **Vertical**: Only the middle column of the kernel contains non-zero values.
```python
kernels['vertical'] = np.zeros((kernel_size, kernel_size))
kernels['vertical'][:, kernel_size // 2] = 1.0 / kernel_size
```

- **Diagonal**: The main diagonal of the kernel contains non-zero values.
```python
kernels['diagonal'] = np.zeros((kernel_size, kernel_size))
np.fill_diagonal(kernels['diagonal'], 1.0 / kernel_size)
```

##### Noise addition `add_noise`
For the noise addition, a Gaussian noise is added to the image.
```python
noise = np.random.normal(0, np.sqrt(noise_variance), image.shape)
noisy_image = image + noise
return np.clip(noisy_image, 0, 255).astype(np.uint8)
```
`np.clip` is used to ensure that the pixel values are in the range [0, 255].

##### Degrading the image `degrade_image`

This method takes an image and a kernel as input, applies the kernel to the image using convolution,
and adds noise to the result.
```python
blurred = cv2.filter2D(image.astype(np.float64), -1, kernel)
degraded = self.add_noise(blurred, noise_variance)
```

#### Wiener Deconvolution `wiener_deconvolution` (Core Method)

The Wiener deconvolution is performed in the frequency domain using the Fast Fourier Transform (FFT).

The steps are as follows:

##### Step 1: Fourier Transform

The degraded input image and kernel are transformed into the frequency domain using the FFT.
```python
G = np.fft.fft2(degraded.astype(np.float64))
H = np.fft.fft2(kernel_padded)
```

##### Step 2: Conjugate of the Kernel

The conjugate of the kernel is computed.
```python
H_conj = np.conj(H)
```
This calculates $H*$ (the complex conjugate of $H$).

##### Step 3: Magnitude Squared of the Kernel
The magnitude squared of the kernel is computed.
```python
H_squared = np.abs(H) ** 2
```
This calculates $|H|^2$ (the magnitude squared of $H$).

##### Step 4: Applying the Wiener Filter

The formula for the Wiener filter is:
$$
W(u,v) = \frac{H*(u,v)}{ |H(u,v)|² + K}
\\
F'(u,v) = W(u,v) * G(u,v)
$$
where:
- $W(u,v)$ is the Wiener filter
- $F'(u,v)$ is the restored image in the frequency domain
- $K$ is the Regularization parameter (noise to signal ratio)

```python
W = H_conj / (H_squared + K + epsilon)
F_estimate = W * G
```

##### Step 5: Inverse Fourier Transform
The inverse FFT is applied to the filtered image to obtain the restored image in the spatial domain.
Only the real part of the restored image is taken, as the imaginary part should be negligible.
```python
restored = np.fft.ifft2(F_estimate)
restored = np.real(restored)
```


### Test-Pictures

To quantify the results, the PSNR (Peak Signal-to-Noise Ratio) is used. The PSNR is defined as:
$$
PSNR = 10 \cdot \log_{10} \left( \frac{MAX^2}{MSE} \right)
$$
where:
- $MAX$ is the maximum possible pixel value (255 for 8-bit images)
- $MSE$ is the mean squared error between the original and restored images

A higher PSNR value indicates a better restoration quality
-> better mathematical similarity to the original image.
Be aware that mathematical similarity does not always mean subjective visual similarity,
as the test results clearly show.

#### Mean Filter

![mean_kernel_size.png](img/mean_kernel_size.png)
![mean_noise_level.png](img/mean_noise_level.png)

![mean_kernel_size.png](img/boat/mean_kernel_size.png)
![mean_noise_level.png](img/boat/mean_noise_level.png)

#### Gaussian Filter
![gauss_kernel_size.png](img/gauss_kernel_size.png)
![gauss_noise_level.png](img/gauss_noise_level.png)

![gauss_kernel_size.png](img/boat/gauss_kernel_size.png)
![gauss_noise_level.png](img/boat/gauss_noise_level.png)

#### Horizontal Motion Blur
![horizontal_kernel_size.png](img/horizontal_kernel_size.png)
![horizontal_noise_level.png](img/horizontal_noise_level.png)

![horizontal_kernel_size.png](img/boat/horizontal_kernel_size.png)
![horizontal_noise_level.png](img/boat/horizontal_noise_level.png)

#### Vertical Motion Blur
![vertical_kernel_size.png](img/vertical_kernel_size.png)
![vertical_noise_level.png](img/vertical_noise_level.png)

![vertical_kernel_size.png](img/boat/vertical_kernel_size.png)
![vertical_noise_level.png](img/boat/vertical_noise_level.png)

#### Diagonal Motion Blur
![diagonal_kernel_size.png](img/diagonal_kernel_size.png)
![diagonal_noise_level.png](img/diagonal_noise_level.png)

![diagonal_kernel_size.png](img/boat/diagonal_kernel_size.png)
![diagonal_noise_level.png](img/boat/diagonal_noise_level.png)


> TODO: Discuss the results

#### Additional approach with additional Butterworth-Filter `butterworth_filter`
For additional filtering, a Butterworth Lowpass-filter can be applied to the restored image.

The Butterworth filter is defined as:
$$
H(u,v) = \frac{1}{1 + \left( \frac{D(u,v)}{D_0} \right)^{2n}}
$$
where:
- $D_0$ is the cutoff frequency
- $D(u,v)$ is the distance from the origin in the frequency domain
- $n$ is the order of the filter
- $H(u,v)$ is the filter transfer function

```python
D = np.sqrt(u ** 2 + v ** 2)
H = 1 / (1 + (D / D0) ** (2 * order))
```

The Butterworth filter is combined with the Wiener filter in the method `wiener_with_butterworth`.
$$
W = W * B
$$
where:
- $W$ is the Wiener filter
- $B$ is the Butterworth filter

### Test-Pictures
#### Mean Filter with Butterworth
![mean_kernel_size_butterworth.png](img/mean_and_butterworth.png)
![mean_and_butterworth.png](img/boat/mean_and_butterworth.png)


#### Gaussian Filter with Butterworth
![gauss_kernel_size_butterworth.png](img/gauss_and_butterworth.png)
![gauss_and_butterworth.png](img/boat/gauss_and_butterworth.png)


#### Horizontal Motion Blur with Butterworth
![horizontal_kernel_size_butterworth.png](img/horizontal_and_butterworth.png)
![horizontal_and_butterworth.png](img/boat/horizontal_and_butterworth.png)

#### Vertical Motion Blur with Butterworth
![vertical_kernel_size_butterworth.png](img/vertical_and_butterworth.png)
![vertical_and_butterworth.png](img/boat/vertical_and_butterworth.png)

#### Diagonal Motion Blur with Butterworth
![diagonal_kernel_size_butterworth.png](img/diagonal_and_butterworth.png)
![diagonal_and_butterworth.png](img/boat/diagonal_and_butterworth.png)

## Advanced Implementation

While the core Wiener filter assumes knowledge of the degradation kernel, this advanced implementation attempts to
estimate the kernel by comparing the degraded image to natural reference images. In theory, this allows for
blind deblurring, where the kernel is unknown.

### Goal
To restore a degraded image without knowing the exact blurring kernel, by:

1. Finding the best matching reference image from a set of natural images.

2. Estimating the kernel by comparing the frequency response of the degraded image and the reference image.

3. Applying Wiener deconvolution with the estimated kernel.

```mermaid
graph LR
A[degraded image] --> B[find best matching reference]
B --> C[estimate kernel from frequency domain]
C --> D[apply Wiener filter]
D --> E[restore image]
```

### Implementation

#### Reference Image Selection

Using a collection of reference images (e.g. landscapes, portraits, text), the best match is determined based on:

- **Gradient correlation** (horizontal + vertical)
- **Structural Similarity Index (SSIM)**

>Gradient Correlation is a similarity metric that compares the gradient (edge) structures of two images rather than
> their raw pixel values. It measures how well the direction and strength of edges in one image align with those in
> another, typically using the dot product of gradient vectors. This method is particularly useful for images with
> illumination differences, as gradients are less sensitive to uniform brightness changes. A higher gradient correlation
> indicates greater structural similarity between the images.

```python
ref_grad = np.gradient(ref_processed)
deg_grad = np.gradient(deg_processed)
```
This computes edge information in x and y directions of both the reference and degraded images.
The average correlation (horizontal and vertical) is then calculated as:
```python
grad_similarity = np.corrcoef(ref_grad[0].flatten(), deg_grad[0].flatten())[0, 1]
grad_similarity += np.corrcoef(ref_grad[1].flatten(), deg_grad[1].flatten())[0, 1]
grad_similarity /= 2
```

>The Structural Similarity Index (SSIM) is a perceptual metric that measures image similarity by comparing luminance,
> contrast, and structural information between two images. Unlike simple metrics like MSE or PSNR, SSIM models the human
> visual system's sensitivity to changes in structure and texture. It produces a value between -1 and 1, where 1
> indicates perfect similarity. SSIM is commonly used for assessing image quality in restoration tasks. 

```python
def structural_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    min_h = min(img1.shape[0], img2.shape[0])
    min_w = min(img1.shape[1], img2.shape[1])
    img1 = img1[:min_h, :min_w]
    img2 = img2[:min_h, :min_w]

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu1 = cv2.GaussianBlur(img1, (7, 7), 1.5)
    mu2 = cv2.GaussianBlur(img2, (7, 7), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (7, 7), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (7, 7), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (7, 7), 1.5) - mu1_mu2

    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

    ssim_map = numerator / (denominator + 1e-10)
    return np.mean(ssim_map)
```

For one final score, the average of the gradient correlation and SSIM is computed:
```python
combined_score = 0.7 * grad_similarity + 0.3 * ssim_score
```