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

### Analysis
The results demonstrate that higher **PSNR** values do not always correlate with better visual quality.
This is particularly evident in the test images where:

- Very low $K$ values (K=0.0001) often achieve visually sharp images but produce poor **PSNR** results with
because of the noise amplification
- Higher $K$ values (K=0.1) yield smoother images with better **PSNR** but lose fine details, leading to a
visually worse restoration but better mathematical similarity to the original image.

Moderate $K$ values ($K=0.001$ to $K=0.01$) provide the best balance between noise suppression and detail preservation.

**Mean Filter Restoration:**

- Generally poor performance
- High noise amplification at low $K$ values
- Limited detail recovery capability

**Gaussian Filter Restoration:**

- Not as good as expected performance
- noise suppression with reasonable detail preservation
- Optimal K values typically around 0.001-0.01

**Motion Blur Restoration:**

- Horizontal and Vertical: Good restoration quality when blur direction matches the kernel
- Diagonal: Good performance 
- Motion blur kernels show the most dramatic improvement with proper Wiener filtering

#### Overall Noise Level Impact:

Higher noise levels (variance 250) significantly degrade restoration quality across all methods:

Increased optimal K values needed for noise suppression
Greater loss of fine details
More aggressive regularization required


### Approach with additional Butterworth-Filter `butterworth_filter`
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

#### Analysis
As expected, the additional Butterworth lowpass filtering provides:

- Improved noise suppression in high-frequency regions
- Smoother visual appearance at the cost of some detail loss
- Better PSNR values in most cases, though with reduced sharpness


#### Conclusion
The Wiener filter demonstrates strong performance for motion blur restoration and moderate success with directional blur.
The critical factor is proper regularization parameter selection, which requires balancing noise suppression against
detail preservation. The results confirm that effective image restoration requires consideration of both quantitative
metrics and visual quality assessment.

<!-- pagebreak -->

## Advanced Implementation

>**Important Notice:**
> This implementation is submitted as-is. While this implementation demonstrates a theoretical understanding of Wiener
>filtering principles, the current solution does not perform sufficiently well to fully meet the requirements of the
> assignment.

While the core Wiener filter assumes knowledge of the degradation kernel, this advanced implementation attempts to
estimate the kernel by comparing the degraded image to natural reference images. In theory, this allows for
blind deblurring, where the kernel is unknown.

### Goal
To restore a degraded image without knowing the exact blurring kernel, by:

1. Finding the best matching reference image from a set of natural images using multiple similarity metrics.
2. Estimating the kernel by comparing the frequency response of the degraded image and the reference image.
3. Applying Wiener deconvolution with the estimated kernel and optimal regularization parameter.

```mermaid
graph LR
A[degraded image] --> B[find best matching reference]
B --> C[estimate kernel from frequency domain]
C --> D[apply Wiener filter with multiple K values]
D --> E[select best restoration]
E --> F[restored image]
```

### Implementation

#### Reference Image Selection (`find_best_reference_match`)

A multi-metric approach is used to find the best matching reference image from a collection of natural images
(landscapes, portraits, animals, food, text, boats). The selection is based on five different similarity metrics:

##### 1. Histogram Correlation
Compares the intensity distribution between images:
```python
hist_ref = cv2.calcHist([ref_processed.astype(np.float32)], [0], None, [256], [0, 1])
hist_deg = cv2.calcHist([degraded_img.astype(np.float32)], [0], None, [256], [0, 1])
hist_corr = cv2.compareHist(hist_ref, hist_deg, cv2.HISTCMP_CORREL)
```

##### 2. Structural Similarity Index (SSIM)
Measures perceptual similarity by comparing luminance, contrast, and structure:
```python
def structural_similarity_simple(img1: np.ndarray, img2: np.ndarray) -> float:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    
    # Calculate means, variances, and covariance
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    # SSIM formula
    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    
    ssim_map = numerator / (denominator + 1e-10)
    return np.mean(ssim_map)
```

##### 3. Edge Correlation
Compares edge structures using Canny edge detection:
```python
edges_ref = cv2.Canny((ref_processed * 255).astype(np.uint8), 50, 150)
edges_deg = cv2.Canny((degraded_img * 255).astype(np.uint8), 50, 150)

# Normalize and calculate correlation
edges_ref = edges_ref.astype(np.float64) / 255.0
edges_deg = edges_deg.astype(np.float64) / 255.0
edge_corr = np.corrcoef(edges_ref.flatten(), edges_deg.flatten())[0, 1]
```

##### 4. Texture Similarity
Analyzes local variance patterns to compare texture:
```python
def texture_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    # Convert to uint8 for texture analysis
    img1_uint8 = (img1 * 255).astype(np.uint8)
    img2_uint8 = (img2 * 255).astype(np.uint8)

    # Calculate local variance as texture measure
    kernel = np.ones((9, 9), np.float32) / 81
    
    mean1 = cv2.filter2D(img1_uint8.astype(np.float32), -1, kernel)
    mean2 = cv2.filter2D(img2_uint8.astype(np.float32), -1, kernel)
    
    sqr1 = cv2.filter2D((img1_uint8.astype(np.float32)) ** 2, -1, kernel)
    sqr2 = cv2.filter2D((img2_uint8.astype(np.float32)) ** 2, -1, kernel)
    
    var1 = sqr1 - mean1 ** 2
    var2 = sqr2 - mean2 ** 2
    
    # Correlation between variance maps
    corr = np.corrcoef(var1.flatten(), var2.flatten())[0, 1]
    return 0 if np.isnan(corr) else corr
```

##### 5. Frequency Domain Correlation
Compares magnitude spectra in the frequency domain:
```python
def frequency_correlation(img1: np.ndarray, img2: np.ndarray) -> float:
    # FFT magnitude spectra
    fft1 = np.fft.fft2(img1)
    fft2 = np.fft.fft2(img2)
    
    mag1 = np.abs(fft1)
    mag2 = np.abs(fft2)
    
    # Log transform to compress dynamic range
    log_mag1 = np.log(mag1 + 1)
    log_mag2 = np.log(mag2 + 1)
    
    # Correlation of log magnitudes
    corr = np.corrcoef(log_mag1.flatten(), log_mag2.flatten())[0, 1]
    return 0 if np.isnan(corr) else corr
```

##### Combined Similarity Score
The final similarity score is a weighted combination of all metrics:
```python
combined_score = (0.25 * hist_corr +
                  0.30 * ssim_score +
                  0.20 * edge_corr +
                  0.15 * texture_corr +
                  0.10 * freq_corr)
```

The weights prioritize SSIM (30%) and histogram correlation (25%) as the most important metrics, followed by edge correlation (20%), texture similarity (15%), and frequency correlation (10%).

#### Kernel Estimation (`estimate_kernel`)

The kernel estimation process uses the frequency domain relationship between the reference and degraded images:

##### Step 1: Adaptive Kernel Size Estimation
```python
def estimate_kernel_size(degraded_img: np.ndarray) -> int:
    min_dim = min(degraded_img.shape[:2])
    size = max(5, min(25, min_dim // 15))
    return size if size % 2 == 1 else size + 1
```

##### Step 2: Frequency Domain Kernel Estimation
The degradation kernel is estimated using:
$$
H(u,v) \approx \frac{G(u,v) \cdot F^*(u,v)}{|F(u,v)|^2 + \epsilon}
$$

```python
F = np.fft.fft2(original_img)
G = np.fft.fft2(degraded_img)

F_conj = np.conj(F)
F_magnitude_sq = np.abs(F) ** 2
epsilon = np.mean(F_magnitude_sq) * 0.01

H_estimate = (G * F_conj) / (F_magnitude_sq + epsilon)
```

##### Step 3: Spatial Domain Conversion and Centering
```python
h_estimate = np.fft.ifft2(H_estimate)
h_estimate = np.real(np.fft.ifftshift(h_estimate))

# Find the peak and center the kernel
max_pos = np.unravel_index(np.argmax(h_estimate), h_estimate.shape)
center_row, center_col = max_pos
half_k = kernel_size // 2

# Extract kernel with boundary checking
start_r = max(center_row - half_k, 0)
start_c = max(center_col - half_k, 0)
end_r = start_r + kernel_size
end_c = start_c + kernel_size

kernel = h_estimate[start_r:end_r, start_c:end_c]

# Normalize kernel
kernel_sum = np.sum(kernel)
kernel /= kernel_sum
```

#### Advanced Wiener Deconvolution (`advanced_wiener_deconvolution`)

The complete restoration process involves:

##### Step 1: Reference Matching and Kernel Estimation
```python
best_idx, similarity, reference = find_best_reference_match(degraded_img, reference_images)
estimated_kernel = estimate_kernel(reference, degraded_img)
```

##### Step 2: Multi-Parameter Optimization
The system tests multiple regularization parameters to find the optimal restoration:
```python
K_values = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5]  # Default values

for K in K_values:
    restored = wiener_deconvolution(degraded_img, estimated_kernel, K)
    quality_metrics[K] = calculate_quality_metrics(reference, restored)

# Select best K based on PSNR
best_K = max(K_values, key=lambda k: quality_metrics[k]['psnr'])
```

##### Step 3: Quality Metrics Evaluation
The system evaluates restoration quality using multiple metrics:

- **PSNR (Peak Signal-to-Noise Ratio)**: Measures pixel-wise accuracy
- **MSE (Mean Squared Error)**: Quantifies reconstruction error
- **SNR (Signal-to-Noise Ratio)**: Evaluates signal integrity
- **Edge Preservation**: Measures preservation of edge structures

```python
def calculate_quality_metrics(original: np.ndarray, restored: np.ndarray) -> Dict:
    mse = np.mean((original - restored) ** 2)
    psnr = float('inf') if mse == 0 else 10 * np.log10(1.0 / mse)

    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - restored) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))

    # Edge preservation using gradient correlation
    grad_orig = np.concatenate([np.gradient(original, axis=1).flatten(), 
                               np.gradient(original, axis=0).flatten()])
    grad_rest = np.concatenate([np.gradient(restored, axis=1).flatten(), 
                               np.gradient(restored, axis=0).flatten()])

    edge_preservation = 0.0
    if np.std(grad_orig) > 1e-10 and np.std(grad_rest) > 1e-10:
        edge_corr = np.corrcoef(grad_orig, grad_rest)[0, 1]
        edge_preservation = edge_corr if not np.isnan(edge_corr) else 0.0

    return {'psnr': psnr, 'mse': mse, 'snr': snr, 'edge_preservation': edge_preservation}
```


#### Visualization and Analysis

Visualization is realised in `plot_results()`, showing:
- Best reference match
- Degraded input image
- Best restoration result
- Estimated vs. true kernel comparison 
- PSNR vs. regularization parameter curve with optimal point highlighted


