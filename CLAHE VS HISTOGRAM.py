import numpy as np
import matplotlib.pyplot as plt
import os
import imageio.v2 as imageio
from skimage.measure import shannon_entropy
from skimage.metrics import structural_similarity as ssim
from math import log10

# --- KERNEL MATRICES (Defined once globally) ---
gaussian_kernel = (1/16) * np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
])

sharpen_kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

# --- IMAGE PROCESSING FUNCTIONS ---

def convolve2d(image, kernel):
    """Applies a 2D convolution to an image using a given kernel."""
    ih, iw = image.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    output = np.zeros_like(image, dtype=np.float32)

    kernel_flipped = np.flipud(np.fliplr(kernel))

    for y in range(ih):
        for x in range(iw):
            roi = padded[y:y+kh, x:x+kw]
            output[y, x] = np.sum(roi * kernel_flipped)

    output = np.clip(output, 0, 255).astype(np.uint8)
    return output

def histogram_equalization(img):
    """Performs global histogram equalization on a grayscale image."""
    height, width = img.shape
    hist = [0]*256
    for y in range(height):
        for x in range(width):
            hist[img[y,x]] += 1

    cdf_arr = [0]*256
    cdf_arr[0] = hist[0]
    for i in range(1, 256):
        cdf_arr[i] = cdf_arr[i-1] + hist[i]

    # Find the first non-zero CDF value
    cdf_min = next((x for x in cdf_arr if x != 0), 0)
    total_pixels = height * width
    if total_pixels - cdf_min == 0: # Avoid division by zero
        return img.copy()

    output = np.zeros_like(img, dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            pixel_val = img[y,x]
            cdf_val = cdf_arr[pixel_val]
            eq_val = round(((cdf_val - cdf_min) / (total_pixels - cdf_min)) * 255)
            output[y,x] = eq_val
    return output

def clip_histogram(hist, clip_limit_count):
    """Clips the histogram at a specific limit and redistributes the excess."""
    excess = 0
    for i in range(256):
        if hist[i] > clip_limit_count:
            excess += hist[i] - clip_limit_count
            hist[i] = int(clip_limit_count)
            
    # Redistribute the excess evenly
    excess = int(excess)
    redistribute = excess // 256
    remainder = excess % 256

    for i in range(256):
        hist[i] += redistribute
    for i in range(remainder):
        hist[i] += 1
    return hist

def cdf(hist):
    """Calculates the cumulative distribution function from a histogram."""
    cdf_arr = [0]*256
    cdf_arr[0] = hist[0]
    for i in range(1, 256):
        cdf_arr[i] = cdf_arr[i-1] + hist[i]
    return cdf_arr

def cdf_lookup(cdf_tile, pixel_val):
    """Maps a pixel value using a given CDF lookup table."""
    cdf_min = next((x for x in cdf_tile if x != 0), 0)
    total = cdf_tile[-1]
    
    if total - cdf_min == 0: # Avoid division by zero
        return pixel_val

    val = round(((cdf_tile[pixel_val] - cdf_min) / (total - cdf_min)) * 255)
    return max(0, min(255, val))

def bilinear_interpolation(v00, v01, v10, v11, dx, dy):
    """Performs bilinear interpolation."""
    top = v00 * (1 - dx) + v01 * dx
    bottom = v10 * (1 - dx) + v11 * dx
    return round(top * (1 - dy) + bottom * dy)

def clahe(img, clip_limit=0.01, tile_grid_size=(8, 8)):
    """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
    h, w = img.shape
    n_tiles_y, n_tiles_x = tile_grid_size
    tile_h = h // n_tiles_y
    tile_w = w // n_tiles_x

    # Calculate CDF for each tile
    cdfs = []
    for i in range(n_tiles_y):
        row_cdfs = []
        y0 = i * tile_h
        y1 = h if i == n_tiles_y - 1 else (i + 1) * tile_h
        for j in range(n_tiles_x):
            x0 = j * tile_w
            x1 = w if j == n_tiles_x - 1 else (j + 1) * tile_w
            tile = img[y0:y1, x0:x1]
            
            if tile.size == 0: continue

            hist = [0]*256
            for y_tile in range(tile.shape[0]):
                for x_tile in range(tile.shape[1]):
                    hist[tile[y_tile, x_tile]] += 1

            # FIX: Cast clip_limit_count to an integer to prevent float errors.
            # This is the line that solves the 'float object cannot be interpreted as an integer' error.
            clip_limit_count = int(clip_limit * tile.size)
            
            hist = clip_histogram(hist, clip_limit_count)
            cdf_tile = cdf(hist)
            row_cdfs.append(cdf_tile)
        cdfs.append(row_cdfs)

    # Perform interpolation
    output = np.zeros_like(img, dtype=np.uint8)
    for y in range(h):
        ty = (y + 0.5) / tile_h - 0.5
        i = int(np.floor(ty))
        dy = ty - i
        i0 = max(0, min(i, n_tiles_y - 1))
        i1 = max(0, min(i + 1, n_tiles_y - 1))

        for x in range(w):
            tx = (x + 0.5) / tile_w - 0.5
            j = int(np.floor(tx))
            dx = tx - j
            j0 = max(0, min(j, n_tiles_x - 1))
            j1 = max(0, min(j + 1, n_tiles_x - 1))

            pixel_val = img[y, x]

            v00 = cdf_lookup(cdfs[i0][j0], pixel_val)
            v01 = cdf_lookup(cdfs[i0][j1], pixel_val)
            v10 = cdf_lookup(cdfs[i1][j0], pixel_val)
            v11 = cdf_lookup(cdfs[i1][j1], pixel_val)

            val = bilinear_interpolation(v00, v01, v10, v11, dx, dy)
            output[y, x] = val

    return output

# --- EVALUATION AND UTILITY FUNCTIONS ---

def psnr(img1, img2):
    """Calculates the Peak Signal-to-Noise Ratio between two images."""
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32))**2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 10 * log10((PIXEL_MAX**2) / mse)

def evaluate_image(image, reference=None):
    """Calculates and returns a dictionary of image quality metrics."""
    metrics = {
        'Entropy': shannon_entropy(image),
        'RMS Contrast': np.sqrt(np.mean((image - np.mean(image))**2)),
        'PSNR': None,
        'SSIM': None
    }
    if reference is not None:
        metrics['PSNR'] = psnr(reference, image)
        metrics['SSIM'] = ssim(reference, image, data_range=image.max() - image.min())
    return metrics

def save_plot_comparison(images, titles, filename, output_folder, metrics_list):
    """Saves a plot comparing images, their histograms, and quality metrics."""
    # Modify to include sharpened image
    if len(images) == 3:  # Original array has [original, he, clahe]
        # We need to add the sharpened image which should be processed before calling this function
        blurred = convolve2d(images[0], gaussian_kernel)
        sharpened = convolve2d(blurred, sharpen_kernel)
        
        # Create new arrays with all 4 images
        all_images = [images[0], sharpened, images[1], images[2]]
        all_titles = ['Original', 'Gaussian+Sharpened', 'Histogram Equalization', 'CLAHE']
        
        # Add sharpened metrics
        sharp_metrics = evaluate_image(sharpened, reference=images[0])
        all_metrics = [metrics_list[0], sharp_metrics, metrics_list[1], metrics_list[2]]
    else:
        # If already has 4 images, use as-is
        all_images = images
        all_titles = titles
        all_metrics = metrics_list
    
    # Create a 2x4 grid (2 rows, 4 columns)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    for i in range(4):  # Loop through the 4 images
        metrics = all_metrics[i]
        title_text = f"{all_titles[i]}\nEntropy: {metrics['Entropy']:.2f} | RMS: {metrics['RMS Contrast']:.2f}"
        if metrics.get('PSNR') is not None:
            title_text += f"\nPSNR: {metrics['PSNR']:.2f} dB"
        if metrics.get('SSIM') is not None:
            title_text += f" | SSIM: {metrics['SSIM']:.4f}"

        axes[0, i].imshow(all_images[i], cmap='gray')
        axes[0, i].set_title(title_text, fontsize=10)
        axes[0, i].axis('off')
        axes[1, i].hist(all_images[i].ravel(), bins=256, range=[0, 256], color='gray')
        axes[1, i].set_title('Histogram')
        axes[1, i].set_xlim([0, 256])

    plt.tight_layout()
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, 'comparison_' + os.path.splitext(filename)[0] + '.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[✓] Gambar perbandingan + histogram disimpan: {save_path}")

# --- MAIN EXECUTION ---

def main():
    """Main function to run the image processing pipeline."""
    input_folder = 'images'
    output_folder = 'results'

    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
        print(f"[INFO] Folder '{input_folder}' tidak ditemukan. Folder telah dibuat.")
        print(f"[INFO] Silakan letakkan gambar Anda di dalam folder 'images' dan jalankan kembali skrip ini.")
        return

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"[WARN] Tidak ada gambar yang ditemukan di folder '{input_folder}'.")
        return

    for filename in image_files:
        try:
            path = os.path.join(input_folder, filename)
            img = imageio.imread(path)
            if img.ndim == 3: # Convert to grayscale if it's a color image
                img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

            # --- Processing Pipeline ---
            # 1. Apply Gaussian blur to the original image
            blurred = convolve2d(img, gaussian_kernel)
            
            # 2. Sharpen the blurred image
            sharpened = convolve2d(blurred, sharpen_kernel)

            # 3. Apply histogram enhancement techniques on the sharpened image
            he_img = histogram_equalization(sharpened)
            clahe_img = clahe(sharpened, clip_limit=0.01, tile_grid_size=(8, 8))

            # All 4 images and their titles
            images = [img, sharpened, he_img, clahe_img]
            titles = ['Original', 'Gaussian+Sharpened', 'HE', 'CLAHE']

            # --- Evaluation ---
            metrics_list = []
            metrics_list.append(evaluate_image(img))
            metrics_list.append(evaluate_image(sharpened, reference=img))
            metrics_list.append(evaluate_image(he_img, reference=img))
            metrics_list.append(evaluate_image(clahe_img, reference=img))

            print(f"\n[INFO] Evaluasi untuk: {filename}")
            for title, metrics in zip(titles, metrics_list):
                print(f"{title}: Entropy={metrics['Entropy']:.2f}, RMS={metrics['RMS Contrast']:.2f}"
                      + (f", PSNR={metrics['PSNR']:.2f} dB" if metrics.get('PSNR') is not None else "")
                      + (f", SSIM={metrics['SSIM']:.4f}" if metrics.get('SSIM') is not None else ""))
            
            # --- Display the four images with evaluation metrics ---
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for i, (image, title, metrics) in enumerate(zip(images, titles, metrics_list)):
                axes[i].imshow(image, cmap='gray')
                metric_text = f"Entropy: {metrics['Entropy']:.2f} | RMS: {metrics['RMS Contrast']:.2f}"
                if metrics.get('PSNR') is not None:
                    metric_text += f"\nPSNR: {metrics['PSNR']:.2f} dB"
                if metrics.get('SSIM') is not None:
                    metric_text += f" | SSIM: {metrics['SSIM']:.4f}"
                axes[i].set_title(f"{title}\n{metric_text}", fontsize=10)
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # Save the comparison plot with all 4 images
            save_plot_comparison(images, titles, filename, output_folder, metrics_list)

        except Exception as e:
            print(f"[ERROR] Gagal memproses gambar {filename}: {e}")

if __name__ == '__main__':
    main()