import numpy as np
import matplotlib.pyplot as plt
from typing import Dict


def plot_results(results: Dict, degraded_img: np.ndarray, true_kernel: np.ndarray = None, kernel_info: str = None):
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 5, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(results['reference_match']['reference_image'], cmap='gray')
    ax1.set_title('Best Reference Match')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(degraded_img, cmap='gray')
    ax2.set_title('Degraded Input')
    ax2.axis('off')

    best_K = results['best_K']
    ax3 = fig.add_subplot(gs[0, 2])
    best_restored = results['restored_images'][best_K]
    ax3.imshow(best_restored, cmap='gray')
    psnr_best = results['quality_metrics'][best_K]['psnr']
    ax3.set_title(f'Best Restoration\nK={best_K}, PSNR={psnr_best:.1f}dB')
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(results['estimated_kernel'], cmap='hot')
    ax4.set_title('Estimated Kernel')
    ax4.axis('off')

    if true_kernel is not None:
        ax5 = fig.add_subplot(gs[0, 4])
        ax5.imshow(true_kernel, cmap='hot')
        title = 'True Kernel'
        if kernel_info:
            title += f'\n({kernel_info})'
        ax5.set_title(title)
        ax5.axis('off')

    ax6 = fig.add_subplot(gs[1, 0:3])
    K_values = list(results['quality_metrics'].keys())
    psnr_values = [results['quality_metrics'][K]['psnr'] for K in K_values]

    ax6.semilogx(K_values, psnr_values, 'bo-', linewidth=2, markersize=8)
    ax6.set_xlabel('Regularization K')
    ax6.set_ylabel('PSNR (dB)')
    ax6.set_title('PSNR vs Regularization Parameter')
    ax6.grid(True, alpha=0.3)

    best_psnr = results['quality_metrics'][best_K]['psnr']
    ax6.semilogx(best_K, best_psnr, 'ro', markersize=10, label=f'Best: K={best_K}')
    ax6.legend()

    fig.suptitle('Advanced Wiener Filter - Kernel Analysis', fontsize=16, y=0.95)
    plt.tight_layout()
    plt.show()


def print_summary(results: Dict):
    best_K = results['best_K']
    best_metrics = results['quality_metrics'][best_K]

    print(f"Best Configuration: K = {best_K}")
    print(f"Best PSNR: {best_metrics['psnr']:.2f} dB")
    print(f"SNR: {best_metrics['snr']:.2f} dB")
    print(f"Edge Preservation: {best_metrics['edge_preservation']:.3f}")
