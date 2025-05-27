import numpy as np
import matplotlib.pyplot as plt
from typing import Dict


def plot_kernel(kernel: np.ndarray, title: str = "Estimated Kernel"):
    """Plot individual kernel."""
    plt.figure(figsize=(4, 4))
    plt.imshow(kernel, cmap='hot')
    plt.title(title)
    plt.colorbar()
    plt.show()


def plot_results(results: Dict, degraded_img: np.ndarray, true_kernel: np.ndarray = None, kernel_info: str = None):
    """Plot results of advanced Wiener filtering."""
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

    # Show true kernel if provided
    if true_kernel is not None:
        ax5 = fig.add_subplot(gs[0, 4])
        ax5.imshow(true_kernel, cmap='hot')
        title = 'True Kernel'
        if kernel_info:
            title += f'\n({kernel_info})'
        ax5.set_title(title)
        ax5.axis('off')

    ax6 = fig.add_subplot(gs[1, 0:2])
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

    if 'content_similarity_analysis' in results and results['content_similarity_analysis']:
        content_analysis = results['content_similarity_analysis']

        ax7 = fig.add_subplot(gs[1, 2:5])
        labels = list(content_analysis.keys())
        spatial_corr = [content_analysis[label]['similarity_metrics']['spatial_correlation']
                        for label in labels]
        best_psnr_per_ref = [content_analysis[label]['best_metrics']['psnr']
                             for label in labels]

        ax7.scatter(spatial_corr, best_psnr_per_ref, s=100, alpha=0.7,
                    c=range(len(labels)), cmap='viridis')
        ax7.set_xlabel('Spatial Correlation with Degraded Image')
        ax7.set_ylabel('Best PSNR (dB)')
        ax7.set_title('Content Similarity vs Restoration Quality')
        ax7.grid(True, alpha=0.3)

        for i, label in enumerate(labels):
            ax7.annotate(label, (spatial_corr[i], best_psnr_per_ref[i]),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)

        if len(spatial_corr) > 2:
            correlation_coeff = np.corrcoef(spatial_corr, best_psnr_per_ref)[0, 1]
            ax7.text(0.05, 0.95, f'Correlation: {correlation_coeff:.3f}',
                     transform=ax7.transAxes, fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    fig.suptitle('Advanced Wiener Filter - Kernel Analysis', fontsize=16, y=0.95)
    plt.tight_layout()
    plt.show()


def print_summary(results: Dict):
    """Print detailed summary of the analysis results."""
    print(f"\n{'=' * 60}")
    print(f"ADVANCED WIENER FILTER - SUMMARY")
    print(f"{'=' * 60}")

    best_K = results['best_K']
    best_metrics = results['quality_metrics'][best_K]

    print(f"✓ Best Configuration: K = {best_K}")
    print(f"✓ Best PSNR: {best_metrics['psnr']:.2f} dB")
    print(f"✓ SNR: {best_metrics['snr']:.2f} dB")
    print(f"✓ Edge Preservation: {best_metrics['edge_preservation']:.3f}")

    if 'content_similarity_analysis' in results and results['content_similarity_analysis']:
        content_analysis = results['content_similarity_analysis']
        print(f"\n📊 CONTENT SIMILARITY INSIGHTS:")

        performance_ranking = []
        for label, data in content_analysis.items():
            best_psnr = data['best_metrics']['psnr']
            spatial_corr = data['similarity_metrics']['spatial_correlation']
            performance_ranking.append((label, best_psnr, spatial_corr))

        performance_ranking.sort(key=lambda x: x[1], reverse=True)

        print(f"🏆 Best performing reference: {performance_ranking[0][0]} "
              f"(PSNR: {performance_ranking[0][1]:.2f} dB, Correlation: {performance_ranking[0][2]:.4f})")
        print(f"📉 Worst performing reference: {performance_ranking[-1][0]} "
              f"(PSNR: {performance_ranking[-1][1]:.2f} dB, Correlation: {performance_ranking[-1][2]:.4f})")

        if len(performance_ranking) > 2:
            psnr_values = [x[1] for x in performance_ranking]
            corr_values = [x[2] for x in performance_ranking]
            correlation_coeff = np.corrcoef(psnr_values, corr_values)[0, 1]

            print(f"📈 Correlation between spatial similarity and restoration quality: {correlation_coeff:.3f}")

            if correlation_coeff > 0.5:
                print("✅ Strong positive correlation - Content similarity significantly helps!")
            elif correlation_coeff > 0.2:
                print("⚖️ Moderate correlation - Content similarity has some impact")
            else:
                print("❌ Weak correlation - Content similarity has limited impact")