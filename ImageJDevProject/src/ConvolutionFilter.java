

public class ConvolutionFilter {

	
	/**
	 * convolution of input image with kernel, normalization to kernel sum 1.0
	 * only use for low-pass filters
	 * 
	 * @param inputImg
	 * @param width
	 * @param height
	 * @param kernel double[][] kernel image
	 * @param radius kernel radius
	 * @return
	 */
	public static double[][] convolveDoubleNorm(double[][] inputImg, int width, int height, double[][] kernel, int radius) {
		double[][] returnImg = new double[width][height];
		
		//TODO: implementation required
		
		return returnImg;
	}
	
	/**
	 * convolution of input image with kernel
	 * 
	 * @param inputImg
	 * @param width
	 * @param height
	 * @param kernel double[][] kernel image
	 * @param radius kernel radius
	 * @return
	 */
	public static double[][] convolveDouble(double[][] inputImg, int width, int height, double[][] kernel, int radius) {
		double[][] returnImg = new double[width][height];
		
		//TODO: implementation required
		
		return returnImg;
	}
	
	/**
	 * returns kernel image according to specified radius for mean low-pass filtering
	 * 
	 * @param tgtRadius
	 * @return
	 */
	public static double[][] getMeanMask(int tgtRadius) {
		int size = 2 * tgtRadius + 1;
		double[][] kernelImg = new double[size][size];
		
		//TODO: implementation required
		
		return kernelImg;
	}
		
}
