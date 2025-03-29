
public class ImageTransformationFilter {

	/**
	 * apply scalar transformation
	 * 
	 * @param inImg
	 * @param width
	 * @param height
	 * @param transferFunction
	 * @return
	 */
	public static int[][] getTransformedImage(int[][] inImg, int width, int height, int[] transferFunction) {
		int[][] returnImg = new int[width][height];
		
		//TODO implementation required
		
		return returnImg;
	}
	
	/**
	 * get transfer function for contrast inversion 
	 * 
	 * @param maxVal
	 * @return
	 */
	public static int[] getInversionTF(int maxVal) {
		int[] transferFunction = new int[maxVal + 1];
		

		//TODO implementation required
		
		return transferFunction;
	}

	
}
