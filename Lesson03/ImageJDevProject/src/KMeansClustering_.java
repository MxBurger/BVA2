import ij.IJ;
import ij.ImagePlus;
import ij.plugin.filter.PlugInFilter;
import ij.process.ImageProcessor;

import java.util.Vector;

public class KMeansClustering_ implements PlugInFilter {

	public int setup(String arg, ImagePlus imp) {
		if (arg.equals("about"))
		{showAbout(); return DONE;}
		return DOES_RGB;
	} //setup


	public void run(ImageProcessor ip) {
		double[] blackCluster = new double[] {0, 0, 0};
		double[] redCluster = new double[] {255, 0, 0};
		double[] blueCluster = new double[] {0, 0, 255};
		double[] greenCluster = new double[] {0, 255, 0};
		Vector<double[]> clusterCentroides = new Vector<double[]>();
		clusterCentroides.add(blackCluster);
		clusterCentroides.add(redCluster);
		clusterCentroides.add(greenCluster);
		clusterCentroides.add(blueCluster);


		int numOfIterations = 15;

		int width = ip.getWidth();
		int height = ip.getHeight();

		//input image ==> 3D
		int[][][] inImgRGB =ImageJUtility.getChannelImageFromIP(ip, width, height, 3);

		for(int i = 0; i < numOfIterations; i++) {
			System.out.println("cluster update # " + i);
			clusterCentroides = UpdateClusters(inImgRGB, clusterCentroides, width, height);
		}

		int[][][] resImgRGB = new int[width][height][3];

		Vector<int[]> intValRGB = new Vector<>();
		for(double[] dblValRGB: clusterCentroides) {
			intValRGB.add(new int[]{(int)Math.round(dblValRGB[0]),
					(int)Math.round(dblValRGB[1]),
					(int)Math.round(dblValRGB[2])});
		}

		for(int x = 0; x < width; x++) {
			for(int y = 0; y < height; y++) {
				int closestClusterIDX = GetBestClusterIdx(inImgRGB[x][y], clusterCentroides);
				resImgRGB[x][y] = intValRGB.get(closestClusterIDX);
			}
		}

		ImageJUtility.showNewImageRGB(resImgRGB, width, height,
				"final segmented image with centroid colors");

	} //run

	/*
    iterate all pixel and assign them to the cluster showing the smallest distance
    then, for each color centroid, the average color (RGB) gets update
     */
	Vector<double[]> UpdateClusters(int[][][] inRGBimg, Vector<double[]> inClusters, int width, int height) {
		//allocate the data structures
		double[][] newClusterMeanSumArr = new double[inClusters.size()][3]; //for all clusters, the sum for R, G and B
		int[] clusterCountArr = new int[inClusters.size()];

		//process all pixels
		for(int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				int[] currRGB = inRGBimg[x][y];
				int bestClusterIDX = GetBestClusterIdx(currRGB, inClusters);
				clusterCountArr[bestClusterIDX]++;
				newClusterMeanSumArr[bestClusterIDX][0] += currRGB[0];
				newClusterMeanSumArr[bestClusterIDX][1] += currRGB[1];
				newClusterMeanSumArr[bestClusterIDX][2] += currRGB[2];
			}
		}

		//finally calc the new cluster centroids
		Vector<double[]> outClusters = new Vector<>();
		int clusterIDX = 0;
		for(double[] clusterCentroid: inClusters) {
			double[] newClusterColor = newClusterMeanSumArr[clusterIDX];
			int numOfElements = clusterCountArr[clusterIDX];

			if(numOfElements > 0) {
				newClusterColor[0] /= numOfElements;
				newClusterColor[1] /= numOfElements;
				newClusterColor[2] /= numOfElements;
				outClusters.add(newClusterColor);
			} else {
				outClusters.add(clusterCentroid); // fallback if empty, take the old centroid
			}

			clusterIDX++;
		}

		return outClusters;
	}

	double ColorDist(double[] refColor, int[] currColor) {
		double diffR = refColor[0] - currColor[0];
		double diffG = refColor[1] - currColor[1];
		double diffB = refColor[2] - currColor[2];

		double resDist = Math.sqrt(diffR * diffR + diffG * diffG + diffB * diffB);
		return  resDist;
	}

	/*
    returns the cluster IDX showing min distance to input pixel
     */
	int GetBestClusterIdx(int[] rgbArr, Vector<double[]> clusters) {
		double minDist = ColorDist(clusters.get(0), rgbArr);
		int minClusterIDX = 0;

		for(int currClusterIDX  = 1; currClusterIDX  < clusters.size(); currClusterIDX ++) {
			double currDist = ColorDist(clusters.get(currClusterIDX), rgbArr);
			if(currDist < minDist) {
				minDist = currDist;
				minClusterIDX = currClusterIDX;
			}
		}
		return minClusterIDX;
	}


	void showAbout() {
		IJ.showMessage("About KMeansClustering_...",
				"this is a PluginFilter to segment RGB input images in an automated way\n");
	} //showAbout

} //class KMeansClustering_
