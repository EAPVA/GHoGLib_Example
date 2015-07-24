#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "utils.h"
#include "difference.h"
#include "displayutils.h"

#include <include/HogDescriptor.h>
#include <include/SVMClassifier.h>
#include <include/Utils.h>

using namespace cv;
using namespace std;

// HOGDescriptor visual_imagealizer
// adapted for arbitrary size of feature sets and training images
Mat get_hogdescriptor_visual_image(Mat& origImg,
	vector< float >& descriptorValues,
	Size winSize,
	Size cellSize,
	int scaleFactor,
	double viz_factor)
{
	Mat visual_image;
	resize(origImg, visual_image,
		Size(origImg.cols * scaleFactor, origImg.rows * scaleFactor));

	int gradientBinSize = 9;
	// dividing 180� into 9 bins, how large (in rad) is one bin?
	float radRangeForOneBin = 3.14 / (float)gradientBinSize;

	// prepare data structure: 9 orientation / gradient strenghts for each cell
	int cells_in_x_dir = winSize.width / cellSize.width;
	int cells_in_y_dir = winSize.height / cellSize.height;
	int totalnrofcells = cells_in_x_dir * cells_in_y_dir;
	float*** gradientStrengths = new float**[cells_in_y_dir];
	int** cellUpdateCounter = new int*[cells_in_y_dir];
	for(int y = 0; y < cells_in_y_dir; y++)
	{
		gradientStrengths[y] = new float*[cells_in_x_dir];
		cellUpdateCounter[y] = new int[cells_in_x_dir];
		for(int x = 0; x < cells_in_x_dir; x++)
		{
			gradientStrengths[y][x] = new float[gradientBinSize];
			cellUpdateCounter[y][x] = 0;

			for(int bin = 0; bin < gradientBinSize; bin++)
				gradientStrengths[y][x][bin] = 0.0;
		}
	}

	// nr of blocks = nr of cells - 1
	// since there is a new block on each cell (overlapping blocks!) but the last one
	int blocks_in_x_dir = cells_in_x_dir - 1;
	int blocks_in_y_dir = cells_in_y_dir - 1;

	// compute gradient strengths per cell
	int descriptorDataIdx = 0;
	int cellx = 0;
	int celly = 0;

	for(int blockx = 0; blockx < blocks_in_x_dir; blockx++)
	{
		for(int blocky = 0; blocky < blocks_in_y_dir; blocky++)
		{
			// 4 cells per block ...
			for(int cellNr = 0; cellNr < 4; cellNr++)
			{
				// compute corresponding cell nr
				int cellx = blockx;
				int celly = blocky;
				if(cellNr == 1)
					celly++;
				if(cellNr == 2)
					cellx++;
				if(cellNr == 3)
				{
					cellx++;
					celly++;
				}

				for(int bin = 0; bin < gradientBinSize; bin++)
				{
					float gradientStrength = descriptorValues[descriptorDataIdx];
					descriptorDataIdx++;

					gradientStrengths[celly][cellx][bin] += gradientStrength;

				} // for (all bins)

				// note: overlapping blocks lead to multiple updates of this sum!
				// we therefore keep track how often a cell was updated,
				// to compute average gradient strengths
				cellUpdateCounter[celly][cellx]++;

			} // for (all cells)

		} // for (all block x pos)
	} // for (all block y pos)

	// compute average gradient strengths
	for(int celly = 0; celly < cells_in_y_dir; celly++)
	{
		for(int cellx = 0; cellx < cells_in_x_dir; cellx++)
		{

			float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

			// compute average gradient strenghts for each gradient bin direction
			for(int bin = 0; bin < gradientBinSize; bin++)
			{
				gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
			}
		}
	}

	cout << "descriptorDataIdx = " << descriptorDataIdx << endl;

	// draw cells
	for(int celly = 0; celly < cells_in_y_dir; celly++)
	{
		for(int cellx = 0; cellx < cells_in_x_dir; cellx++)
		{
			int drawX = cellx * cellSize.width;
			int drawY = celly * cellSize.height;

			int mx = drawX + cellSize.width / 2;
			int my = drawY + cellSize.height / 2;

			rectangle(visual_image,
				Point(drawX * scaleFactor, drawY * scaleFactor),
				Point((drawX + cellSize.width) * scaleFactor,
					(drawY + cellSize.height) * scaleFactor),
				CV_RGB(100, 100, 100), 1);

			// draw in each cell all 9 gradient strengths
			for(int bin = 0; bin < gradientBinSize; bin++)
			{
				float currentGradStrength = gradientStrengths[celly][cellx][bin];

				// no line to draw?
				if(currentGradStrength == 0)
					continue;

				float currRad = bin * radRangeForOneBin + radRangeForOneBin / 2;

				float dirVecX = cos(currRad);
				float dirVecY = sin(currRad);
				float maxVecLen = cellSize.width / 2;
				float scale = viz_factor; // just a visual_imagealization scale,
				// to see the lines better

				// compute line coordinates
				float x1 = mx
					- dirVecX * currentGradStrength * maxVecLen * scale;
				float y1 = my
					- dirVecY * currentGradStrength * maxVecLen * scale;
				float x2 = mx
					+ dirVecX * currentGradStrength * maxVecLen * scale;
				float y2 = my
					+ dirVecY * currentGradStrength * maxVecLen * scale;

				// draw gradient visual_imagealization
				line(visual_image, Point(x1 * scaleFactor, y1 * scaleFactor),
					Point(x2 * scaleFactor, y2 * scaleFactor),
					CV_RGB(0, 0, 255), 1);

			} // for (all bins)

		} // for (cellx)
	} // for (celly)

	// don't forget to free memory allocated by helper data structures!
	for(int y = 0; y < cells_in_y_dir; y++)
	{
		for(int x = 0; x < cells_in_x_dir; x++)
		{
			delete[] gradientStrengths[y][x];
		}
		delete[] gradientStrengths[y];
		delete[] cellUpdateCounter[y];
	}
	delete[] gradientStrengths;
	delete[] cellUpdateCounter;

	return visual_image;
}

int main(int argc,
	char** argv)
{
	std::vector< std::string > file_list = getImagesList("resources/images");
	cv::Mat train_data_ghoglib;
	cv::Mat expected_outputs;
	cv::Mat img;
	cv::Mat grad_mag;
	cv::Mat grad_phase;
	cv::Mat histograms;
	cv::Mat descriptor;
	cv::Size detection_window(180, 240);
	cv::Size cell_size(12, 12);
	cv::Size block_size(24, 24);
	cv::Size block_stride(12, 12);
	cv::Size histogram_grid = ghog::lib::Utils::partition(detection_window,
		cell_size);
	std::cout << "Histogram grid: " << histogram_grid << std::endl;
	int num_bins = 9;
	std::vector< float > descriptorValue;

	ghog::lib::HogDescriptor ghog_lib("hog.xml");
	int descriptor_dim = ghog_lib.get_descriptor_size();
	ghog::lib::IClassifier* classifier = new ghog::lib::SVMClassifier();

	cv::HOGDescriptor hog_opencv(detection_window, block_size, block_stride,
		cell_size, num_bins);
	CvSVM svm;
	CvSVMParams svm_params;
	cv::Mat train_data_opencv(file_list.size(), hog_opencv.getDescriptorSize(),
		CV_32FC1);

	ghog_lib.alloc_buffer(detection_window, CV_32FC3, img, 1);
	ghog_lib.alloc_buffer(detection_window, CV_32FC1, grad_mag, 0);
	ghog_lib.alloc_buffer(detection_window, CV_32FC1, grad_phase, 0);
	ghog_lib.alloc_buffer(histogram_grid, CV_32FC(9), histograms, 0);
	ghog_lib.alloc_buffer(cv::Size(descriptor_dim, 1), CV_32FC1, descriptor, 0);
	train_data_ghoglib.create(file_list.size(), descriptor_dim, CV_32FC1);

	for(int i = 0; i < file_list.size(); ++i)
	{
		std::cout << "Reading image " << file_list[i] << std::endl;
		cv::imread(file_list[i], CV_LOAD_IMAGE_COLOR).convertTo(img, CV_32FC3);
		ghog_lib.image_normalization_sync(img);
		ghog_lib.calc_gradient_sync(img, grad_mag, grad_phase);
		ghog_lib.create_descriptor_sync(grad_mag, grad_phase, descriptor,
			histograms);
		descriptor.copyTo(train_data_ghoglib.row(i));
		img = cv::imread(file_list[i], CV_LOAD_IMAGE_COLOR);
		std::vector< float > temp;
		std::vector< float > temp_lib;
		hog_opencv.compute(img, temp);
		for(int j = 0; j < temp.size(); ++j)
		{
			//std::cout << "train_data_opencv " << std::endl;
			train_data_opencv.at< float >(i, j) = temp[j];
		}

		for(int aux1 = 0; aux1 < descriptor.rows; aux1++)
		{
			for(int aux2 = 0; aux2 < descriptor.cols; aux2++)
			{
				temp_lib.push_back(descriptor.at< float >(aux1, aux2));

			}
		}

//		double aux = compare_matrices(train_data_ghoglib.row(i),
//			train_data_opencv.row(i));
//		std::cout << "A diferença entre os descritores é de :" << aux
//			<< std::endl;

		// Gera imagem do descrito para o opencv
		Mat result = get_hogdescriptor_visual_image(img, temp, Size(180, 240),
			Size(12, 12), 1, 3);

//		char path[100] = "resources/images/desc_";
		char path[100] = "";
		strcat(path, file_list[i].c_str());
		strcat(path, "_desc_opencv.png");
		imwrite(path, result);

		// gera imagem do descrito para a lib GHoG
		Mat result2 = get_hogdescriptor_visual_image(img, temp_lib,
			Size(180, 240), Size(12, 12), 1, 3);

		//char path[100] = "resources/images/desc_";
		char path2[100] = "";
		strcat(path2, file_list[i].c_str());
		strcat(path2, "_desc_GHoGLib.png");
		imwrite(path2, result2);

		char path_norm[100] = "";
		strcat(path_norm, file_list[i].c_str());
		strcat(path_norm, "_norm_GHoGLib.png");
		imwrite(path_norm, img);

		char path_mag[100] = "";
		strcat(path_mag, file_list[i].c_str());
		strcat(path_mag, "_mag_GHoGLib.png");
		imwrite(path_mag, grad_mag * 256.0f);

		char path_phase[100] = "";
		strcat(path_phase, file_list[i].c_str());
		strcat(path_phase, "_phase_GHoGLib.png");
		imwrite(path_phase, grad_phase * 256.0f);

		cv::Mat result_hists;
		img.copyTo(result_hists);
		display_histograms(histograms, histogram_grid, cell_size, num_bins,
			result_hists);

		char path_hists[100] = "";
		strcat(path_hists, file_list[i].c_str());
		strcat(path_hists, "_hists_GHoGLib.png");
		imwrite(path_hists, result_hists);

	}

	/*expected_outputs = generateLabels(file_list);

	 std::cout << "Training GHogLib..." << std::endl;
	 classifier->train_sync(train_data_ghoglib, expected_outputs);

	 std::cout << "Training OpenCV..." << std::endl;
	 svm.train_auto(train_data_opencv, expected_outputs, cv::Mat(), cv::Mat(),
	 svm_params, 5);

	 int num_errors_opencv = 0;
	 int num_errors_ghoglib = 0;

	 for(int i = 0; i < train_data_ghoglib.rows; ++i)
	 {
	 float result = classifier->classify_sync(train_data_ghoglib.row(i))
	 .at< float >(0);
	 if(result != expected_outputs.at< float >(i))
	 {
	 num_errors_ghoglib++;
	 std::cout << "GHOGLIB: Erro na imagem " << file_list[i]
	 << ". Esperado: " << expected_outputs.at< float >(i)
	 << "  Obtido:" << result << std::endl;
	 }
	 result = svm.predict(train_data_opencv.row(i));
	 if(result != expected_outputs.at< float >(i))
	 {
	 num_errors_opencv++;
	 std::cout << "OPENCV: Erro na imagem " << file_list[i]
	 << ". Esperado: " << expected_outputs.at< float >(i)
	 << "  Obtido:" << result << std::endl;
	 }
	 }

	 std::cout << "O OpenCV errou " << num_errors_opencv << " de "
	 << file_list.size() << " casos de teste. ("
	 << 100 * (float)num_errors_opencv / (float)file_list.size() << "\%)"
	 << std::endl;
	 std::cout << "A GHogLib errou " << num_errors_ghoglib << " de "
	 << file_list.size() << " casos de teste. ("
	 << 100 * (float)num_errors_ghoglib / (float)file_list.size() << "\%)"
	 << std::endl;

	 std::cout << "Execution finished." << std::endl;
	 */

	return 0;
}
