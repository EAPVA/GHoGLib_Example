#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

#include "utils.h"
#include "difference.h"

#include <include/HogDescriptor.h>
#include <include/SVMClassifier.h>

int main(int argc,
	char** argv)
{
	std::vector< std::string > file_list = getImagesList("resources/images");
	cv::Mat train_data_ghoglib;
	cv::Mat expected_outputs;
	cv::Mat img;
	cv::Mat grad_mag;
	cv::Mat grad_phase;
	cv::Mat descriptor;
	cv::Size detection_window(180, 240);
	cv::Size cell_size(6, 6);
	cv::Size block_size(12, 12);
	cv::Size block_stride(6, 6);
	int num_bins = 9;
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
	ghog_lib.alloc_buffer(cv::Size(descriptor_dim, 1), CV_32FC1, descriptor, 0);
	train_data_ghoglib.create(file_list.size(), descriptor_dim, CV_32FC1);

	for(int i = 0; i < file_list.size(); ++i)
	{
		std::cout << "Reading image " << file_list[i] << std::endl;
		cv::imread(file_list[i], CV_LOAD_IMAGE_COLOR).convertTo(img, CV_32FC3);
		ghog_lib.image_normalization_sync(img);
		ghog_lib.calc_gradient_sync(img, grad_mag, grad_phase);
		ghog_lib.create_descriptor_sync(grad_mag, grad_phase, descriptor);
		descriptor.copyTo(train_data_ghoglib.row(i));
		img = cv::imread(file_list[i], CV_LOAD_IMAGE_COLOR);
		std::vector< float > temp;
		hog_opencv.compute(img, temp);
		for(int j = 0; j < temp.size(); ++j)
		{
			train_data_opencv.at< float >(i, j) = temp[j];
		}
//		double aux = compare_matrices(train_data_ghoglib.row(i),
//			train_data_opencv.row(i));
//		std::cout << "A diferença entre os descritores é de :" << aux
//			<< std::endl;
	}
	expected_outputs = generateLabels(file_list);

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

	return 0;
}
