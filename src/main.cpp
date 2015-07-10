#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

#include "utils.h"

#include <include/HogDescriptor.h>
#include <include/SVMClassifier.h>

int main(int argc,
	char** argv)
{
	std::vector< std::string > file_list = getImagesList("resources/images");
	cv::Mat train_data;
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
	ghog::lib::HogDescriptor hog("hog.xml");
	int descriptor_dim = hog.get_descriptor_size();
	ghog::lib::IClassifier* classifier = new ghog::lib::SVMClassifier();

	hog.alloc_buffer(detection_window, CV_32FC3, img);
	hog.alloc_buffer(detection_window, CV_32FC1, grad_mag);
	hog.alloc_buffer(detection_window, CV_32FC1, grad_phase);
	hog.alloc_buffer(cv::Size(descriptor_dim, 1), CV_32FC1, descriptor);
	train_data.create(file_list.size(), descriptor_dim, CV_32FC1);

	for(int i = 0; i < file_list.size(); ++i)
	{
		std::cout << "Reading image " << file_list[i] << std::endl;
		cv::imread(file_list[i], CV_LOAD_IMAGE_COLOR).convertTo(img, CV_32FC3);
		hog.image_normalization_sync(img);
		hog.calc_gradient_sync(img, grad_mag, grad_phase);
		hog.create_descriptor_sync(grad_mag, grad_phase, descriptor);
		descriptor.copyTo(train_data.row(i));
	}
	expected_outputs = generateLabels(file_list);

	CvSVM svm;
	CvSVMParams svm_params;

	std::cout << "Training..." << std::endl;

	svm.train_auto(train_data, expected_outputs, cv::Mat(), cv::Mat(),
		svm_params, 5);

	for(int i = 0; i < train_data.rows; ++i)
	{
		float result = svm.predict(train_data.row(i));
		if(result != expected_outputs.at< float >(i))
		{
			std::cout << "Erro na imagem " << file_list[i] << ". Esperado: "
				<< expected_outputs.at< float >(i) << "  Obtido:" << result
				<< std::endl;
		} else
		{
			std::cout << "Acertou imagem " << file_list[i] << std::endl;
		}
	}

	return 0;
}
