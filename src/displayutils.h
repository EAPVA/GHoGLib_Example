/*
 * displayutils.h
 *
 *  Created on: Jul 11, 2015
 *      Author: marcelo
 */

#ifndef DISPLAYUTILS_H_
#define DISPLAYUTILS_H_

#include <opencv2/core/core.hpp>

void display_histograms(cv::Mat& histograms,
	cv::Size hist_grid,
	cv::Size cell_size,
	int num_classes,
	cv::Mat& out);

#endif /* DISPLAYUTILS_H_ */
