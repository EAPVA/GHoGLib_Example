/*
 * displayutils.cpp
 *
 *  Created on: Jul 11, 2015
 *      Author: marcelo
 */

#include "displayutils.h"

#include <cmath>
#include "math_constants.h"

#include <iostream>

void display_histograms(cv::Mat& histograms,
	cv::Size hist_grid,
	cv::Size cell_size,
	int num_classes,
	cv::Mat& out)
{
	cv::Scalar GRID_COLOR(0, 128, 0);
	cv::Scalar HIST_COLOR(255, 0, 0);

	for(int i = 0; i < histograms.rows; ++i)
	{
		float* hist_ptr = histograms.ptr< float >(i);
		int center_y = i * cell_size.height + cell_size.height / 2;
		for(int j = 0; j < histograms.cols; ++j)
		{
			float max_value = 0.0f;
			float magnitude = 0.0f;
			int center_x = j * cell_size.width + cell_size.width / 2;
			for(int k = 0; k < num_classes; ++k)
			{
				if(hist_ptr[k] > max_value)
				{
					max_value = hist_ptr[k];
				}
				magnitude += hist_ptr[k] * hist_ptr[k];
			}
			magnitude = sqrt(magnitude) / max_value;
			if (max_value == 0) {
				max_value = 1.0f;
			}
			for(int k = 0; k < num_classes; ++k)
			{
				float hist_mag = 0.9 * hist_ptr[k] / max_value;
				float angle = 2 * CUDART_PI_F * ((float)k / (float)num_classes);
				float hist_x = hist_mag * cos(angle);
				float hist_y = hist_mag * sin(angle);
				hist_x = cell_size.width * hist_x / 2;
				hist_y = cell_size.height * hist_y / 2;
				cv::line(out, cv::Point(center_x, center_y),
					cv::Point(center_x + hist_x, center_y + hist_y),
					HIST_COLOR);
			}
			hist_ptr += num_classes;
		}
	}
}

