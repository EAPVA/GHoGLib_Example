#include "difference.h"

#include <iostream>

double compare_matrices(cv::Mat m1,
	cv::Mat m2)
{
	double distance = 0.0;
	double magnitude_1 = 0.0;
	double magnitude_2 = 0.0;
	int num_different = 0;
	for(int i = 0; i < m1.rows; ++i)
	{
		for(int j = 0; j < m1.cols; ++j)
		{
			double m1_ij = (double)m1.at< float >(i, j);
			double m2_ij = (double)m2.at< float >(i, j);
			magnitude_1 += (m1_ij * m1_ij);
			magnitude_2 += (m2_ij * m2_ij);
			double difference = fabs(m1_ij - m2_ij);
			distance += difference * difference;
		}
	}
	double sum_magnitudes = sqrt(magnitude_1) + sqrt(magnitude_2);
	return sqrt(distance) / sum_magnitudes;
}
