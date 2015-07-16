/*
 * utils.h
 *
 *  Created on: Mar 6, 2015
 *      Author: teider
 */
#ifndef UTILS_H_
#define UTILS_H_

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

void cleanOutputDir(const char *dir);

std::vector<std::string> getImagesList(const char *dir);
cv::Mat generateLabels(std::vector<std::string> file_list);

bool file_exists(std::string filename);
bool is_near(float f1, float f2);

#endif /* UTILS_H_ */
