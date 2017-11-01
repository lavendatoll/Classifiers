#pragma once
#ifndef IMAGEPROCESS_H
#define IMAGEPROCESS_H

#include <iostream>
#include <algorithm>
#include <fstream>
#include <stdio.h>
#include <time.h>
#include <string>
#include <vector>
#include <map>
#include <set>
using namespace std;

//#include <opencv2/opencv.hpp>
//using namespace cv;

#define IMAGE_SIZE 28
#define FEATURE_NUM 14
#define GRID_SIZE IMAGE_SIZE/FEATURE_NUM
#define LABEL_NUM 10
void read_training_image();
void extract_training_feature(double);
void training_naive_bayesian();
void read_test_image();
void extract_test_feature();
double test_naive_bayesian();
#endif // !IMAGEPROCESS_H

