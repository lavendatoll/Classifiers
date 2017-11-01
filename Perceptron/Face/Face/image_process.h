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
using namespace std;
#define TRAINING_MAX_NUM 30
#define IMAGE_HEIGHT 70
#define IMAGE_WIDTH 60
#define FEATURE_HNUM 14
#define FEATURE_WNUM 12
#define GRID_SIZE IMAGE_HEIGHT/FEATURE_HNUM
#define LABEL_NUM 2
void read_training_image();
void extract_training_feature();
void read_test_image();
void extract_test_feature();
void training_perceptron(double);
double test_perceptron();
#endif // !IMAGEPROCESS_H

