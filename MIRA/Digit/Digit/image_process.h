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
#define IMAGE_SIZE 28
#define FEATURE_NUM 14
#define GRID_SIZE IMAGE_SIZE/FEATURE_NUM
#define LABEL_NUM 10
#define C 0.001
void read_training_image();
void extract_training_feature();
void read_test_image();
void extract_test_feature();
void training_MIRA(double);
double test_MIRA();
#endif // !IMAGEPROCESS_H

