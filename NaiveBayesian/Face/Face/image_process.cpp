#include "image_process.h"
vector<int> training_labels;
string *training_image;
vector<int> test_labels;
string *test_image;

map<int,vector<multiset<double>>>training_features;
map<int, vector<map<double, double>>> training_result;
vector<vector<double>> test_features;
void read_training_image()
{

	ifstream file("facedata/facedatatrainlabels");
	int num = 0;
	int count = 0;

	while (!file.eof())
	{
		file >> num;
		training_labels.push_back(num);
		count++;
	}
	file.close();

	file.open("facedata/facedatavalidationlabels");

	while (!file.eof())
	{
		file >> num;
		training_labels.push_back(num);
		count++;
	}
	file.close();

	file.open("facedata/facedatatrain");
	string line;
	training_image = new string[IMAGE_HEIGHT * count];
	int i = 0;
	while (getline(file, line)) // line中不包括每行的换行符
	{
		training_image[i] = line;
		i++;
	}
	file.close();

	file.open("facedata/facedatavalidation");
	while (getline(file, line)) // line中不包括每行的换行符
	{
		training_image[i] = line;
		i++;
	}
	file.close();
}

void extract_training_feature(double percentage)
{
	int image_num = training_labels.size()*percentage;
	for (int image_cnt = 0; image_cnt <image_num; image_cnt++)
	{
		map<int, vector<multiset<double>>>::iterator it;
		it = training_features.find(training_labels[image_cnt]);
		for (int grid_row = 0; grid_row < FEATURE_HNUM; grid_row++)
			for (int grid_col = 0; grid_col < FEATURE_WNUM; grid_col++)
			{
				if (it == training_features.end())
				{
					training_features.insert(pair<int, vector<multiset<double>>>(training_labels[image_cnt], vector<multiset<double>>()));
					it = training_features.find(training_labels[image_cnt]);
					for(int cnt=0;cnt<FEATURE_HNUM*FEATURE_WNUM;cnt++)
						it->second.push_back(multiset<double>());
				}
				double num = 0;
				for (int row_cnt = 0; row_cnt < GRID_SIZE; row_cnt++)
					for (int col_cnt=0; col_cnt < GRID_SIZE; col_cnt++)
						if (training_image[image_cnt*IMAGE_HEIGHT+grid_row * GRID_SIZE + row_cnt][grid_col * GRID_SIZE + col_cnt] == '#')
							num++;
				it->second[grid_row * FEATURE_WNUM + grid_col].insert(num/ (GRID_SIZE*GRID_SIZE));
				
			}

	}
}

void training_naive_bayesian()
{
	for (int label_cnt = 0; label_cnt < LABEL_NUM; label_cnt++)
	{
		for (int cnt = 0; cnt < FEATURE_HNUM*FEATURE_WNUM; cnt++)
			training_result[label_cnt].push_back(map<double, double>());
		for (int feature_cnt = 0; feature_cnt < FEATURE_HNUM*FEATURE_WNUM; feature_cnt++)
		{
			multiset<double>::iterator it;
			for (it = training_features[label_cnt][feature_cnt].begin(); it != training_features[label_cnt][feature_cnt].end(); it++)
				if (training_result[label_cnt][feature_cnt].find(*it) == training_result[label_cnt][feature_cnt].end())
				{
					double num = training_features[label_cnt][feature_cnt].count(*it);
					training_result[label_cnt][feature_cnt].insert(pair<double,double>((*it), num/ training_features[label_cnt][feature_cnt].size()));
				}
		}
	}
}

void read_test_image()
{
	ifstream file("facedata/facedatatestlabels");
	int num = 0;
	int count = 0;

	while (!file.eof())
	{
		file >> num;
		test_labels.push_back(num);
		count++;
	}
	file.close();

	file.open("facedata/facedatatest");
	string line;
	test_image = new string[IMAGE_HEIGHT * count];
	int i = 0;
	while (getline(file, line)) // line中不包括每行的换行符
	{
		test_image[i] = line;
		i++;
	}
	file.close();
}

void extract_test_feature()
{
	int image_num = test_labels.size();
	for (int image_cnt = 0; image_cnt < image_num; image_cnt++)
	{
		vector<double> temp(FEATURE_HNUM*FEATURE_WNUM, 0);
		for (int grid_row = 0; grid_row < FEATURE_HNUM; grid_row++)
			for (int grid_col = 0; grid_col < FEATURE_WNUM; grid_col++)
			{
				double num = 0;
				for (int row_cnt = 0; row_cnt < GRID_SIZE; row_cnt++)
					for (int col_cnt = 0; col_cnt < GRID_SIZE; col_cnt++)
						if (test_image[image_cnt*IMAGE_HEIGHT+grid_row * GRID_SIZE + row_cnt][grid_col * GRID_SIZE + col_cnt] == '#')
							num++;
				temp[grid_row*FEATURE_WNUM + grid_col] = num / (GRID_SIZE*GRID_SIZE);
			}
		test_features.push_back(temp);
	}
}

double test_naive_bayesian()
{
	int image_num = test_labels.size();
	int *result_label = new int[image_num];
	double accuracy = 0;
	for (int image_cnt = 0; image_cnt < image_num; image_cnt++)
	{
		double label[LABEL_NUM];
		for (int label_cnt = 0; label_cnt < LABEL_NUM; label_cnt++)
			label[label_cnt] = ((double)training_features[label_cnt][0].size())/training_labels.size();
		for (int label_cnt = 0; label_cnt < LABEL_NUM; label_cnt++)
		{
			map<double, double>::iterator it;
			for (int feature_cnt = 0; feature_cnt < FEATURE_HNUM*FEATURE_WNUM; feature_cnt++)
			{
				it = training_result[label_cnt][feature_cnt].find(test_features[image_cnt][feature_cnt]);
				if (it != training_result[label_cnt][feature_cnt].end())
					label[label_cnt] *= it->second;
				else
					label[label_cnt] *= 0.0001;
			}
		}


		double max = label[0];
		result_label[image_cnt] = 0;
		for (int label_cnt = 1; label_cnt < LABEL_NUM; label_cnt++)
		{
			if (label[label_cnt] > max)
			{
				max = label[label_cnt];
				result_label[image_cnt] = label_cnt;
			}
		}
	}
	for (int cnt = 0; cnt < image_num; cnt++)
		if (result_label[cnt] == test_labels[cnt])
			accuracy++;
	accuracy /= image_num;
	delete[] result_label;
	return accuracy;

}

