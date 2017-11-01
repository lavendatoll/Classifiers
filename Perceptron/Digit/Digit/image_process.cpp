#include "image_process.h"
vector<int> training_labels;
vector<int> test_labels;
string *training_image;
string *test_image;
vector<vector<double>> training_features;
vector<vector<double>> test_features;
vector<int> result;
double weight[LABEL_NUM][FEATURE_NUM*FEATURE_NUM];
void read_training_image()
{

	ifstream file("digitdata/traininglabels");
	int num = 0;
	int count = 0;

	while (!file.eof())
	{
		file >> num;
		training_labels.push_back(num);
		count++;
	}
	file.close();

	file.open("digitdata/validationlabels");

	while (!file.eof())
	{
		file >> num;
		training_labels.push_back(num);
		count++;
	}
	file.close();

	file.open("digitdata/trainingimages");
	string line;
	training_image = new string[IMAGE_SIZE * count];
	int i = 0;
	while (getline(file, line)) 
	{
		training_image[i] = line;
		i++;
	}
	file.close();

	file.open("digitdata/validationimages");
	while (getline(file, line))
	{
		training_image[i] = line;
		i++;
	}
	file.close();
}

void extract_training_feature()
{
	int training_data_num = training_labels.size();
	for (int image_cnt = 0; image_cnt < training_data_num; image_cnt++)
	{
		vector<double> temp(FEATURE_NUM*FEATURE_NUM,0);
		for (int grid_row = 0; grid_row < FEATURE_NUM; grid_row++)
			for (int grid_col = 0; grid_col < FEATURE_NUM; grid_col++)
			{
				double num = 0;
				for (int row_cnt = 0; row_cnt < GRID_SIZE; row_cnt++)
					for (int col_cnt=0; col_cnt < GRID_SIZE; col_cnt++)
						if (training_image[image_cnt*IMAGE_SIZE + grid_row * GRID_SIZE + row_cnt][grid_col * GRID_SIZE + col_cnt] == '#')
							num++;
						else if (training_image[image_cnt*IMAGE_SIZE + grid_row * GRID_SIZE + row_cnt][grid_col * GRID_SIZE + col_cnt] == '+')
							num+=0.1;
				temp[grid_row * FEATURE_NUM + grid_col] = num/(GRID_SIZE*GRID_SIZE);
			}
		training_features.push_back(temp);
	}
}


void read_test_image()
{

	ifstream file("digitdata/testlabels");
	int num = 0;
	int count = 0;

	while (!file.eof())
	{
		file >> num;
		test_labels.push_back(num);
		count++;
	}
	file.close();

	file.open("digitdata/testimages");
	string line;
	test_image = new string[IMAGE_SIZE * count];
	int i = 0;
	while (getline(file, line))
	{
		test_image[i] = line;
		i++;
	}
	file.close();

}

void extract_test_feature()
{
	int test_data_num = test_labels.size();
	for (int image_cnt = 0; image_cnt < test_data_num; image_cnt++)
	{
		vector<double> temp(FEATURE_NUM*FEATURE_NUM);
		for (int grid_row = 0; grid_row < FEATURE_NUM; grid_row++)
			for (int grid_col = 0; grid_col < FEATURE_NUM; grid_col++)
			{
				double num = 0;
				for (int row_cnt = 0; row_cnt < GRID_SIZE; row_cnt++)
					for (int col_cnt = 0; col_cnt < GRID_SIZE; col_cnt++)
						if (test_image[image_cnt*IMAGE_SIZE + grid_row * GRID_SIZE + row_cnt][grid_col * GRID_SIZE + col_cnt] == '#')
							num++;
						else if (test_image[image_cnt*IMAGE_SIZE + grid_row * GRID_SIZE + row_cnt][grid_col * GRID_SIZE + col_cnt] == '+')
							num += 0.1;
				temp[grid_row * FEATURE_NUM + grid_col] = num / (GRID_SIZE*GRID_SIZE);
			}
		test_features.push_back(temp);
	}
}

void training_perceptron(double percentage)
{
	bool convergence = true;
	int training_num = 0;
	int training_data_num = training_labels.size()*percentage;
	do
	{
		convergence = true;
		for (int image_cnt = 0; image_cnt < training_data_num; image_cnt++)
		{
			double score[LABEL_NUM] = { 0 };
			for (int label_cnt = 0; label_cnt < LABEL_NUM; label_cnt++)
				for (int feature_cnt = 0; feature_cnt < FEATURE_NUM*FEATURE_NUM; feature_cnt++)
					score[label_cnt] += weight[label_cnt][feature_cnt] * training_features[image_cnt][feature_cnt];
			double max = score[0];
			int max_index = 0;
			for (int label_cnt = 0; label_cnt < LABEL_NUM; label_cnt++)
				if (score[label_cnt] > max)
				{
					max = score[label_cnt];
					max_index = label_cnt;
				}
			if (max_index != training_labels[image_cnt])
			{
				for (int feature_cnt = 0; feature_cnt < FEATURE_NUM*FEATURE_NUM; feature_cnt++)
				{
					weight[max_index][feature_cnt] -= training_features[image_cnt][feature_cnt];
					weight[training_labels[image_cnt]][feature_cnt] += training_features[image_cnt][feature_cnt];
				}
				convergence = false;
			}
		}
		training_num++;
	} while (!convergence && training_num < TRAINING_MAX_NUM);
}

double test_perceptron()
{
	double accuracy = 0;
	int test_data_num = test_labels.size();
	for (int image_cnt = 0; image_cnt < test_data_num; image_cnt++)
	{
		double score[LABEL_NUM] = { 0 };
		for (int label_cnt = 0; label_cnt < LABEL_NUM; label_cnt++)
			for (int feature_cnt = 0; feature_cnt < FEATURE_NUM*FEATURE_NUM; feature_cnt++)
				score[label_cnt] += weight[label_cnt][feature_cnt] * test_features[image_cnt][feature_cnt];
		double max = score[0];
		int max_index = 0;
		for(int label_cnt = 0; label_cnt < LABEL_NUM; label_cnt++)
			if (score[label_cnt] > max)
			{
				max = score[label_cnt];
				max_index = label_cnt;
			}
		if (max_index == test_labels[image_cnt])
			accuracy++;
	}
	accuracy /= test_labels.size();
	return accuracy;
}
