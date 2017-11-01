#include "image_process.h"

extern string *training_image;
extern string *test_image;
void main(int argc, const char * argv[]) 
{
	cout << "Please Input the Percentage of Training Data You Want To Use: ";
	double percentage = 1;
	cin >> percentage;
	cin.ignore(1024, '\n');
	while (cin.fail() || percentage <= 0)
	{
		cout << "Your Input is Invaild. Please Re-enter:" << endl;
		cin >> percentage;
	}
	cout << "Reading Training Data..." << endl;
	read_training_image();
	extract_training_feature();
	cout << "Reading Test Data..." << endl;
	read_test_image();
	extract_test_feature();
	cout << "Training Classifier..." << endl;
	training_MIRA(percentage);
	cout << "Testing and Calculating Accuracy..." << endl;
	double accuracy = test_MIRA();
	cout << "Accuracy is: " << accuracy << endl;
	delete[] training_image;
	delete[] test_image;
	system("Pause");
}
