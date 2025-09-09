🏦 Bank Marketing Campaign Prediction
🎯 1. Project Overview
This project focuses on building a machine learning model to predict whether a customer will subscribe to a fixed-term deposit at a bank. The model is a Decision Tree Classifier trained on a dataset from a Portuguese banking institution's direct marketing campaigns.

The primary goal is to help the bank target potential customers more effectively, thereby increasing the campaign's success rate and reducing marketing costs.

📊 2. The Dataset
The data is the Bank Marketing Dataset from the UCI Machine Learning Repository. It contains demographic and behavioral data for over 45,000 clients based on real-world telemarketing campaigns.

Source: UCI Machine Learning Repository: Bank Marketing Data Set

File Used: bank-full.csv (contains all 45,211 examples).

Citation: Moro et al., 2014. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31.

🛠️ 3. Requirements
To run this project, you will need Python 3 and the following libraries:

pandas

scikit-learn

You can install these dependencies with a single command:

pip install pandas scikit-learn

🚀 4. How to Use
To get this project running on your local machine, follow these simple steps.

Clone the Repository:

git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)

Navigate to the Directory:

cd your-repository-name

Place the Dataset: Ensure the bank-full.csv file is in the project's root directory.

Execute the Script:

python your_script_name.py

(Remember to replace your_script_name.py with the actual name of your Python file.)

📈 5. Model Performance and Results
The script trains a Decision Tree Classifier and evaluates its performance on a 20% test split of the data.

Overall Accuracy: 87.40%

Classification Report
Class

Precision

Recall

F1-Score

Support

0 (No)

0.93

0.93

0.93

7952

1 (Yes)

0.48

0.48

0.48

1091

Accuracy





0.87

9043

Macro Avg

0.70

0.70

0.70

9043

Weighted Avg

0.87

0.87

0.87

9043

Analysis of Results
The model achieves a high overall accuracy, but this number is misleading. The classification report reveals a critical weakness:

Excellent at Predicting 'No': The model is very reliable at identifying customers who will not subscribe (Class 0), with precision and recall at 93%.

Poor at Predicting 'Yes': The model is very unreliable at identifying customers who will subscribe (Class 1). Its precision and recall of 48% mean its predictions for this crucial group are no better than a coin toss.

This is a classic case of class imbalance. The model has learned that it can be "accurate" by simply predicting the majority class ('No') most of the time, failing the main business goal of finding interested customers.

💡 6. Conclusion and Next Steps
The initial Decision Tree model provides a good baseline but is not suitable for the primary business objective. Future work should focus on improving the model's ability to predict the minority class ('Yes').

Potential improvements include:

Handling Class Imbalance: Implementing techniques like SMOTE (Synthetic Minority Over-sampling Technique) or using class_weight in the model.

Feature Engineering: Creating new, more predictive features from the existing data.

Trying Different Models: Experimenting with ensemble algorithms like Random Forest or Gradient Boosting, which often perform better on imbalanced datasets.

🤝 7. How to Contribute
Contributions are welcome! If you have suggestions for improving the model or the code, please feel free to:

Fork the repository.

Create a new branch (git checkout -b feature/AmazingImprovement).

Commit your changes (git commit -m 'Add some AmazingImprovement').

Push to the branch (git push origin feature/AmazingImprovement).

Open a Pull Request.