# Bird-Strike-Cost-Prediction-Project
This is my  Machine Learning Project for the Master Program at University of South Florida

# Objective

This project focuses on predicting the cost of bird strikes on aircraft by analyzing various incident attributes. Accurate prediction of bird strike costs is crucial for airlines to assess potential damage and make informed decisions immediately following an incident. The project utilizes machine learning models and advanced data preprocessing techniques to achieve this predictive goal, focusing on regression techniques to estimate the `Total Cost` associated with each bird strike incident.

## Research Questions and Answers

**Q1. Can we accurately predict the total cost of a bird strike incident using incident attributes such as the number of engines, number of objects struck, and the description of the incident?**

**Q2. How does the "Engine to Object Ratio" influence the prediction of bird strike costs, and what is its significance in the model?**

**Q3. How do different machine learning models compare in terms of predictive accuracy for estimating the total cost of bird strikes?**

## Data Description

The dataset used in this project, `airline.csv`, includes detailed information about bird strike incidents on aircraft. The following variables are included:

- **Aircraft**: Specifies the type of aircraft involved in the incident (Categorical).
- **Number_Objects**: Indicates the number of objects (e.g., birds) that struck the aircraft (Numerical).
- **Engines**: Specifies the number of engines on the aircraft (Numerical).
- **Airline**: Identifies the airline or operator of the aircraft (Categorical).
- **Origin State**: The state where the incident originated (Categorical).
- **Phase**: Phase of flight during the incident (Categorical).
- **Description**: A textual description of the incident (Text).
- **Object Size**: Specifies the size of the object (e.g., bird) that struck the aircraft (Categorical).
- **Weather**: Describes the sky condition during the incident, such as "No Cloud" or "Some Cloud" (Categorical).
- **Warning**: A binary value (YES/NO) indicating whether the pilot was warned about the object before the incident (Binary).
- **Altitude**: Records the altitude of the aircraft during the incident (Numerical).
- **Total Cost**: The natural log of the total cost associated with the incident, which serves as the target variable (Numerical).

## Methodology

The methodology for this project involves several key steps to ensure the development of accurate and robust predictive models:

1. **Data Preprocessing**:
   - **Handling Missing Values**: Missing values in numerical columns were handled using `SimpleImputer` with a median strategy, while categorical missing values were imputed with the most frequent category.
   - **Feature Scaling**: Numerical features were standardized using `StandardScaler` to ensure that all features contribute equally to the model.
   - **Categorical Encoding**: Categorical variables were converted to a suitable format for machine learning models using `OneHotEncoder`.
   - **Text Processing**: The `Description` text field was vectorized using `TfidfVectorizer` to capture the importance of words, followed by dimensionality reduction using `TruncatedSVD`.

2. **Feature Engineering**:
   - **Engine to Object Ratio**: A new feature was created to represent the severity of the incident by calculating the ratio of the number of engines to the number of objects struck. This ratio is critical in determining the potential damage from a bird strike.

3. **Model Development**:
   - **Baseline Model**: A Dummy Regressor was used as a baseline to compare the performance of more complex models.
   - **Decision Tree**: A Decision Tree Regressor was implemented and optimized for better predictive accuracy.
   - **Voting Regressor**: An ensemble model combining multiple algorithms (e.g., Decision Tree, SVR, SGDRegressor) was developed to improve performance.
   - **Gradient Boosting**: A Gradient Boosting model was used to capture complex relationships within the data.
   - **Neural Network**: A Multi-Layer Perceptron (MLP) was implemented to leverage the power of deep learning in regression tasks.
   - **Grid Search**: Hyperparameter tuning was performed using Grid Search to identify the best model configuration.

4. **Model Evaluation**:
   - Models were evaluated using Root Mean Squared Error (RMSE) on both the training and test datasets to ensure they generalize well to unseen data.
   - Techniques like early stopping were employed to prevent overfitting, particularly in models like Gradient Boosting and Neural Networks.

## Advanced Techniques and Libraries

This project leverages several advanced libraries and techniques to enhance the efficiency and accuracy of the machine learning pipeline:

### **1. ColumnTransformer and Pipeline**
- **ColumnTransformer**: Used to apply different preprocessing techniques to different subsets of features within the dataset. This allows for a highly customized and efficient preprocessing pipeline.
- **Pipeline**: Facilitates the construction of a streamlined ETL (Extract, Transform, Load) process. By chaining together data transformation steps, the pipeline ensures that data flows smoothly from raw input to model-ready features.

### **2. Data Imputation and Scaling**
- **SimpleImputer**: Handles missing values in both numerical and categorical data, ensuring that the dataset is complete and ready for modeling.
- **StandardScaler**: Standardizes numerical features to have a mean of 0 and a standard deviation of 1, which is essential for models sensitive to feature scaling.

### **3. Categorical Encoding**
- **OneHotEncoder**: Converts categorical variables into a format that can be provided to ML algorithms to improve their performance. It creates binary columns for each category and removes any issues related to ordinal nature.

### **4. Custom Feature Engineering**
- **FunctionTransformer**: Allows for custom transformations within the pipeline. In this project, it was used to engineer the "Engine to Object Ratio," a critical feature that captures the severity of bird strikes.

### **5. Text Feature Extraction and Dimensionality Reduction**
- **TfidfVectorizer**: Transforms text data (incident descriptions) into numerical features that capture the importance of words within the text, allowing the model to understand the context of each incident.
- **TruncatedSVD**: Performs dimensionality reduction on the text features, reducing the complexity of the model and helping prevent overfitting.

## Key Features and Insights

### **Feature Engineering: "Engine to Object Ratio"**

- **Concept**: The "Engine to Object Ratio" was engineered to represent the number of impacts per engine during a bird strike incident. This feature was designed to capture the severity of an incident, with higher ratios indicating potentially higher damage and costs.

- **Significance**: This new feature played a crucial role in improving the accuracy of the predictive models. Its inclusion allowed the models to better understand the relationship between the number of engines, objects struck, and the total cost.

### **Model Performance Summary**

- **Baseline Model (Dummy Regressor)**: 
  - Test RMSE = 482,867.50

- **Decision Tree**: 
  - Initial Model (max_depth=5): Test RMSE = 629,822.63
  - Optimized Model (min_samples_leaf=26, max_depth=20): Test RMSE = 473,840.15

- **Voting Regressor**: 
  - Initial Model: Test RMSE = 512,367.84
  - Optimized Model: Test RMSE = 469,450.16 (Best Performing Model)

- **Gradient Boosting**: 
  - Initial Model: Test RMSE = 509,069.51
  - Optimized Model with Early Stopping: Test RMSE = 479,889.25

- **Neural Network**: 
  - Initial Model: Test RMSE = 499,243.95
  - Optimized Model with Early Stopping: Test RMSE = 499,464.66

- **Grid Search Decision Tree**: 
  - Test RMSE = 470,670.42


**Q1. Can we accurately predict the total cost of a bird strike incident using incident attributes such as the number of engines, number of objects struck, and the description of the incident?**

- **Answer**: Yes, we can accurately predict the total cost of a bird strike incident using these attributes. The models built in this project, particularly the optimized Voting Regressor, demonstrated a strong ability to generalize and accurately predict costs based on the given features. The test RMSE of **469,450.16** for the Voting Regressor model indicates that the model performs well in estimating the total costs, showing that these incident attributes are indeed predictive of the total cost.

**Q2. How does the "Engine to Object Ratio" influence the prediction of bird strike costs, and what is its significance in the model?**

- **Answer**: The "Engine to Object Ratio" significantly influences the prediction of bird strike costs. This engineered feature captures the severity of an incident by representing the number of impacts per engine. The inclusion of this feature improved model accuracy by providing a more nuanced understanding of how the number of engines and objects struck correlate with the cost. Models that incorporated this ratio, such as the Voting Regressor and Decision Tree, showed better performance, confirming its importance in the predictive process.

**Q3. How do different machine learning models compare in terms of predictive accuracy for estimating the total cost of bird strikes?**

- **Answer**: Different machine learning models were compared, and the Voting Regressor (optimized model) outperformed others in terms of predictive accuracy with a Test RMSE of **469,450.16**. The Decision Tree and Gradient Boosting models also performed well, but the Voting Regressor’s ability to combine the strengths of multiple models gave it a competitive edge. The neural network, while powerful, did not outperform the ensemble methods, indicating that simpler models with the right hyperparameter tuning and feature engineering can be more effective for this specific task.

### **Best Model**

The **Voting Regressor (Optimized Model)** achieved the best performance with a Test RMSE of **469,450.16**. This model was selected for its ability to generalize well to unseen data, outperforming other models in terms of predictive accuracy.

### **Comparison to Baseline**
The optimized Voting Regressor model reduced the Test RMSE from 482,867.50 (baseline) to 469,450.16, demonstrating a significant improvement in the model’s ability to predict bird strike costs.

### **Contributing**
Contributions are welcome. Please open an issue or submit a pull request for any enhancements or bug fixes.

### **Acknowledgments**
Akanksha Kushwaha for project submission.
Scikit-learn Documentation for guidance on model implementation and evaluation.
