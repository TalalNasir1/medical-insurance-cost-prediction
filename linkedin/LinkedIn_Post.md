# LinkedIn Post — Medical Insurance Cost Prediction

**Published:** September 29, 2026  
**Live post:** https://www.linkedin.com/feed/update/urn:li:activity:7510427083039408128/

I’m excited to share my completed machine learning project: **Medical Insurance Cost Prediction**.

The real-world problem I explored is the variation in annual medical insurance charges. My goal was to build an interpretable application that estimates costs from six user inputs: age, BMI, number of children, gender, smoking status, and region.

**What I built**

I developed an end-to-end regression workflow covering data inspection, cleaning, exploratory analysis, preprocessing, model training, evaluation, prediction, documentation, and deployment. The final application accepts user information and returns an immediate estimated insurance cost.

**Dataset and model**

I used the Medical Cost Personal Dataset, which contains 1,338 observations and seven original columns. After removing one duplicate, categorical variables were one-hot encoded and the data was divided into training and test sets using an 80/20 split.

The selected algorithm was **Multiple Linear Regression** because it provides a clear, interpretable baseline for understanding how each feature is associated with predicted charges.

**Key results**

- R² score: **0.8069**
- Mean Absolute Error: **$4,177.05**
- Root Mean Squared Error: **$5,956.34**
- Smoking was the strongest cost-related feature, with a fitted coefficient of approximately **+$23,078**, holding the other inputs constant.
- Age, BMI, and number of children also had positive fitted coefficients.

These results mean the model explained approximately **80.69% of the variation** in charges in the held-out test set. The model is intended as an educational demonstration and not for medical, financial, or underwriting decisions.

**Technologies used**

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib, Gradio, Google Colab, GitHub, and Hugging Face Spaces.

GitHub repository:
https://github.com/TalalNasir1/medical-insurance-cost-prediction

Live application:
https://huggingface.co/spaces/TalalNasir/medical-insurance-cost-prediction

I would appreciate any feedback on the project, model evaluation, or application design.

#MachineLearning #DataScience #Python #ScikitLearn #LinearRegression #DataAnalytics #ArtificialIntelligence #MLOps #Gradio #PortfolioProject #OpenToWork
