import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import seaborn as sns
import numpy as np
from scipy import stats
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import necessary libraries for machine learning models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error

import joblib

# Set visual style for plots
sns.set(font_scale=1.4)

@st.cache_resource
def train_models(X_train_scaled, y_train):
    # Define a dictionary of models to train
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(),
        "Random Forest Regressor": RandomForestRegressor(),
        "KNN Regressor": KNeighborsRegressor(),
        "SVM Regressor": SVR()
    }
    trained_models = {}
    for name, model in models.items():
        # Train each model on the scaled training data
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
    return trained_models

@st.cache_resource
def scale_data(X_train, X_test):
    # Standardize the training and test datasets
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled

def reformat_large_tick_values(tick_val, pos):
    """
    Converts large tick values (in billions, millions, and thousands) into a more readable format.
    """
    if tick_val >= 1000000000:
        val = round(tick_val/1000000000, 1)
        new_tick_format = '{:}B'.format(val)
    elif tick_val >= 1000000:
        val = round(tick_val/1000000, 1)
        new_tick_format = '{:}M'.format(val)
    elif tick_val >= 1000:
        val = round(tick_val/1000, 1)
        new_tick_format = '{:}K'.format(val)
    elif tick_val < 1000:
        new_tick_format = round(tick_val, 1)
    else:
        new_tick_format = tick_val

    # Convert new_tick_format to a string value
    new_tick_format = str(new_tick_format)
    
    # Remove unnecessary decimal zero
    index_of_decimal = new_tick_format.find(".")
    
    if index_of_decimal != -1:
        value_after_decimal = new_tick_format[index_of_decimal + 1]
        if value_after_decimal == "0":
            new_tick_format = new_tick_format[0:index_of_decimal] + new_tick_format[index_of_decimal + 2:]
            
    return new_tick_format

# Streamlit application title and introduction
st.write("## This Project is based on the football player data available from Kaggle repository (https://www.kaggle.com/datasets/thedevastator/footballpriceprediction?resource=download.")
st.write("* It contains the details of 18944 football players.")
st.write("* My project task is to create a machine learning model which can predict the price of a player based on its characteristics.")
st.write("* For solving this problem, I will approach the task with a step-by-step approach to create a data analysis and prediction model based on machine learning algorithms.")

# Step 1: Load and clean the data
st.write("## Step 1: Reading the data with Python")
st.write("This is one of the most important steps in data analysis! You must understand the data and the domain well before trying to apply any machine learning/AI algorithm.")         
         
# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Read the dataset
FootballData = pd.read_csv('footballData.csv')
st.write('* Shape before deleting duplicate values:', FootballData.shape)

# Remove duplicate rows if any
FootballData = FootballData.drop_duplicates()
st.write('* Shape After deleting duplicate values:', FootballData.shape)

# Display sample data for initial assessment
st.dataframe(FootballData.head(10), use_container_width=True)

# Data description observations
st.write("### Key observations from Step 1 about Data Description")
st.write("* The file contains 18944 player details from FIFA Database along with some additional features.")
st.write("* The file has 106 attributes, which describe the player's various skills, rankings, and player statistics.")

# Step 2: Define the problem statement
st .write("## Step 2 : Problem Statement Definition")
st.write("* Creating a prediction model to predict the price (value_eur) of a player.")
st.write("* Target Variable: value_eur")
st.write("* Predictors/Features: overall, potential, pace, shooting, etc.")

target = "value_eur"

# Step 3: Visualize the target variable's distribution
st.write("## Step 3: Visualizing the Target Variable")
st.write("* If the target variable's distribution is too skewed, the predictive modeling will lead to poor results.")
st.write("* A bell curve is desirable, but a slight positive or negative skew is also acceptable.")

# Plot the distribution of the target variable
fig, ax = plt.subplots()
ax.hist(FootballData[target], bins=30, edgecolor='k', alpha=0.7)
ax.set_title("Distribution of Value")
ax.set_xlabel("Value")
ax.set_ylabel('Frequency')
ax.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
st.pyplot(fig)

# Remove outliers from the target variable
Q1 = FootballData[target].quantile(0.25)
Q3 = FootballData[target].quantile(0.75)
IQR = Q3 - Q1
FootballData_Cleaned = FootballData[~((FootballData[target] < (Q1 - 1.5 * IQR)) | (FootballData[target] > (Q3 + 1.5 * IQR)))]

# Plot the distribution after removing outliers
st.write("###  Histogram of Target Variable after removing the outliers")
fig, ax = plt.subplots()
ax.hist(FootballData_Cleaned[target], bins=30, edgecolor='k', alpha=0.7)
ax.set_title("Distribution of Value")
ax.set_xlabel("Value")
ax.set_ylabel('Frequency')
ax.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
st.pyplot(fig)

st.write("## Observations from Step 3")
st.write("*  The data distribution of the target variable is skewed and we need to remove the outliers in the dataset.")
st.write("* The data distribution of the target variable is satisfactory to proceed further after removing the outliers.")
st.write("* There are sufficient number of rows for each type of values to learn from.")

# Step 4: Basic exploratory data analysis
st.write("## Step 4: Basic Exploratory Data Analysis")
st.write("* This step is performed to gauge the overall data.")
st.write("* The volume of data, the types of columns present in the data.")
st.write("* Initial assessment of the data should be done to identify which columns are Quantitative, Categorical, or Qualitative.")

# Display sample rows in the data
st.write("### First five rows of data")
st.dataframe(FootballData.head())
st.write("### Last five rows of data")
st.dataframe(FootballData.tail())

# Display data types and missing values
st.write("### Data Types:")
dtype_df = pd.DataFrame(FootballData.dtypes.astype(str), columns=["Data Type"]).reset_index()
dtype_df = dtype_df.rename(columns={"index": "Column Name"})
st.dataframe(dtype_df, use_container_width=True)

# Display summary statistics
st.write("### Summary Statistics:")
st.dataframe(FootballData.describe(), use_container_width=True)

# Display unique values for each column
st.write("### Unique Values:")
st.dataframe(FootballData.nunique().rename("Count"), use_container_width=True)

st.write("## Observations from Step 4")
st.write("* Unwanted columns can be removed from the dataset, especially those with very few values or those that do not contribute to player price prediction.")
st.write("* The following columns will be removed:")
st.write("* * sofifa_id, player_url, short_name, long_name, dob, club_name, league_name, preferred_foot, nationality, player_positions,")
st.write("* * work_rate, real_face, team_position, player_tags, team_jersey_number, loaned_from, joined, nation_position, nation_jersery_number")
st.write("* * and position rankings (e.g., ls, st, rs, lw, etc.).")

# Step 5: Visual exploratory data analysis
st.write("## Step 5: Visual Exploratory Data Analysis")
st.write("* Visualize the distribution of all the Categorical and Continuous Predictor variables in the data using bar plots and histograms respectively.")

# Define categorical and continuous variables
categorical = ("league_rank","international_reputation",\
               "weak_foot","skill_moves","contract_valid_until")
continuous = ("age","height_cm","weight_kg","overall","potential","value_eur","wage_eur","release_clause_eur","pace","shooting","passing","dribbling","defending","physic","gk_diving",\
              "gk_handling","gk_kicking","gk_reflexes","gk_speed","gk_positioning")

# Plot bar charts for categorical columns
st.write("### Categorical Predictors:")
st.write(categorical)
st.write("### Continuous Predictors:")
st.write(continuous)

st.write("### Bar charts for Categorical columns.")
for column in categorical:

    fig,ax=plt.subplots(figsize=(15,10))
    fig.suptitle('Bar charts of: '+ str(column))
    FootballData.groupby(column).size().plot(kind='bar')
    st.pyplot(fig)

st.write("### Histograms for Continuous variables")
for column in continuous:
            fig, ax = plt.subplots(figsize=(15, 10))
            ax.hist(FootballData[column], bins=30, edgecolor='k', alpha=0.7)
            ax.set_title(f"Distribution of {column}")
            ax.set_xlabel(column)
            ax.set_ylabel('Frequency')
            st.pyplot(fig)


st.write("## Observations from step 5")
st.write("### Bar Charts")
st.write("* Bar charts show the frequency of each category on the Y-axis and category names on the X-axis.")
st.write("* Ideally, categories should have comparable frequencies to provide sufficient data for ML/AI algorithms.")
st.write("* A skewed distribution with one dominant category may indicate low contribution to model development and correlation with the target variable.")
st.write("* Correlation analysis will help decide on the inclusion or exclusion of such columns.")

st.write("### Histograms")
st.write("* Histograms illustrate the distribution of continuous variables, with the X-axis showing value ranges and the Y-axis showing frequencies.")
st.write("* A desirable histogram resembles a bell curve. Excessive skewness may require outlier removal and further evaluation of the column.")

# Step 6: Outlier analysis
st.write("## Step 6: Outlier Analysis")
st.write("* Outliers are extreme values in the data that are significantly distant from the majority of the values.")
st.write("* Outliers can skew the development of machine learning models. When the algorithm attempts to accommodate these extreme values, it may deviate from the majority of the data.")

#outlier removal of age
Q1 = FootballData["age"].quantile(0.25)
Q3 = FootballData["age"].quantile(0.75)
IQR = Q3 - Q1
FootballData_Cleaned = FootballData[~((FootballData["age"] < (Q1 - 1.5 * IQR)) | (FootballData["age"] > (Q3 + 1.5 * IQR)))]

st.write("### Histogram of 'age' after removing outliers")
fig, ax = plt.subplots()
ax.hist(FootballData_Cleaned["age"], bins= 30, edgecolor='k', alpha=0.7)
ax.set_title("Distribution of 'age'")
ax.set_xlabel("Value")
ax.set_ylabel('Frequency')
ax.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
st.pyplot(fig)

#outlier removal of wage_eur
Q1 = FootballData["wage_eur"].quantile(0.25)
Q3 = FootballData["wage_eur"].quantile(0.75)
IQR = Q3 - Q1
FootballData_Cleaned = FootballData[~((FootballData["wage_eur"] < (Q1 - 1.5 * IQR)) | (FootballData["wage_eur"] > (Q3 + 1.5 * IQR)))]

st.write("### Histogram of 'wage_eur' after removing outliers")
fig, ax = plt.subplots()
ax.hist(FootballData_Cleaned["wage_eur"], bins= 30, edgecolor='k', alpha=0.7)
ax.set_title("Distribution of 'wage_eur'")
ax.set_xlabel("Value")
ax.set_ylabel('Frequency')
ax.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
st.pyplot(fig)

#outlier removal of release_clause_eur
Q1 = FootballData["release_clause_eur"].quantile(0.25)
Q3 = FootballData["release_clause_eur"].quantile(0.75)
IQR = Q3 - Q1
FootballData_Cleaned = FootballData[~((FootballData["release_clause_eur"] < (Q1 - 1.5 * IQR)) | (FootballData["release_clause_eur"] > (Q3 + 1.5 * IQR)))]

st.write("### Histogram of 'release_clause_eur' after removing outliers")
fig, ax = plt.subplots()
ax.hist(FootballData_Cleaned["release_clause_eur"], bins= 30, edgecolor='k', alpha=0.7)
ax.set_title("Distribution of 'release_clause_eur'")
ax.set_xlabel("Value")
ax.set_ylabel('Frequency')
ax.xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
st.pyplot(fig)

st.write("## Observations from step 6")
st.write("* The distribution of values have improved significantly by removing the outliers.")

# Step 7: Data cleaning
st.write("## Step 7: Missing values Analysis")
st.write("* Missing values should be addressed for each column individually.\
        \n* If a column has over 30% of its data missing, it cannot be treated, and that column should be excluded due to excessive missing information.\
        \n* Here are some options for handling missing values:\
        \n* * Remove rows with missing values if there are only a few records affected.\
        \n* * Impute missing values with the MEDIAN for continuous variables.\
        \n* * Impute missing values with the MODE for categorical variables.")

missing_values = FootballData_Cleaned.isnull().sum()
st.write("### Missing Values in Each Column")
dtype_df_missing_values = pd.DataFrame(missing_values, columns=["Missing Values"]).reset_index()
dtype_df_missing_values = dtype_df_missing_values.rename(columns={"index": "Column Name"})
st.dataframe(dtype_df_missing_values, use_container_width=True)

st.write("###  Observations from step 7")
st.write("* From the missing values table, it is clear we have to remove multiple columns due to huge amounts of missing data.\
        \n* The columns to be dropped are:\
        \n* * player_tags\
        \n* * loaned_from\
        \n* * nation_position\
        \n* * nation_jersey_number\
        \n* * defending_marking\
        \n* * gk_diving\
        \n* * gk_handling\
        \n* * gk_kicking\
        \n* * gk_reflexes\
        \n* * gk_speed\
        \n* * gk_positioning\
        \n* * player_traits")

unwanted = ("sofifa_id","player_url", "short_name", "long_name", "dob", "club_name", "league_name", "nationality", "player_positions",
            "work_rate", "real_face", "team_position", "player_tags", "team_jersey_number", "loaned_from", "joined", "nation_position",
            "nation_jersey_number", "defending_marking", "ls", "st", "rs", "lw", "lf", "cf", "rf", "rw", "lam", "cam", "ram", "lm", "lcm", "cm", "rcm", "rm",
            "lwb", "ldm", "cdm", "rdm", "rwb", "lb", "lcb", "cb", "rcb", "rb", "gk_diving", "gk_handling", "gk_kicking", "gk_reflexes",
            "gk_speed", "gk_positioning", "player_traits")

st.write("####  Dropped Columns:")
st.write(unwanted)

New_FootballData = FootballData_Cleaned.drop(columns=list(unwanted), axis=1)  

st.write("### Dataset after removing unwanted columns")
st.dataframe(New_FootballData.head()) 

#updating the continuous and categorical values after removing them from the dataset
new_continuous=[]
new_categorical=[]
for column in continuous:
     if column not in unwanted:
          new_continuous.append(column)
for column in categorical:
     if column not in unwanted:
          new_categorical.append(column)

#replacing null values with median for continuous variables and mode for categorical variables
for  column in new_continuous:
    New_FootballData[column] = New_FootballData[column].fillna(New_FootballData[column].median())

for column in  new_categorical:
    New_FootballData[column] = New_FootballData[column].fillna(New_FootballData[column].mode()[0])

st.write("### Dataset after replacing missing values")
st.dataframe(New_FootballData.head())

#Step 8: Feature Selection
st.write("## Step 8: Feature Selection - Correlation Matrix")
st.write("* Correlation value can only be calculated between two numeric columns.\
        \n* A correlation value in the range of [-1, 0) indicates an inverse relationship.\
        \n* A correlation value in the range of (0, 1] indicates a direct relationship.\
        \n* A correlation value close to 0 suggests no relationship.\
        \n* If the correlation value between two variables exceeds 0.5 in magnitude, it indicates a strong relationship, regardless of the sign.\
        \n* We examine the correlations between the target variable and all other predictor variables to identify which features are genuinely related to the target variable in question.")

# Select only continuous numerical columns for correlation analysis
numeric_columns = New_FootballData[new_continuous]
if not numeric_columns.empty:
    # Calculate and display the correlation matrix
    correlation_matrix = numeric_columns.corr()
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    # Display the full correlation matrix
    st.write("### Correlation Matrix:")
    st.dataframe(correlation_matrix, use_container_width=True)

    # Extract correlations with the target variable
    target_correlations = correlation_matrix[target].drop(target)  # Drop correlation of target with itself

    # Display the correlation of each feature with the target variable
    st.write(f"### Correlation of Features with Target Variable: `{target}`")
    st.dataframe(target_correlations, use_container_width=True)

    # Identify strong, moderate, and weak relationships
    strong_corr = target_correlations[target_correlations.abs() >= 0.5]
    moderate_corr = target_correlations[(target_correlations.abs() >= 0.3) & (target_correlations.abs() < 0.5)]
    weak_corr = target_correlations[target_correlations.abs() < 0.3]

    # Display the correlations classified as strong, moderate, and weak
    st.write("### Strong Correlations (|correlation| ≥ 0.5):")
    st.dataframe(strong_corr, use_container_width=True)

    st.write("### Moderate Correlations (0.3 ≤ |correlation| < 0.5):")
    st.dataframe(moderate_corr, use_container_width=True)

    st.write("### Weak Correlations (|correlation| < 0.3):")
    st.dataframe(weak_corr, use_container_width=True)
else:
    st.write("No continuous numeric columns available for correlation analysis.")

st.write("### Observations from step 8")
st.write("* We can select attributes that have a moderate and strong correlation with the target variable to improve the model's performance.\
         \n* The columns chosen are:\
         \n* * overall\
         \n* * wage_eur\
         \n* * release_clause_eur\
         \n* * potential\
         \n* * passing\
         \n* * dribbling")

final_continuous = ["overall", "wage_eur", "release_clause_eur", "potential", "passing", "dribbling"]

# Step 9: Statistical Feature Selection using ANOVA for Categorical Variables
st.write("## Step 9: Statistical Feature Selection (ANOVA for Categorical Variables)")
st.write("* Analysis of Variance (ANOVA) is conducted to determine whether there is a relationship between a given continuous variable and a categorical variable.")

# Select categorical columns for ANOVA analysis
categorical_columns = New_FootballData[new_categorical]

if not categorical_columns.empty:

    # Ensure the target variable is continuous
    if pd.api.types.is_numeric_dtype(New_FootballData[target]):
        # Dictionary to store ANOVA results
        anova_results = []

        # Perform ANOVA for each selected categorical variable
        for cat_col in new_categorical:
            anova_groups = New_FootballData.groupby(cat_col)[target].apply(list)
            f_val, p_val = stats.f_oneway(*anova_groups)

            # Append the results to a list
            anova_results.append({"Categorical Variable": cat_col, "F-value": f_val, "p-value": p_val})

        # Convert results to DataFrame
        anova_df = pd.DataFrame(anova_results)

        # Display the ANOVA results
        st.write("### ANOVA Results")
        st.write("The following table shows F-values and P-values for each categorical variable.")
        st.dataframe(anova_df, use_container_width=True)

        # Display significant variables based on p-value
        if "p-value" in anova_df.columns:
            significant_vars = anova_df[anova_df["p-value"] < 0.05]
            st.write("### Significant Variables (p < 0.05):")
            if not significant_vars.empty:
                st.dataframe(significant_vars, use_container_width=True)
            else:
                st.write("No significant variables found.")
            # Visualize the relationship using box plots
            st.write("### Box Plot: Categorical Variable vs Target")
            for cat_col in new_categorical:
                fig, ax = plt.subplots(figsize=(15, 8))
                sns.boxplot(x=cat_col, y=target, data=New_FootballData, ax=ax)
                ax.set_title(f"{cat_col} vs {target}")
                st.pyplot(fig)
        else:
            st.write("No categorical variables selected for ANOVA.")
    else:
        st.write("The target variable must be continuous for ANOVA analysis.")
else:
    st.write("No categorical variables available for ANOVA analysis.")

st.write("## Observation from step 9")
st.write("* The Categorical variables have a correlation with  the target variable, as the p-value is less than 5%.\
        \n* The columns chosen are:\
        \n* * league_rank\
        \n* * international_reputation\
        \n* * weak_foot\
        \n* * skill_moves\
        \n* * contract_valid_until")

final_categorical = ["league_rank", "international_reputation", "weak_foot", "skill_moves", "contract_valid_until"]

# Step 10: Selecting Final Predictors for Building Machine Learning Model
st.write("## Step 10: Selecting Final Predictors")
selected_features = final_continuous+final_categorical

st.write("* The final dataframe with selected features is:")
Final_FootballData = New_FootballData[selected_features]
st.dataframe(New_FootballData.head())

st.write("* Based on thorough exploratory data analysis, we can finalize the features/predictors/columns for machine learning model development as:")
st.write("### Selected Features:", selected_features)
st.write(f"### Target Variable: `{target}`")

# Step 11: Data conversion to numeric values for machine learning/predictive analysis
st.write("## Step 11: Data conversion to numeric values for machine learning/predictive analysis")
st.write("#### Steps to be performed on predictor variables before the data can be used for machine learning:\
        \n* Convert each ordinal categorical column to numeric.\
        \n* Convert binary nominal categorical columns to numeric using a 1/0 mapping.\
        \n* Convert all other nominal categorical columns to numeric using pd.get_dummies().\
        \n* Data transformation (optional): Apply standardization, normalization, log, or square root transformations, especially if using distance-based algorithms like KNN or neural networks.\
        \n* Convert ordinal variables to numeric—note that there are no ordinal categorical variables in this dataset.\
        \n* Convert binary nominal variables to numeric using a 1/0 mapping—this dataset does not contain any binary nominal variables in string format.")

Final_FootballData=pd.get_dummies(New_FootballData)
Final_FootballData["value_eur"] = New_FootballData["value_eur"]

#Step 12: Train/test data split and standardisation/normalisation of data
st.write("### Step 12: Train/test data split and standardisation/normalisation of data")
# Extracting the features and target variable
X = New_FootballData[selected_features]
y = New_FootballData[target]

# Splitting the data into train and test sets
st.write("#### Select the test size (percentage)")
test_size = st.slider("select", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Cache the scaler and scaled data
scaler, X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
st.write(f"### Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# Step 13: Model Training and Evaluation
st.write("## Step 13: Model Training and Evaluation")
# Check if models are already cached, otherwise train and cache them
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = train_models(X_train_scaled, y_train)

trained_models = st.session_state.trained_models

model_performance = {}
for name, model in trained_models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Store the results in the dictionary
    model_performance[name] = {"MSE": mse, "R2 Score": r2, "MAE": mae}

# Convert the performance dictionary to a pandas DataFrame for better visualization
performance_df = pd.DataFrame(model_performance).T  # Transpose to get model names as rows

# Apply styles using pandas built-in styling
styled_df = performance_df.style.format(precision=2) \
    .background_gradient(subset=["MSE"], cmap="Blues", low=0, high=1) \
    .background_gradient(subset=["R2 Score"], cmap="Greens", low=0, high=1) \
    .background_gradient(subset=["MAE"], cmap="Reds", low=0, high=1) \
    .set_properties(**{'text-align': 'center'}) \
    .set_table_styles([{
    'selector': 'th',
    'props': [('font-size', '14px'), ('text-align', 'center'), ('color', '#ffffff'),
                ('background-color', '#404040')]
}])

# Display the table with st.table or st.dataframe
st.write("## Model Performance Table")
st.dataframe(styled_df, use_container_width=True)

# Visualizing the Performance Comparison between Models
st.write("## Visualizing Model Performance Comparison")

# Extracting model names and their respective performance metrics
model_names = list(model_performance.keys())
mse_values = [model_performance[model]["MSE"] for model in model_names]
r2_values = [model_performance[model]["R2 Score"] for model in model_names]
mae_values = [model_performance[model]["MAE"] for model in model_names]

# Creating a bar plot to compare MSE, R2, and MAE across models
fig, ax = plt.subplots(3, 1, figsize=(20, 17))

# MSE Comparison
ax[0].bar(model_names, mse_values, color='blue')
ax[0].set_title("Model Comparison: Mean Squared Error (MSE)")
ax[0].set_ylabel("MSE")

# R2 Score Comparison
ax[1].bar(model_names, r2_values, color='green')
ax[1].set_title("Model Comparison: R2 Score")
ax[1].set_ylabel("R2 Score")

# MAE Comparison
ax[2].bar(model_names, mae_values, color='red')
ax[2].set_title("Model Comparison: Mean Absolute Error (MAE)")
ax[2].set_ylabel("MAE")

# Display the plot
plt.tight_layout()
st.pyplot(fig)

# Step 14: Selecting the Best Model
st.write("## Step 14: Selection of the best model")
st.write("#### Based on previous trials, we select the algorithm that yields the best average accuracy.\
         \n* Save the Model: Serialize the model into a file that can be stored anywhere.")

# Check if model performance dictionary has been populated
if model_performance:
    # Select the model with the lowest MSE
    best_model_mse = min(model_performance, key=lambda x: model_performance[x]["MSE"])
    st.write("* Best Model based on Lowest Mean Squared Error (MSE): ",best_model_mse)

    # Step 14: Retraining the Best Model on Entire Data
    st.write("### Retraining the Best Model")
    best_model = trained_models[best_model_mse]
    # Retrain the best model on the entire dataset
    # Combine and scale the entire dataset
    X_combined_scaled = scaler.fit_transform(X)
    best_model.fit(X_combined_scaled, y)

    # Save the best model in session state and also as a file
    if 'best_model' not in st.session_state:
        st.session_state.best_model = best_model

    # save the model after retraining
    model_filename = "best_model.pkl"
    joblib.dump(best_model, model_filename)
    st.write("* Model `",model_filename,"` has been retrained and saved as `",model_filename,"`.")
else:
    st.write("No model performance results available. Please ensure models were trained successfully.")

# Step 15: Deployment of the best model in production
st.write("## Step 15: Deployment of the best model in production")
st.write("To deploy the model, we will follow these steps:\
        \n* Train/Build the Model: Rebuild the model using 100% of the available data.\
        \n* Select Important Variables: It's advantageous to limit the number of predictors when deploying the model in production.\
        \n* Fewer predictors lead to a more stable model, reducing dependency on individual features, which is particularly important in high-dimensional data.")

# Load the saved model
model_filename = "best_model.pkl"

try:
    loaded_model = joblib.load(model_filename)
    st.write(f"Model `{model_filename}` loaded successfully!")

    # Allow user to input values for the features
    st.write("### Provide the input values for prediction")

    # Generate input fields dynamically based on the selected features
    user_input_values = {}
    for feature in selected_features:  # Ensure selected_features from step 11 is available
        user_input_values[feature] = st.number_input(f"Enter value for {feature}",
                                                        value=float(Final_FootballData[feature].mean()))

    # Add Predict button
    if st.button("Predict"):
        # Convert the user inputs into a DataFrame
        user_input_df = pd.DataFrame([user_input_values])

        # Scale the user inputs using the same scaler
        user_input_scaled = scaler.transform(user_input_df)

        # Make predictions using the loaded model
        predicted_value = loaded_model.predict(user_input_scaled)

        # Display the predicted value
        st.write(f"## Predicted {target}: {predicted_value[0]:.2f}")

except FileNotFoundError:
    st.write(f"Model `{model_filename}` not found. Please ensure the model has been saved correctly.")