# UK Carbon Emissions Data Mining Project

## Project Overview

This project aims to analyze UK carbon emissions data to gain insights into emission trends, identify key contributing factors, and build predictive models. This analysis can inform policy decisions and contribute to a better understanding of the UK's progress towards carbon emission reduction goals.

## Key Features & Benefits

*   **Data Exploration and Visualization:** Comprehensive exploratory data analysis (EDA) to understand the data's structure, identify patterns, and visualize key trends.
*   **Predictive Modeling:** Development of machine learning models to forecast future carbon emissions based on historical data. Including ARIMA time series modelling.
*   **Fuel Classification:** Identifying key features for fuel classification.
*   **Streamlit Dashboard:** An interactive Streamlit dashboard to visualize the data and explore model predictions.
*   **Clustering Analysis:** Use of K-means to group similar emissions profiles.

## Prerequisites & Dependencies

Before running this project, ensure you have the following installed:

*   **Python (3.7 or higher):** Required for running the scripts and notebooks.
*   **pip:** Python package installer.

Install the following Python libraries using pip:

```bash
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install streamlit
pip install joblib
pip install pmdarima # For ARIMA model
```

## Installation & Setup Instructions

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Shreyasrs23/UK-Carbon-Emissions-Data-Mining-Project.git
    cd UK-Carbon-Emissions-Data-Mining-Project
    ```

2.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt #create this if it doesn't exist and list all dependencies
    ```

3.  **Download the dataset:**

    Place the `final-greenhouse-gas-emissions-tables-2023.xlsx` file in the project root directory.

4. **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

## Usage Examples & API Documentation

### Streamlit App:
To run the Streamlit application, execute the following command:
```bash
streamlit run app.py
```
The application will open in your web browser, providing interactive visualizations and model predictions.

### Data Exploration and Modeling:
The `Data_Exploration.ipynb` and `ModelTrainings.ipynb` notebooks contain detailed code for data exploration, preprocessing, model training, and evaluation.  These provide examples of working with the data and models.

## Project Structure

```
├── Data_Exploration.ipynb                # Jupyter notebook for data exploration and visualization
├── IE 6318 Project Proposal - Shreyas Rajapur Sanjay - 1002221283.pdf #Project proposal document
├── ModelTrainings.ipynb                  # Jupyter notebook for model training and evaluation
├── README.md                             # This README file
├── app.py                                # Streamlit application code
├── arima_ghg_model.pkl                   # Trained ARIMA model for GHG emissions prediction
├── final-greenhouse-gas-emissions-tables-2023.xlsx # Dataset containing UK carbon emissions data
├── fuel_classification_features.csv      # CSV containing Fuel Classification Features.
├── fuel_classification_model.pkl         # Trained Fuel Classification Model
├── kmeans_clustering_model.pkl         # Trained K-means Clustering Model
```

## Configuration Options

The `app.py` file contains configurable settings that can be adjusted to customize the application. These may include:

*   **Data source:** The path to the `final-greenhouse-gas-emissions-tables-2023.xlsx` file.
*   **Model selection:** The specific model used for predictions.
*   **Visualization options:** Customization of plot colors, labels, and other visual elements.

## Contributing Guidelines

We welcome contributions to this project! If you would like to contribute, please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Implement your changes, ensuring that the code is well-documented and follows coding best practices.
4.  Test your changes thoroughly.
5.  Submit a pull request with a clear description of your changes.

## License Information

This project has no specified license. Usage of the code is not authorized.

## Acknowledgments

*   [UK Government Department for Energy Security and Net Zero](https://www.gov.uk/government/organisations/department-for-energy-security-and-net-zero) - For providing the UK greenhouse gas emissions data.
*   [Streamlit](https://streamlit.io/) - For providing an excellent framework for building interactive web applications.
*   [Scikit-learn](https://scikit-learn.org/stable/) - For providing the machine learning tools for this project.