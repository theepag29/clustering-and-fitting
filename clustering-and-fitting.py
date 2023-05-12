# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:18:30 2023

@author: ta22ado
"""

import numpy as np
import pandas as pd
import scipy.optimize as opt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


# Read the dataset from a csv file and store it in a pandas dataframe
dataset = pd.read_csv('API_19_DS2_en_csv_v2_5361599.csv', skiprows=4)
# Define the indicators of interest and the year of interest
indicators = ["EN.ATM.CO2E.SF.ZS", "AG.LND.ARBL.ZS"]
year = ["2005"]

# Define a function to extract the data from the dataset for a given Indicator Code and year


def extract_data(indicators, year):
    """
    This function takes two arguments, `indicators` and `year`, and extracts the data from the dataset for a given Indicator Code and given year. It then drops some columns from the dataset to create a new dataframe and returns a list of dataframes. 
    """
    # Select the data for the two indicators of interest
    co2_emi_solid = dataset[dataset["Indicator Code"] == indicators[0]]
    arable_land = dataset[dataset["Indicator Code"] == indicators[1]]
    # Drop unnecessary columns from the dataframes
    co2_emi_solid = co2_emi_solid.drop(
        ["Country Code", "Indicator Name", "Indicator Code"], axis=1).set_index("Country Name")
    arable_land = arable_land.drop(
        ["Country Code", "Indicator Name", "Indicator Code"], axis=1).set_index("Country Name")
    dfs = []
    # Iterate over the list of years and create a new dataframe for each year with the relevant data
    for year in year:
        co2_emi_ = co2_emi_solid[year]
        arable_ = arable_land[year]
        data_frame = pd.DataFrame(
            {"CO2 emissions from solid fuel consumption (% of total) ": co2_emi_, "Arable land": arable_})
        # Remove any rows with missing values and add the new dataframe to the list of dataframes
        df = data_frame.dropna(axis=0)
        dfs.append(df)
    # Return the list of dataframes
    return dfs


# Call the extract_data function to get the dataframes for the indicators and year of interest
cluster_df = extract_data(indicators, year)


def clustering(data_frames):
    """
This function clusters a list of dataframes using k-means algorithm. The function first scales the data using standard scaling technique and then finds the optimum number of clusters using the elbow method. It then fits the k-means algorithm to the dataset and plots the clusters and their centroids on the original data. 
"""
    year = "2005"
    for df in data_frames:
        x = df.values
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        # Finding the optimum number of clusters
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++',
                            max_iter=300, n_init=10, random_state=42)
            kmeans.fit(x_scaled)
            wcss.append(kmeans.inertia_)

        # Plotting the elbow graph
        plt.plot(range(1, 11), wcss)
        plt.title('The Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()

        # Fitting kmeans to the dataset
        kmeans = KMeans(n_clusters=4, init='k-means++',
                        max_iter=300, n_init=10, random_state=0)
        y_kmeans = kmeans.fit_predict(x_scaled)

        # Plotting the clusters on original data
        plt.scatter(x_scaled[y_kmeans == 0, 0], x_scaled[y_kmeans ==
                    0, 1], s=50, c='red', label='Cluster 1')
        plt.scatter(x_scaled[y_kmeans == 1, 0], x_scaled[y_kmeans ==
                    1, 1], s=50, c='blue', label='Cluster 2')
        plt.scatter(x_scaled[y_kmeans == 2, 0], x_scaled[y_kmeans ==
                    2, 1], s=50, c='green', label='Cluster 3')
        plt.scatter(x_scaled[y_kmeans == 3, 0], x_scaled[y_kmeans ==
                    3, 1], s=50, c='cyan', label="Cluster 4")

        # Plotting the centroids
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
                    :, 1], s=100, marker="*", c='black', label='Centroids')

        plt.title(f'Clusters of countries ' + year)
        plt.xlabel("CO2 emissions from solid fuel consumption (% of total)")
        plt.ylabel("Arable land")

        plt.legend()
        plt.savefig('scatterplotads.png', dpi=300)
        plt.show()


clustering(cluster_df)


def read_wb_data(file_path: str):
    """Read the World Bank data from the given file path and return the original and transposed dataframes."""
    # Read the CSV file using pandas.read_csv function
    original_df = pd.read_csv(file_path)
    # Extract the 'Country Name' column from the original dataframe as a list
    country = list(original_df['Country Name'])
    # Transpose the original dataframe using the .transpose() method and set
    # the columns to the list of country names
    transposed_df = original_df.transpose()
    transposed_df.columns = country
    # Remove the first 4 rows and last row from the transposed dataframe using
    # .iloc[]
    transposed_df = transposed_df.iloc[4:]
    transposed_df = transposed_df.iloc[:-1]
    # Reset the index of the transposed dataframe and rename the 'index'
    # column to 'Year'
    transposed_df = transposed_df.reset_index()
    transposed_df = transposed_df.rename(columns={"index": "Year"})
    # Convert the 'Year' column to an integer datatype
    transposed_df['Year'] = transposed_df['Year'].astype(int)
    return original_df, transposed_df


# Call the read_wb_data function with the given file path and store
# the returned dataframes in df1 and df2
df1, df2 = read_wb_data('data 2.csv')


def calculate_ED_stats(dataframe):
    """Calculate the mean NRP and NRP in 2013 for the input dataframe."""
    # Calculate the mean NRP for the dataframe using the .mean() method
    mean_nrp = dataframe.mean()
    print("Mean NRP : ")
    print(mean_nrp)
    print("\n\n")

    # Filter the dataframe to only include rows where the 'Year' column is 2013
    nrp_2013 = dataframe[dataframe['Year'] == 2013]
    print("NRP 2013 : ")
    print(nrp_2013)


# Call the calculate_ED_stats function with the transposed dataframe df2
calculate_ED_stats(df2)


def plot_ED_over_time(df, countries):
    """Plot the enery depletion over time for the specified countries in the input dataframe."""
    # Plot the specified countries' enery depletion over time
    # using the .plot() method
    df.plot(x='Year', y=countries)
    # Set the plot title, x-axis label, and y-axis label
    plt.title("Energy Depletion for each country", fontsize=15)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("% of GNI", fontsize=14)
    # Show the plot
    plt.savefig('ED_over_time.png', dpi=300)
    plt.show()


# Extract the list of country names from the original dataframe df1
country_name = list(df1['Country Name'])
# Call the plot_ED_over_time function with the transposed dataframe df2
# and the list of country names
plot_ED_over_time(df2, country_name)


# Call the read_wb_data function with the given file path and store
# the returned dataframes in original_df and transposed_df
original_df, transposed_df = read_wb_data("data 1.csv")

# Extract the list of country names from the original dataframe original_df
country = list(original_df['Country Name'])


def plot_depletion(df, country):
    """Plot the enery depletion over time for the specified country in the input dataframe."""
    # Plot the enery depletion over time using the .plot() method
    df.plot("Year", country, color="purple")
    # Set the plot title, x-axis label, y-axis label, and legend
    plt.title(
        f"{country}'s enery depletion (% of GNI)",
        fontsize=15)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("% of GNI", fontsize=14)
    plt.legend(["ED"])
    # Show the plot
    plt.savefig('depletion.png', dpi=300)
    plt.show()


# Call the plot_depletion function with the transposed
# dataframe transposed_df and the first country in the list of countries
plot_depletion(transposed_df, country[0])


def logistic(t, n0, g, t0):
    """Calculate the logistic function with the specified parameters."""
    f = n0 / (1 + np.exp(-g * (t - t0)))
    return f


# Set the input dataframe as transposed_df
df = transposed_df
# Fit the logistic function to the data using scipy.optimize.curve_fit()
# function
param, covar = opt.curve_fit(
    logistic, df["Year"], df[country].squeeze(), p0=(
        float(
            df[country].iloc[0]), 0.03, 2000.0))
# Calculate the standard deviation of the parameters
sigma = np.sqrt(np.diag(covar))
print("parameters:", param)
print("std. dev.", sigma)

# Add a new column to the input dataframe transposed_df that contains the
# values of the fitted logistic function
df["fit"] = logistic(df["Year"], *param)


def plot_logistic_fit(df, country):
    """
    Plots the logistic fit of a country's enery depletion.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the data to be plotted.
    country (str): The name of the country to be plotted.

    Returns:
    None
    """
    df.plot("Year", [country, "fit"], color=["purple", "black"])
    plt.title(
        "Logistic fit of {}'s enery depletion".format(country),
        fontsize=15)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("% of GNI", fontsize=14)
    plt.legend(["ED"])
    plt.savefig('logistic_fit.png', dpi=300)
    plt.show()


plot_logistic_fit(df, country[0])


future_years = [2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030]
future_log = logistic(future_years, *param)

print("Future Logisitc Values 2021-2030 :")
print(future_log)


year = np.arange(df['Year'][0], 2031)
print(year)
forecast = logistic(year, *param)


def plot_future_prediction(df, country, forecast, year):
    """
    Plots the future prediction using the logistic fit of a country's enery depletion.

    Parameters:
    df : The dataframe containing the data to be plotted.
    country: The name of the country to be plotted.
    forecast: The forecasted values for the country.
    year: The years for the forecasted values.
    """
    plt.plot(df["Year"], df[country], label="PG", color="purple")
    plt.plot(year, forecast, label="forecast", color="black")

    plt.title(
        "Future Year prediction using Logistic of {}'s ED".format(country),
        fontsize=15)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("% annual", fontsize=14)
    plt.legend(["PG"])
    plt.legend()
    plt.savefig('future_predic.png', dpi=300)
    plt.show()


plot_future_prediction(df, country[0], forecast, year)


df2 = pd.DataFrame({'Future Year': future_years, 'Logistic': future_log})
df2


def err_ranges(x, func, param, sigma):
    """
    Calculates the error ranges for a given function.

    Parameters:
    x (numpy.array): The x values for the function.
    func (callable): The function to calculate error ranges for.
    param (list): The parameters of the function.
    sigma (numpy.array): The standard deviations of the parameters.

    Returns:
    tuple: A tuple containing the lower and upper limits of the error ranges.
    """
    import itertools as iter

    lower = func(x, *param)
    upper = lower

    uplow = []
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))

    pmix = list(iter.product(*uplow))

    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)

    return lower, upper


low, up = err_ranges(year, logistic, param, sigma)


def plot_prediction_limits(df, country, forecast, year, low, up):
    """
    Plots the upper and lower limit of a country's enery depletion (ED) prediction.

    Args:
    - df (pandas.DataFrame): Dataframe containing the country's ED data.
    - country (str): Country's name.
    - forecast (numpy.ndarray): Array containing predicted values for future years.
    - year (numpy.ndarray): Array containing all years from the first year of data to the last predicted year.
    - low (numpy.ndarray): Array containing lower limit values for each year.
    - up (numpy.ndarray): Array containing upper limit values for each year.

    Returns:
    - None
    """
    plt.figure()
    plt.plot(df["Year"], df[country], label="ED", color="purple")
    plt.plot(year, forecast, label="forecast", color="black")

    plt.fill_between(year, low, up, color="pink", alpha=0.7)
    plt.title(
        "Upper and Lower Limit of {}'s enery depletion".format(country),
        fontsize=15)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("% of GNI", fontsize=14)
    plt.legend()
    plt.savefig('plot_predic_limit.png', dpi=300)
    plt.show()


# Plot upper and lower limit of ED prediction for the chosen country
plot_prediction_limits(df, country[0], forecast, year, low, up)

# Print the upper and lower limit values for ED prediction for the year 2025
print(err_ranges(2025, logistic, param, sigma))
