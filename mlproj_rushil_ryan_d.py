import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from statsmodels.tsa.ar_model import AutoReg
from datetime import date

def preprocess_data(url='https://econdata.s3-us-west-2.amazonaws.com/Reports/Core/RDC_Inventory_Core_Metrics_County_History.csv') -> pd.DataFrame:
    '''Retrieves and preprocesses housing data from the web.
    
    Parameters
    ----------
    url: string
        URL of link housing data is retrieved from
    
    Returns
    -------
    data: DataFrame
        DataFrame of housing data, which includes statistics on its location, date, and median pricing '''
    data = pd.read_csv(url).drop('median_days_on_market', axis = 1)
    data.rename(columns={
        'county_name' : 'Location',
        'month_data_yyyymm': 'Date',
        'median_listing_price_per_square_foot': 'Median PPSF',
        'median_listing_price': 'Median Price'})
    return data

        
# Obtains the data of a specific location based on the type of location chosen and the location name
def get_location_data(data_set, location_input, type_location):
    locations = data_set['county_name']
    (date, median_price_sqft, median_price) = get_essential_data()
    states_dict = {'alabama': 'al', 'alaska': 'ak', 'arizona': 'az', 'arkansas': 'ar', 'california': 'ca', 'connecticut': 'ct', 'delaware': 'de', 'district of columbia': 'dc', 'florida': 'fl', 'georgia': 'ga', 'hawaii': 'hi', 'idaho': 'id', 'illinois': 'il', 'indiana': 'in', 'ia': 'iowa', 'ks': 'kansas', 'ky': 'kentucky', 'louisiana': 'la', 'maine': 'me', 'maryland': 'md', 'massachusetts': 'ma', 'michigan': 'mi', 'minnesota': 'mn', 'mississippi': 'ms', 'missouri': 'mo', 'montana': 'mt', 'nebraska': 'ne', 'nevada': 'nv', 'new hampshire': 'nh', 'new jersey': 'nj', 'new mexico':'nm', 'new york': 'ny', 'north carolina': 'nc', 'north dakota': 'nd', 'ohio': 'oh', 'oklahoma': 'ok', 'oregon': 'or', 'pennsylvania': 'pa', 'rhode island': 'ri', 'south carolina': 'sc', 'south dakota': 'sd', 'tennessee': 'tn', 'texas': 'tx', 'utah': 'ut', 'vermont': 'vt', 'virginia': 'va', 'washington': 'wa', 'west virginia': 'wv', 'wisconsin': 'wi', 'wyoming': 'wy'}
    counter = 0
    if type_location == 'state':
        states_data = []
        for state_str in locations:
            state_str = str(state_str)
            state = state_str[-2:]
            if states_dict[location_input] == state:
                states_data.append([translate_time(date[counter]), median_price[counter], median_price_sqft[counter]])
            counter += 1
        return states_data
    else:
        county_data = []
        for i in range(len(data_set)):
            if locations[i] == location_input:
                county_data.append([translate_time(date[counter]), median_price[counter], median_price_sqft[counter]])
            counter += 1
        return county_data

# Prediction Functions 
        
# Linear Regression
def predict_cost_LinReg(location, year, x, y):
    X_trained = np.array(x).reshape(-1, 1); Y_trained = np.array(y).reshape(-1, 1)
    '''Using linear regression, predicts the future price of a home in a specified location'''
    regressor = LinearRegression()
    regressor.fit(X_trained, Y_trained)
    future_median = regressor.predict([[year]])
    return future_median

# Autoregression
def predict_cost_AR(data, year, x, y, z):
    current_year= float(date.today().year)
    current_month = float((date.today().month) / 12)
    current = current_year + current_month
    lead = int(year - current)
    lead *= 12
    test_removed = int(len(y) * .2)
    y.reverse()
    # print(y)
    # print(str(len(y) + test_removed + lead) + " months ahead of now")
    '''Using a time series analysis called autoregression, predicts the future price of a home using a certain amount of lags, or previous data points'''
    AR_model = AutoReg(y, lags=1)
    AR_model_fitted = AR_model.fit()
    prediction = AR_model_fitted.predict(start=len(y), end=(len(y) + test_removed +  lead), dynamic=False)
    # print(prediction[-1])
    '''link: https://pythondata.com/forecasting-time-series-autoregression/; does something called autoregression'''
    return prediction[-1]
# # Lasso Regressions
# def predict_cost_Lasso(location, year, x, y, z):
#     '''Using Lasso regression, predicts the future price of a home using a time and median listing price per square foot feature; performs a linear regression to gather the median listing price per square foot that corresponds with the median listing price overall'''
#     sqft_prediction = LinearRegression()
#     sqft_prediction_fitted = sqft_prediction.fit(x, z)
#     sqft_prediction_fitted = sqft_prediction.predict([[year]])
    
#     lasso_model = linear_model.Lasso(alpha=1)
#     lasso_model_fitted = lasso_model.fit(x, y, z)
#     prediction = lasso_model_fitted.predict([[year, sqft_prediction_fitted]])
#     return prediction

def output_results():
    pass

def main():
    data = preprocess_data()
    print("---Future Housing Price Predictor!---\n")
    while True:
        location = input("Enter the full name of the state you wish to look at: ")
        X_train, X_test, y_train, y_test = train_test_split()
        if len(time) == 6:
            time = translate_time(time)

        print("Cost based on Linear Regression:", predict_cost_LinReg(location, float(time), X_data, Y_data))
        print("Cost based on Autoregression (time series analysis):", predict_cost_AR(location, float(time), X_data, Y_data, Z_data))

        # Allows user to perform a new calculation if desired
        retry = input("Query another location/date? (Y/N): ", )
        if retry.lower() == "y":
            continue
        elif retry.lower() == "n":
            print("\nThanks for using our program! Goodbye!")
            break
        else:
            print("Not a valid input. Please try again.")

if __name__ == "__main__":
    main()