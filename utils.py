'''
Programmer: Jovan Koledin
Class: CPSC 222-02, Fall 2022
10/25/22

Description: This program contains the functions used for gather weather data for a given city and helps run Jovan's Final Project
'''
import json
import requests
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt

def replace_space(city_name):
    '''
    Replaces spaces in a city name with "+"
    Parameter city_name: name of city to adjust
    Returns: city name with all spaces replaced with a +
    '''
    city_out = ""
    for letter in city_name:
        if letter == " ":
            city_out += "+"
        else:
            city_out += letter
    return city_out

def get_lat_lng(city_name):
    '''
    Gets lattitude and longitude of a city
    Parameter city_name: name of city to find latitude and longitude of
    Returns: latitude and longitude
    '''
    #Establish URL
    mapquest_key = "HvM9FjwlGAFk6gG4B4blvvXxnpEIBUNm"
    url = "http://www.mapquestapi.com/geocoding/v1/address"
    url += "?key=" + mapquest_key
    url += "&city=" + city_name
    response = requests.get(url=url)

    #Convert to JSON 
    json_object = json.loads(response.text)
    results_object = json_object["results"]
    results_object = results_object[0]
    locations_object = results_object["locations"]
    locations_object = locations_object[0]
    latLng_object = locations_object["latLng"]

    return latLng_object["lat"], latLng_object["lng"]

def get_weather_station_ID(lat, lng):
    '''
    Gets closest weather station ID from given latitude and longitude
    Parameters lat,lng: Latitude and Longitude
    Returns: Weather station ID
    '''
    #Establish URL
    url = "https://meteostat.p.rapidapi.com/stations/nearby"
    querystring = {"lat":str(lat), "lon":str(lng), "limit":"1"}
    headers = {
        "X-RapidAPI-Key": "f04d55200bmsh4d41172f5100e51p1fcbdajsn18f729157713",
        "X-RapidAPI-Host": "meteostat.p.rapidapi.com"
    }
    #Convert to JSON
    response = requests.get(url=url, headers=headers, params=querystring)
    json_object = json.loads(response.text)
    data_object = json_object["data"]
    data_object = data_object[0]

    return data_object["id"]

def get_daily_weather_data(ID):
    '''
    Gets daily weather data from a weather station ID between (2021-02-21 through 2022-02-20)
    Parameters ID: weather station ID
    Returns: Json dictionary
    '''
    #Establish URL
    url = "https://meteostat.p.rapidapi.com/stations/daily"
    querystring = {"station":ID, "start":"2022-09-21", "end":"2022-12-03", "units":"imperial"}
    headers = {
        "X-RapidAPI-Key": "f04d55200bmsh4d41172f5100e51p1fcbdajsn18f729157713",
        "X-RapidAPI-Host": "meteostat.p.rapidapi.com"
    }
    #Convert to JSON 
    response = requests.get(url=url, headers=headers, params=querystring)
    json_object = json.loads(response.text)

    return json_object["data"]

def clean(df):
    '''
    Remove columns with more than 50% of data missing and ...
    fill in missing values using interpolate and backfilling and forward filling
    Parameters df: data frame to be cleaned
    Returns: cleaned dataframe
    '''
    #Remove columns with more than 50% of data missing
    limit = 50.0 
    min_count = int(((100-limit)/100)*df.shape[0] + 1)
    df = df.dropna(axis=1, thresh=min_count)

    #Fill in remaining missing values
    df = df.interpolate()
    df = df.fillna(method='bfill')
    df = df.fillna(method='ffill')

    return df

def cleaner(df):
    # Reassign W/L column to 1s and 0s for wins and losses respectively
    import warnings
    warnings.filterwarnings("ignore")
    df['W/L'][df['W/L'] == 'W'] = 1
    df['W/L'][df['W/L'] == 'L'] = 0
    # Reassign Game Type column to 1s and 0s for 5-on-5 and 3-on-3 respectively
    df['Game Type'][df['Game Type'] == '5-on-5'] = 1
    df['Game Type'][df['Game Type'] == '3-on-3'] = 0
    
    return df

def most_points(df):
    '''
    Determine if more more points are scored during 3-on-3 games
    Parameters df: data frame to be analyzed
    Returns: N/A
    '''
    #Organize data
    points_5 = df[df['Game Type']==1]['Points']
    points_5_mean = points_5.mean()
    print("Average points for 5-on-5 games:", round(points_5_mean, 2))
    points_5 = points_5.values.tolist()
    points_3 = df[df['Game Type']==0]['Points']
    points_3_mean = points_3.mean()
    print("Average points for 3-on-3 games:", round(points_3_mean, 2))
    points_3 = points_3.values.tolist()

    #Perfrom test with scipy
    from scipy import stats
    t, p_val = stats.ttest_ind(points_3, points_5)
    p_val /= 2
    alpha = 0.01
    if (t > 0 and p_val < alpha):
        print("Jovan scores more points in 3-on-3 games versus 5-on-5 games")
    elif (t < 0 and p_val < alpha):
        print("Jovan scores more points in 5-on-5 games versus 3-on-3 games")
    else:
        print("There is no difference between how many points Jovan scores in 3-on-3 versus 5-on-5")
    
def comp_received(df):
    '''
    Determine if more compliments were recieved during winning games
    Parameters df: data frame to be analyzed
    Returns: N/A
    '''
    #Organize data
    comp_rec_win = df[df['W/L']==1]['Compliments received']
    print("Average compliments recieved for winning games: ", round(comp_rec_win.mean(), 2))
    comp_rec_win = comp_rec_win.values.tolist()
    comp_rec_loss = df[df['W/L']==0]['Compliments received']
    print("Average compliments recieved for losing games: ", round(comp_rec_loss.mean(), 2))
    comp_rec_loss = comp_rec_loss.values.tolist()

    #Perfrom test with scipy
    from scipy import stats
    t, p_val = stats.ttest_ind(comp_rec_win, comp_rec_loss)
    p_val /= 2
    alpha = 0.01
    if (t > 0 and p_val < alpha):
        print("Jovan recieved more compliments during winning games")
    elif (t < 0 and p_val < alpha):
        print("Jovan recieved less compliments during winning games")
    else:
        print("There is no difference between the amounts of compliments recieved for winning and losing games")

def points_win(df):
    '''
    Determine if more points were scored during winning games
    Parameters df: data frame to be analyzed
    Returns: N/A
    '''
    #Organize data
    points_win = df[df['W/L']==1]['Points']
    print("Average points scored for winning games: ", round(points_win.mean(), 2))
    points_win = points_win.values.tolist()
    points_loss = df[df['W/L']==0]['Points']
    print("Average points scored for losing games: ", round(points_loss.mean(), 2))
    points_loss = points_loss.values.tolist()

    #Perfrom test with scipy
    from scipy import stats
    t, p_val = stats.ttest_ind(points_win, points_loss)
    p_val /= 2
    alpha = 0.01
    if (t > 0 and p_val < alpha):
        print("Jovan scored more points during winning games")
    elif (t < 0 and p_val < alpha):
        print("Jovan scored less points during winning games")
    else:
        print("There is no difference between the amount of points scored during winning and losing games")

def visualize(df):
    '''
    Visualize Final project data
    Parameters df: data frame to be analyzed
    Returns: N/A
    '''
    import matplotlib.pyplot as plt

    #Overall stats 
    total_games = df.count()["date"]
    total_games_5 = df['W/L'].sum()
    total_games_3 = total_games - total_games_5

    two_per = 100*df['Two-pointers made'].sum()/df['Two-pointers attempted'].sum()
    three_per = 100*df['Three-pointers made'].sum()/df['Three-pointers attempted'].sum()
    win_per = 100*df[df['W/L']==1]["W/L"].count()/total_games
    data = {'Two-Pointer':two_per, 'Three-Pointer':three_per, 'Win':win_per}
    categories = list(data.keys())
    values = list(data.values())
    plt.figure()
    plt.ylim([0, 65])
    plt.bar(categories, values, color ='maroon', width = 0.4)
    plt.ylabel("Percentage (%)")
    plt.title("Shot and win percentage for all games")
    plt.show()

    #5-on-5 stats
    two_makes_5 = df[df['Game Type']==1]['Two-pointers made'].sum()
    two_takes_5 = df[df['Game Type']==1]['Two-pointers attempted'].sum()
    two_per_5 = 100*two_makes_5/two_takes_5
    three_makes_5 = df[df['Game Type']==1]['Three-pointers made'].sum()
    three_takes_5 = df[df['Game Type']==1]['Three-pointers attempted'].sum()
    three_per_5 = 100*three_makes_5/three_takes_5
    win_per_5 = 100*df[df['Game Type']==1]['W/L'].sum()/total_games_5
    data = {'Two-Pointer':two_per_5, 'Three-Pointer':three_per_5, 'Win':win_per_5}
    categories = list(data.keys())
    values = list(data.values())
    plt.ylim([0, 65])
    plt.bar(categories, values, width = 0.4, color = 'red', alpha = 0.5, label = '5-on-5')
    plt.ylabel("Percentage (%)")

    #3-on-3 stats
    two_makes_3 = df[df['Game Type']==0]['Two-pointers made'].sum()
    two_takes_3 = df[df['Game Type']==0]['Two-pointers attempted'].sum()
    two_per_3 = 100*two_makes_3/two_takes_3
    three_makes_3 = df[df['Game Type']==0]['Three-pointers made'].sum()
    three_takes_3 = df[df['Game Type']==0]['Three-pointers attempted'].sum()
    three_per_3 = 100*three_makes_3/three_takes_3
    win_per_3 = 100*df[df['Game Type']==0]['W/L'].sum()/total_games_3
    data = {'Two-Pointer':two_per_3, 'Three-Pointer':three_per_3, 'Win':win_per_3}
    categories = list(data.keys())
    values = list(data.values())
    plt.ylim([0, 65])
    plt.bar(categories, values, width = 0.4, color = 'green', alpha = 0.5, label = '3-on-3')
    plt.ylabel("Percentage (%)")
    plt.title("Shot and win percentage for 5-on-5 and 3-on-3 games")
    plt.legend()

    #Attempts per game for both game types
    plt.figure()
    two_attg_5 = df[df['Game Type']==1]['Two-pointers attempted'].sum()/total_games
    two_attg_3 = df[df['Game Type']==0]['Two-pointers attempted'].sum()/total_games
    three_attg_5 = df[df['Game Type']==1]['Three-pointers attempted'].sum()/total_games
    three_attg_3 = df[df['Game Type']==0]['Three-pointers attempted'].sum()/total_games
    data = {'Two-Point-Attempts':two_attg_3, 'Three-Point-Attempts':three_attg_3}
    categories = list(data.keys())
    values = list(data.values())
    plt.bar(categories, values, width = 0.4, color = 'green', alpha = 0.5, label = '3-on-3')
    plt.ylabel("Attempts per games")
    plt.title("Two and Three Point attempts per game for each game type")
    data = {'Two-Point-Attempts':two_attg_5, 'Three-Point-Attempts':three_attg_5}
    categories = list(data.keys())
    values = list(data.values())
    plt.bar(categories, values, width = 0.4, color = 'red', alpha = 0.5, label = '5-on-5')
    plt.legend()

    #Scatter plot for average weather temperature and points scored
    plt.figure()
    plt.scatter(df["tavg"], df["Points"])
    plt.xlabel('Average Weather (Farenheight)', fontsize=15)
    plt.ylabel('Points', fontsize=15)
    plt.show()

    #Scatter for Wins and points scored
    plt.figure()
    plt.scatter(df["W/L"], df["Points"])
    plt.xlabel('Win/Loss', fontsize=15)
    plt.ylabel('Points', fontsize=15)
    plt.show()

def comp_given(df):
    '''
    Determine if more compliments were given during winning games
    Parameters df: data frame to be analyzed
    Returns: N/A
    '''
    #Organize data
    comp_given_win = df[df['W/L']==1]['Compliments given']
    print("Average compliments given for winning games: ", round(comp_given_win.mean(), 2))
    comp_given_win = comp_given_win.values.tolist()
    comp_given_loss = df[df['W/L']==0]['Compliments given']
    print("Average compliments given for losing games: ", round(comp_given_loss.mean(), 2))
    comp_given_loss = comp_given_loss.values.tolist()

    #Perfrom test with scipy
    from scipy import stats
    t, p_val = stats.ttest_ind(comp_given_win, comp_given_loss)
    p_val /= 2
    alpha = 0.01
    if (t > 0 and p_val < alpha):
        print("Jovan gaven more compliments during winning games")
    elif (t < 0 and p_val < alpha):
        print("Jovan gave less compliments during winning games")
    else:
        print("There is no difference between the amounts of compliments given for winning and losing games")
    
def tree(X_train, y_train, X_test, y_test, X):
    '''
    Decision Tree classifier predict and plot
    Parameters training and testing data: data frame to be cleaned
    Returns: predicted accuracy and precision and tree 
    '''
    from sklearn.tree import plot_tree
    from sklearn.tree import DecisionTreeClassifier
    import matplotlib.pyplot as plt

    tree = DecisionTreeClassifier(max_depth = 3)
    tree.fit(X_train, y_train)
    predicted = tree.predict(X_test)

    tree_acc = accuracy_score(predicted, y_test)
    tree_pre = precision_score(predicted, y_test)

    print("Accuracy of Decision Tree Classifier: ", tree_acc)
    print("Precision of Decision Tree Classifier: ", tree_pre)

    plt.figure()
    plt.figure(figsize = (15,10))
    t = plot_tree(tree, feature_names=X.columns, class_names={1: "Win", 0: "Loss"}, filled=True, fontsize=10)

def add_weather(df):
    lat, lng = get_lat_lng("Spokane")
    station_ID = get_weather_station_ID(lat, lng)
    weather_df = get_daily_weather_data(station_ID)
    weather_df = pd.json_normalize(weather_df)
    weather_df = weather_df[['date', 'tavg']]
    df = pd.merge(df, weather_df, on ='date')
    
    return df

def kNN(X_train, y_train, X_test, y_test):
    '''
    kNN classifier predict
    Parameters training and testing data: data frame to be cleaned
    Returns: predicted accuracy and precision and tree 
    '''
    neigh = KNeighborsClassifier(n_neighbors=2)
    neigh.fit(X_train, y_train)
    predicted = neigh.predict(X_test)

    kNN_acc = accuracy_score(predicted, y_test)
    kNN_pre = precision_score(predicted, y_test)

    print("Accuracy of kNN: ", kNN_acc)
    print("Precision of kNN: ",kNN_pre)
    
def main():
    '''
    Gets name of a city from user and writes out a csv file 
    containing weather data from the last year near that city
    '''
    #Prompt user for the name of a large city
    city = input("Enter the name of a large city: ")
    city_name = replace_space(city)

    #Using the user-entered city make a request to MapQuest to get the city's latitude and longitude
    lat, lng = get_lat_lng(city_name)
    print("Latitude:", lat)
    print("Longitude:", lng)

    #Using the latitude and longitude variables, make a request to MeteoStat to get the coordinates' station ID
    weather_station_ID = get_weather_station_ID(lat, lng)
    print("Weather station ID:", weather_station_ID)

    #Using your weather station ID variable, get daily weather data for the previous year (2021-02-21 through 2022-02-20)
    daily_data = pd.json_normalize(get_daily_weather_data(weather_station_ID))
    print("Original Weather Data:")
    print(daily_data.head(25))

    #Write the DataFrame to a csv file using the filename convention: <city name>_daily_weather.csv
    filename = city + "_daily_weather.csv"
    daily_data.to_csv(filename)

    #Clean the DataFrame so there are no missing values
    cleaned_data = clean(daily_data)
    print("Cleaned Weather Data:")
    print(cleaned_data.head(25))

    #Write the cleaned DataFrame to a csv file using the filename convention: <city name>_daily_weather_cleaned.csv
    filename = city + "_daily_weather_cleaned.csv"
    cleaned_data.to_csv(filename)

