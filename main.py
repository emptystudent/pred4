import pickle
import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import random
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
import requests

df = pd.read_csv('Book12.csv')
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
df['engine'] = pd.to_numeric(df['engine'], errors='coerce')
df['year_built'] = pd.to_numeric(df['year_built'], errors='coerce')
df = df[(df['year_built'] != 'Used') &
        (df['year_built'] != 1499.0) &
        (df['year_built'] != 1598.0) &
        (df['year_built'] != 2003.0) &
        (df['year_built'] != 1995.0)]
df.dropna(inplace=True)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_url_hello = "https://lottie.host/1ca73a63-6ef3-4571-a2ee-b7d1daaf66c6/kpqHFWxK0c.json"
lottie_url_download = "https://lottie.host/1ca73a63-6ef3-4571-a2ee-b7d1daaf66c6/kpqHFWxK0c.json"
lottie_hello = load_lottieurl(lottie_url_hello)
lottie_download = load_lottieurl(lottie_url_download)



st.title("Analysis and ML model of over 70k Car Data in Malaysia")
# Create the sidebar menu
st.sidebar.image("WhatsApp Image 2023-07-29 at 20.58.50.jpeg")


page = st.sidebar.selectbox("", ["Introduction🪔","Car price predictor Model 🔦", "I am a Data scientist💻", "I have a vehicle related business 💡","I am a Recruiter 💳 "])

if page == "Introduction🪔":
    st_lottie(lottie_hello)  #
    st.header("Introduction 🧪 ")
    st.subheader("Here Is how you can use the navigation Bar on your top left corner")
    st.write("")
    st.subheader(" ➥ Car price predictor Model 🔦")
    st.write(" Introducing My dynamic AI coalition – a fusion of Hypertuned **Decision Tree, Random Forest, KNN, and Gradient Boosting regressors, working in harmonious precision to predict your car's worth with an astonishing 97% accuracy**. 🎯🔮 Feed in your car details and see  AI magic! ")
    st.subheader(" ➥ I am a Data scientist💻")
    st.write("🔍 To understand more about what's going on under the hood of the project...the ML models and the statistics behind it")
    st.subheader(" ➥ I have a vehicle related business 💡")
    st.write("This Tab will  provide you an incredible insight on how you can buy and sell for your business or if you want to know more about cars Data in Malaysia!!.")
    st.subheader(" ➥ I am a Recruiter and would love to connect 💳 ")
    st.write("Find my contact info in this tab if you need me to add anything more!!!")


if page == "I have a vehicle related business 💡":
    st.header("Here is how you can buy and sell cars in Malaysia!! 🚗 ")

    lottie_url_hello3 = "https://lottie.host/e73a4d16-3b89-4b45-854e-1c58c227f5af/ilJjqxJTt3.json"
    lottie_url_download3 = "https://lottie.host/e73a4d16-3b89-4b45-854e-1c58c227f5af/ilJjqxJTt3.json"
    lottie_hello3 = load_lottieurl(lottie_url_hello3)
    lottie_download = load_lottieurl(lottie_url_download3) #def var
    st_lottie(lottie_hello3)

    st.write("lets have an overview of what the resell car markets looks in Malaysia ( counts are in Thousands)")
    df['company'] = df['company'].astype(str)

    # Get value counts for 'company' column
    fig = px.pie(df, names='location', title='Distribution of Cars by Location',
                 color_discrete_sequence=px.colors.qualitative.Pastel1)

    st.plotly_chart(fig)

    #NEXT PLOT ------------------
    st.write("As you can see the Majority of the car are near and around the capital not surprising? .... follow along!!")
    st.title("Price Depreciation Analysis")


    st.subheader("**Here, we have a bar plot showing the correlation between mileage and price for each company.**")
    st.write("The negative correlations between the resale price of cars and their mileage for different car companies, it suggest that as the mileage of cars from these companies increases, the resale price tends to decrease. In simpler terms, cars with higher mileage are generally associated with lower resale prices for these specific car manufacturers.")
    st.write("**So if you want to buy or sell a car in malaysia make sure that you evaluate the degradation over mileage.The more negative the correlation the faster the depreciation over mileage**")

    correlation_data = df.groupby('company')[['mileage', 'price', 'year_built']].corr().reset_index()

    # Filter the correlation data to keep only the correlations with 'price'
    correlation_data = correlation_data[correlation_data['level_1'] == 'mileage']

    # Rename the correlation column for clarity
    correlation_data.rename(columns={'price': 'correlation'}, inplace=True)

    # Create an interactive bar plot
    fig = px.bar(correlation_data, x='company', y='correlation', color='company',
                 labels={'company': 'Company', 'correlation': 'Correlation (Mileage vs. Price)'},
                 title='Correlation between Mileage and Price for Each Company Hover cursor over graph to see exact numbers')

    # Add a line through each bar
    fig.update_traces(marker=dict(line=dict(width=1, color='black')))

    # Display the plot
    st.plotly_chart(fig)

    #NEXT PLOT ------------

    grouped_df = df.groupby(['year_built', 'company']).agg({'price': 'mean'}).reset_index()

    def remove_outliers(group, std_devs=2):
        mean_price = group['price'].mean()
        std_price = group['price'].std()
        lower_bound = mean_price - std_devs * std_price
        upper_bound = mean_price + std_devs * std_price
        return group[(group['price'] >= lower_bound) & (group['price'] <= upper_bound)]


    grouped_df = grouped_df.groupby('company').apply(remove_outliers)

    companies = grouped_df['company'].unique()

    # Create the Streamlit app

    st.header("Price Depreciation Over Years for Different Car Companies")

    st.write("This analysis shows how the average price of cars changes over the years for different car companies.As you can see the furthher back we go in time the lower the average prrice of car with respect to brands.")
    st.write("**Our focus is on the Gradient (red line) as it shows how quickly the car depreciates over time for each brand.**")
    st.write("The higher the gradient i.e the more vertical the red line is the faster the car value depreciates over time in Malaysia. You must consider that when buying or selling a car,try to buy a car that depreciates slower so you can can sell it later again at a resonable price")

    # Display the dropdown widget
    selected_company = st.selectbox("Select Company", companies)
    def best_fit_line(x, y):
        coefficients = np.polyfit(x, y, 1)
        gradient = coefficients[0]
        intercept = coefficients[1]
        return gradient, intercept


    def update_plot(selected_company):
        filtered_df = grouped_df[grouped_df['company'] == selected_company]

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=filtered_df['year_built'], y=filtered_df['price'], mode='lines+markers',
                                 name=selected_company))

        gradient, intercept = best_fit_line(filtered_df['year_built'], filtered_df['price'])
        best_fit_line_values = [gradient * x + intercept for x in filtered_df['year_built']]

        fig.add_trace(go.Scatter(x=filtered_df['year_built'], y=best_fit_line_values, mode='lines',
                                 name=f'Best Fit (Gradient: {gradient:.2f})'))

        fig.update_layout(title=f"{selected_company} - Price Depreciation Over Years",
                          xaxis_title="Year Built",
                          yaxis_title="Average Price",
                          template="plotly_white")

        st.plotly_chart(fig)


    # Call the update_plot function with the selected company
    update_plot(selected_company)

    st.subheader("The next plot shows the gradient in a more readible and comparible manner")

    #NEXTPLOT---------------

    def remove_outliers(group, std_devs=2):
        mean_price = group['price'].mean()
        std_price = group['price'].std()
        lower_bound = mean_price - std_devs * std_price
        upper_bound = mean_price + std_devs * std_price
        return group[(group['price'] >= lower_bound) & (group['price'] <= upper_bound)]


    def best_fit_line(x, y):
        coefficients = np.polyfit(x, y, 1)
        gradient = coefficients[0]
        return gradient


    def calculate_gradients(dataframe):
        # Step 1: Group the DataFrame by 'year_built' and 'company' to calculate average price
        grouped_df = dataframe.groupby(['year_built', 'company']).agg({'price': 'mean'}).reset_index()

        # Step 2: Remove outliers using the standard deviation method for each company
        grouped_df = grouped_df.groupby('company').apply(remove_outliers)

        # Step 3: Calculate gradients for each company
        companies = grouped_df['company'].unique()
        gradients = {}
        for company in companies:
            filtered_df = grouped_df[grouped_df['company'] == company]
            gradient = best_fit_line(filtered_df['year_built'], filtered_df['price'])
            gradients[company] = gradient
        return gradients


    def plot_gradients_as_barchart(gradients):
        companies, gradient_values = zip(*gradients.items())

        fig = make_subplots(rows=1, cols=1)

        fig.add_trace(
            go.Bar(x=companies, y=gradient_values),
            row=1, col=1
        )

        fig.update_layout(title_text='Gradient Values for Each Company',
                          xaxis_title='Company',
                          yaxis_title='Gradient',
                          template='plotly_white')

        st.plotly_chart(fig)


    # Create the Streamlit app
    st.subheader("Gradient Values for Each Car Company")
    st.write("As you can see that Merc followed by lexus, and ford depreciates the fasted in the Malaysian market whereas, proton and perdua maintains it's relative values. This could because they are not luxury cars or other factors but in short make sure if you are buying a car with a higer gradient then sell it at the right time or it will depreciate quickly!!!")
    st.write("Each bar represents the gradient value of car depreciation over time.")

    gradients = calculate_gradients(df)
    plot_gradients_as_barchart(gradients)

    #NEXTPLO-------------
    def generate_random_color():
        color = "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        return color


    def update_plot(selected_company, selected_brand, selected_condition):
        if selected_company == 'All Companies':
            filtered_df = df
        else:
            filtered_df = df[df['company'] == selected_company]

        # Update available brands based on the selected company
        available_brands = ['All Models'] + filtered_df['model'].unique().tolist()
        if selected_brand == 'All Models':
            pass
        else:
            filtered_df = filtered_df[filtered_df['model'] == selected_brand]

        if selected_condition == 'All Conditions':
            pass
        else:
            filtered_df = filtered_df[filtered_df['condition'] == selected_condition]

        # Group the data by location and calculate the mean prices
        location_mean_prices = filtered_df.groupby('location')['price'].mean().reset_index()

        # Generate random colors for each brand
        colors = [generate_random_color() for _ in range(len(location_mean_prices))]

        # Create the interactive bar chart
        fig = make_subplots(rows=1, cols=1)

        fig.add_trace(
            go.Bar(x=location_mean_prices['location'], y=location_mean_prices['price'],
                   marker=dict(color=colors), name=selected_brand),
            row=1, col=1
        )

        fig.update_layout(title_text='Mean Prices of Each Model with Respect to Each Location',
                          xaxis_title='Location',
                          yaxis_title='Mean Price',
                          template='plotly_white')

        st.plotly_chart(fig)


    # Create the Streamlit app
    st.title("Locational Analysis")
    st.subheader("Now that you understand how price of different brands depreciate over time and mileage, the logical question is, what about location?")

    st.write("This analysis shows the mean prices of different car models with respect to different locations.Select the brand, model and location along with the condition to see how it deffers from location to location")
    st.write("Use the dropdown menus to filter the results based on company, model, and condition.")

    # Dropdown widgets for company, brand, and condition
    all_companies = ['All Companies'] + df['company'].unique().tolist()
    selected_company = st.selectbox("Company:", all_companies)
    filtered_df = df[df['company'] == selected_company] if selected_company != 'All Companies' else df
    available_brands = ['All Models'] + filtered_df['model'].unique().tolist()
    selected_brand = st.selectbox("Model:", available_brands)
    selected_condition = st.selectbox("Condition:", ['All Conditions', 'Used', 'New', 'Recon'])

    # Call the update_plot function with the selected values
    update_plot(selected_company, selected_brand, selected_condition)

    st.write("**Easy way to make money would be to buy from areas where the mean price of model is lower and sell at an area where the prices are higher, and don't worry this is not a static analysis, the dataset will automatically get updated every 120 days meaning you will have ever increasing dataset to lean on!!!**")

    #NEXT PLOT----

    def generate_random_color():
        return "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


    def update_plot(selected_location, selected_top_n):
        if selected_location == 'All Locations':
            filtered_df = df
        else:
            filtered_df = df[df['location'] == selected_location]

        brand_counts = filtered_df['model'].value_counts().reset_index()
        brand_counts.columns = ['model', 'Count']
        brand_counts = brand_counts.sort_values(by='Count', ascending=False).head(selected_top_n)

        colors = [generate_random_color() for _ in range(selected_top_n)]

        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Bar(x=brand_counts['model'], y=brand_counts['Count'], marker_color=colors, name='Count'))
        fig.update_layout(title_text=f'Top {selected_top_n} models in {selected_location}',
                          xaxis_title='Model',
                          yaxis_title='Count',
                          template='plotly_white')

        st.plotly_chart(fig)


    # Load your DataFrame (replace this with your actual DataFrame)
    # df = pd.read_csv('your_data.csv')

    st.subheader('Top Models Visualization')

    all_locations = ['All Locations'] + df['location'].unique().tolist()
    selected_location = st.selectbox('Select Location:', all_locations)

    selected_top_n = st.selectbox('Select Top N Models:', [6, 10, 15])

    update_plot(selected_location, selected_top_n)
    st.write('This visualisation shows the count of different models with respect to locations,the idea is that the greate the avalibility of a model in a certain location the **cheaper** the maintenence cost as the spare parts avalibility will be greater. So make sure to check the distibution of your model in yuor location to minimise the mainainces cost')
    st.subheader('Key takeaways')
    st.write(' ##### --> Most of the cars are distributed in and around the capital which makes a great market to buy and sell')
    st.write(' ##### --> The distritbuion of cars and its resell value is not the same from location to location, if you can buy from areas where the prices are low and sell in areas where the prices are higher for specific brands or companies, you can make upto 6k plus per car on average.')
    st.write(' ##### --> This Eda shows depreciation of cars by mileage and time which provides a great window into the timing and selling of cars or buying a second hand car for that matter. if you are looking for longivity then buy those brands that deprecciates at a gradual rate relative to those that depreciates faster but if you want to buy a luxury or imported car make sure to sell it in coupele of years to make the most out of the value')
    st.write(' ##### --> for furture analysis of the indept details of the project refer to the data science tab or use the car price predictor to predict the value of your car today!!!')

#NEXT TAB-----------------------------
if page == "I am a Data scientist💻":
    st.subheader('This section of the projects dives deep into the statistics logic and the working behind the model. **The goal is to help client understand what is going on under the hood but also more importantly showcase some good model building and coding practices for students to think about**')
    st.write("## Table of Contents")
    st.write("- [Data scrapping](#data-scrapping)")
    st.write("- [EDA](#eda)")
    st.write("- [Data preprocessing](#data-preprocessing)")
    st.write("- [Data Selection](#data-selection)")
    st.write("- [Data Modelling](#data-modelling)")
    st.write("- [Cross validation and Hyperparameter tuning](#cv-and-hyperparameter-tuning)")
    st.write("- [Data Testing and pipeline building](#data-testing)")
    st.write("- [Data Deployment](#data-deployment)")


    lottie_url_hello2 = "https://lottie.host/3dcc432d-13e8-4a56-a79f-2e4e3f6ad184/lq8PJnL47f.json"
    lottie_url_download2 = "https://lottie.host/3dcc432d-13e8-4a56-a79f-2e4e3f6ad184/lq8PJnL47f.json"
    lottie_hello2 = load_lottieurl(lottie_url_hello2)
    lottie_download = load_lottieurl(lottie_url_download2)

    st_lottie(lottie_hello2)

    #CONTENT OF EACH TABLE

    st.write("<h2 id='data-scrapping'>Data scrapping</h2>", unsafe_allow_html=True)
    st.write("I used **scrapy,Beautifulsoup, selenium** to automate the scraping bot for this and for future scraping. ")
    st.write(" DATA SCRAPED ----> FEEDS INTO AN CSV FILE ----> BOT RUNS VBA CODE TO CLEAN DATA -----> GETS PUSHED INTO PYCHARM ----> CODE RUNS AND BUILDS MODEL IN ONE CLICK ----> DEPLOYERD MODEL GETS UPDATED EVERY 120 DAYS ADDING 50K LISTING MORE ON EACH UPDATES")

    st.write("**In summary I scraped more then 70,000 listing and this is not static, the bot I created automatically scrapes and updates the CSV file every 6 months the listing will double. Meaning in a years time it may very well be over 150k, probably the largest data collection and analysis in malayasian Car resell market**")
    st.write("The scraping bot also called a **spider** In technical jargo automatically scrapes data every 120 days. The data gets pushed into CSV file whcih with a **click** runs VBA commands and it cleans the data brining it to the right format and then **python** code runs and trains the model, a **semi supervised** automated model :) cool!!!?????")
    st.write("Video below is a good representation of how it can be done")


    st.write("Here is what the dataFrame looks like after cleaning and scraping")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    st.write(df.head())
    st.write("There are 7 features and 1 target and 67250 non-null values in each column. All none values were droped as they accounted for less then 2% of the data. There would be no significant statistical impact of droping these null rows or semi filled rows. Columns were converted into the right datatype and the final cleaned information is as below")
    st.image("Screenshot 2023-08-06 153032.png")

    st.write("<h2 id='eda'>EDA</h2>", unsafe_allow_html=True)
    st.subheader("Here is the Exploratory Analysis of the variables")

    condition_counts = df['condition'].value_counts()

    condition_counts_df = pd.DataFrame({'condition': condition_counts.index, 'count': condition_counts.values})

    # Create the pie chart using Plotly Express
    fig = px.pie(condition_counts_df, values='count', names='condition', title='Distribution of car Conditions')
    st.plotly_chart(fig)

    #NEXT PLOT---------
    location_counts = df['location'].value_counts()

    # Calculate percentages
    total_count = location_counts.sum()
    location_percentages = (location_counts / total_count) * 100

    # Create a DataFrame for the pie chart
    location_data = pd.DataFrame({'Location': location_percentages.index, 'Percentage': location_percentages.values})

    # NEXT PLOT-------------
    fig = px.pie(location_data, values='Percentage', names='Location', title='Pie Chart of Location Distribution')
    st.plotly_chart(fig)

    st.write('As we can see from the Analysis of catagorical variables, the distibution of cars are mostly around the Capital,**this could be a inference to the development of Malaysia** The listing of resell cars fiddles out as we go away from the capital and we also know that the public transportation is not as good away from urban areas relative to rural areas.This analysis reflects that for the malaysian market. **This also provides oppertunities for businesses to try to get more listings in areas outside capital for the Malaysian markets**')
    # Get value counts for 'company' column
    company_counts = df['company'].value_counts().reset_index()
    company_counts.columns = ['Company', 'Count']

    # Next plot
    fig = px.bar(
        company_counts,
        x='Company',
        y='Count',
        title='Value Counts of Companies',
        labels={'Company': 'Company', 'Count': 'Count'}
    )

    st.plotly_chart(fig)

    #CORRELATION GRAPH

    selected_columns = ['price', 'mileage', 'engine', 'year_built']
    correlation_matrix = df[selected_columns].corr()
    fig = px.imshow(correlation_matrix, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)

    # Add correlation values as text annotations
    correlation_values = np.round(correlation_matrix.values, 2)
    for i in range(correlation_matrix.shape[0]):
        for j in range(correlation_matrix.shape[1]):
            fig.add_annotation(
                x=i,
                y=j,
                text=str(correlation_values[i, j]),
                showarrow=False,
                font=dict(color='white', size=12)
            )
    st.plotly_chart(fig)

    st.write("The relationships shown in the correlation graphs are expected which validates the purity of data scraping and a good sign for the future model building. It is not possible to infer anything from the values of correlations without future analysis.")

    def update_plot(selected_company):
        company_df = df[df['company'] == selected_company]

        fig = px.histogram(company_df, x='price', nbins=30, marginal='rug',
                           title=f'Distribution of Prices for {selected_company} Cars',
                           histnorm='probability', color_discrete_sequence=['rgb(255, 191, 0)'])

        return fig

    st.write('')
    st.write('')
    st.write('')

    # Create a select box to choose a car company
    car_companies = df['company'].unique()
    selected_company = st.selectbox('Select a Car Company to see price distribution probability', car_companies)
    fig = update_plot(selected_company)
    st.plotly_chart(fig)

    st.write('')
    st.write('')
    st.write('')

    def update_plot(selected_company):
        company_df = df[df['company'] == selected_company]

        fig = px.histogram(company_df, x='mileage', nbins=30, marginal='rug',
                           title=f'Distribution of Mileages for {selected_company} Cars',
                           histnorm='probability', color_discrete_sequence=['rgb(210,105,30)'])

        return fig

    st.write('')

    # Assuming df is your DataFrame containing car data

    # Create a select box to choose a car company
    car_companies = df['company'].unique()
    selected_company = st.selectbox('Select a Car Company to see the distribution probability of mileage', car_companies, key='company_select')

    # Provide a unique key for the widget
    fig = update_plot(selected_company)
    st.plotly_chart(fig)

    engine_counts = df['engine'].value_counts()

    st.write('')
    st.write('')
    st.write('')


    def update_pie_chart(num_engines):
        top_engines = engine_counts.head(num_engines)

        fig = go.Figure(data=[go.Pie(labels=top_engines.index, values=top_engines)])
        fig.update_layout(title_text=f'Top {num_engines} Engine Models in CC Engine Types')

        return fig



    # Display the pie chart using Streamlit
    num_engines = st.slider('Select the number of top engine models to display', min_value=1,
                            max_value=len(engine_counts), value=20)
    fig = update_pie_chart(num_engines)
    st.plotly_chart(fig)

    st.write('Percentage Distribution of cars distribution across the years ranging 2008 - 2023')

    year_built_counts = df['year_built'].value_counts()

    year_built_counts_df = pd.DataFrame({'Year Built': year_built_counts.index, 'Count': year_built_counts.values})

    year_built_counts_df = year_built_counts_df.sort_values(by='Count', ascending=False)

    total_count = year_built_counts_df['Count'].sum()
    year_built_counts_df['Percentage'] = (year_built_counts_df['Count'] / total_count) * 100

    # Round the Percentage column to 2 decimal places
    year_built_counts_df['Percentage'] = year_built_counts_df['Percentage'].round(2)

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(year_built_counts_df.columns),
                    fill_color='darkblue',  # Adjust the header fill color for a dark background
                    font_color='white',  # Adjust the header font color for a dark background
                    align='left'),
        cells=dict(values=[year_built_counts_df['Year Built'], year_built_counts_df['Count'],
                           year_built_counts_df['Percentage']],
                   fill_color='black',  # Adjust the cell fill color for a dark background
                   font_color='white',  # Adjust the cell font color for a dark background
                   align='left'))
    ])

    st.plotly_chart(fig)

    st.header('Analysis')
    st.write("A positively skewed distribution in car data, where values are concentrated on the lower end with a few outliers on the higher end, is common and not a major issue. It helps identify outliers, doesn't significantly affect analyses or models, and can be addressed through transformations if needed. It's crucial to understand the data's nature and consider appropriate steps based on analysis goals.")
    st.write("However, skewness of data can be an issue for **model building** and outliers are generally removed or the data normalised.However, part of the skewness could also be due to the fact that there are different models within each company and some models may be on the luxuray side and beacuse of that I am not going to remove outliers in the model building. Instead am going to use **Four different hypertuned models that will take into account these variation such that the accuracy of prediction is not hampered**")
    st.write("Most of the cars listed are between 2013-2018 Feel free to use all the interactive grpahs to know more")
#PREPROCESSING-------------------------------------------------------------DATA SCI TAB
    st.write("<h2 id='data-preprocessing'>Data preprocessing</h2>", unsafe_allow_html=True)
    st.header('This part of the project focus on how the Machine learning model was built and the statistical logic behind it')
    code_to_display = '''
    from sklearn.linear_model import LinearRegression, ElasticNet, HuberRegressor, RANSACRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.svm import SVR
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import GridSearchCV
    from tabulate import tabulate
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    '''

    st.subheader('Importing libraries')

    st.code(code_to_display, language='python')

    code_to_display = '''
    X = df.drop('price', axis=1)  # Drop 'price' column
    y = df['price']  # Target variable

    # Splitting the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Further splitting the training data into training and validation sets
    X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    # Define the column names of categorical and numeric columns
    categorical_cols = ['condition', 'model', 'company', 'location']
    numeric_cols = ['mileage', 'engine', 'year_built']

    # Creating the transformers for the ColumnTransformer
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numeric_transformer = StandardScaler()

    # Create the ColumnTransformer with specified transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols),
            ('num', numeric_transformer, numeric_cols)
        ])
    X_train_final_transformed = preprocessor.fit_transform(X_train_final)

    X_val_transformed = preprocessor.transform(X_val)
    X_test_transformed = preprocessor.transform(X_test)
    '''

    st.subheader('split and column transformation')
    st.write("I use onehotencoder to transform the catagorical variables and standard scalar for numeric variables. **To be fair we don't need to transform numeric variables for all the ML models but there is no harm in doing so as it allows integration of many more models which do need scaling such as KNN which rely on eculidien distance to predict and create the model**")

    st.code(code_to_display, language='python')

    st.write("<h2 id='data-selection'>Data Selection</h2>", unsafe_allow_html=True)

    code_to_display = '''
    # Define the models
    linear_reg_model = LinearRegression(n_jobs=-1)
    elastic_net_model = ElasticNet()
    huber_model = HuberRegressor(max_iter=1000, alpha=0.1)
    ransac_model = RANSACRegressor()
    decision_tree_model = DecisionTreeRegressor()
    knn_model = KNeighborsRegressor(n_jobs=-1)
    random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    gradient_boosting_model = GradientBoostingRegressor(random_state=42)
    polynomial_reg_model = LinearRegression(n_jobs=-1)  
    svr_model = SVR()  # For SVR

    # List of models
    models = [
        ('Linear Regression', linear_reg_model),
        ('ElasticNet Regression', elastic_net_model),
        ('Huber Regression', huber_model),
        ('RANSAC Regression', ransac_model),
        ('Decision Tree Regression', decision_tree_model),
        ('K-Nearest Neighbors Regression', knn_model),
        ('Random Forest Regression', random_forest_model),
        ('Gradient Boosting Regressor', gradient_boosting_model),
        ('Polynomial Regression', polynomial_reg_model),
        ('Support Vector Regression', svr_model)
    ]

    results = []
    # Loop over the models and perform cross-validation on the training set
    for name, model in models:
        scores = cross_val_score(model, X_train_final_transformed, y_train_final, cv=10, scoring='neg_mean_squared_error')
        mse = abs(scores.mean())

        # Fit the model on the whole training set and evaluate on the validation set
        model.fit(X_train_final_transformed, y_train_final)
        y_pred_val = model.predict(X_val_transformed)
        r2_val = r2_score(y_val, y_pred_val)

        # Append the results to the list
        results.append((name, mse, r2_val))

    # Convert the results list into a Pandas DataFrame
    results_df = pd.DataFrame(results, columns=['Model', 'Cross-validated MSE', 'Validation R2 Score'])

    # Display the results
    print(results_df)
    '''

    st.subheader('selecting and running different models')
    st.write("When it comes to model selection there is no harm in running many models and finding out which one works best. **My philosophy is to always start with atleat 10 models and then select top 4 which can be hypertuned and perfected**. Beacuse a varity in models can balance the overfitting and underfitting when the data sees the new data. But we must also balance the time constraints, as you can see I have not choosen to use any deep learning models as it may be overcooking the data we have and it's important to not make a product complex for no valid reason")

    st.code(code_to_display, language='python')

    st.subheader(" The data is split 80-20 and then the training set is split further to develop validation set")
    st.write("We put aside the test set for later use and focus on evaluating on validation set first")

    results_data = [
        ('Linear Regression', 1.848858, 0.862302),
        ('ElasticNet Regression', 5.084030, 0.589342),
        ('Huber Regression', 4.821918, 0.613927),
        ('RANSAC Regression', 4.421839, 0.626865),
        ('Decision Tree Regression', 5.985303, 0.964642),
        ('K-Nearest Neighbors Regression', 4.908798, 0.971245),
        ('Random Forest Regression', 4.715588, 0.975719),
        ('Gradient Boosting Regressor', 8.640424, 0.948948),
        ('Polynomial Regression', 1.848858, 0.862302),
        ('Support Vector Regression', 1.398327, -0.099223)
    ]

    results_df = pd.DataFrame(results_data, columns=['Model', 'Cross-validated MSE', 'Validation R2 Score'])

    st.subheader('Results')

    # Display the results table
    st.table(results_df)

    st.write("<h2 id='data-modelling'>Data Modelling</h2>", unsafe_allow_html=True)
    code_to_display = '''
    # Define hyperparameters for each model
    dt_params = {
        'max_depth': [None, 15, 20, 25],
        'min_samples_split': [5, 10, 15],
        'min_samples_leaf': [1, 2, 4]
    }

    knn_params = {
        'n_neighbors': [7, 9, 11, 13],
        'weights': ['uniform', 'distance']
    }

    rf_params = {
        'n_estimators': [100, 150, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    gradient_boosting_params = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.5],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Create model instances
    dt_model = DecisionTreeRegressor(random_state=42)
    knn_model = KNeighborsRegressor(n_jobs=-1)
    rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    gradient_boosting_model = GradientBoostingRegressor(random_state=42)

    # Create a list of models and their respective hyperparameters
    models_hyperparams = [
        (dt_model, dt_params),
        (knn_model, knn_params),
        (rf_model, rf_params),
        (gradient_boosting_model, gradient_boosting_params)
    ]

    best_params_dict = {}  # Initialize the dictionary to store best hyperparameters

    for model, params in models_hyperparams:
        if params:
            # Perform GridSearchCV to find the best hyperparameters
            grid_search = GridSearchCV(model, params, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
            grid_search.fit(X_train_final_transformed, y_train_final)
            best_params_dict[model.__class__.__name__] = grid_search.best_params_

    # Print the best hyperparameters for each algorithm
    for model_name, best_params in best_params_dict.items():
        print(f"Best Hyperparameters for {model_name}: {best_params}")
    '''

    st.subheader('selecting the top 4 performing models and hypertuning')
    st.write("Honestly, these models in default state are performing really well and hypertuning won't really improve much but am doing it anyways as this is part of my portfolio project and there is no harm in going the distance. If you choose to go ahead with default models it should still be fine.")

    st.code(code_to_display, language='python')

    hyperparameters_results = [
        ("DecisionTreeRegressor", {'max_depth': 25, 'min_samples_leaf': 1, 'min_samples_split': 15}),
        ("KNeighborsRegressor", {'n_neighbors': 7, 'weights': 'uniform'}),
        ("RandomForestRegressor",
         {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}),
        ("GradientBoostingRegressor",
         {'learning_rate': 0.5, 'max_depth': 5, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 150})
    ]

    st.subheader('Hyperparameters Results')
    st.write(" I am using **GRIDSEARCHCV** to find out the best parms for the top 4 models with 10 folds to get the best combination")

    # Display the hyperparameters results
    for model_name, best_params in hyperparameters_results:
        st.write(f"Best Hyperparameters for {model_name}: {best_params}")


    # Add content related to Data Modelling here

    st.write("<h2 id='cv-and-hyperparameter-tuning'>Cross validation and Hyperparameter tuning</h2>", unsafe_allow_html=True)

    code_to_display = '''
    # Define the best hyperparameters for each model
    best_dt_params = {
        'max_depth': 25,
        'min_samples_leaf': 1,
        'min_samples_split': 15
    }

    best_knn_params = {
        'n_neighbors': 7,
        'weights': 'uniform'
    }

    best_rf_params = {
        'max_depth': None,
        'min_samples_leaf': 1,
        'min_samples_split': 5,
        'n_estimators': 200
    }

    best_gb_params = {
        'learning_rate': 0.5,
        'max_depth': 5,
        'min_samples_leaf': 4,
        'min_samples_split': 10,
        'n_estimators': 150
    }

    # Create model instances with the best hyperparameters
    best_dt_model = DecisionTreeRegressor(**best_dt_params, random_state=42)
    best_knn_model = KNeighborsRegressor(**best_knn_params, n_jobs=-1)
    best_rf_model = RandomForestRegressor(**best_rf_params, random_state=42, n_jobs=-1)
    best_gb_model = GradientBoostingRegressor(**best_gb_params, random_state=42)

    # Fit the tuned models on the training data
    best_dt_model.fit(X_train_final_transformed, y_train_final)
    best_knn_model.fit(X_train_final_transformed, y_train_final)
    best_rf_model.fit(X_train_final_transformed, y_train_final)
    best_gb_model.fit(X_train_final_transformed, y_train_final)

    # Make predictions on the validation set
    y_pred_dt = best_dt_model.predict(X_val_transformed)
    y_pred_knn = best_knn_model.predict(X_val_transformed)
    y_pred_rf = best_rf_model.predict(X_val_transformed)
    y_pred_gb = best_gb_model.predict(X_val_transformed)

    # Evaluate the predictions using MSE and R2 score
    mse_dt = mean_squared_error(y_val, y_pred_dt)
    r2_dt = r2_score(y_val, y_pred_dt)

    mse_knn = mean_squared_error(y_val, y_pred_knn)
    r2_knn = r2_score(y_val, y_pred_knn)

    mse_rf = mean_squared_error(y_val, y_pred_rf)
    r2_rf = r2_score(y_val, y_pred_rf)

    mse_gb = mean_squared_error(y_val, y_pred_gb)
    r2_gb = r2_score(y_val, y_pred_gb)

    # Display the results
    print("DecisionTreeRegressor:")
    print(f"Validation MSE: {mse_dt}")
    print(f"Validation R2 Score: {r2_dt}")
    print()

    print("KNeighborsRegressor:")
    print(f"Validation MSE: {mse_knn}")
    print(f"Validation R2 Score: {r2_knn}")
    print()

    print("RandomForestRegressor:")
    print(f"Validation MSE: {mse_rf}")
    print(f"Validation R2 Score: {r2_rf}")
    print()

    print("GradientBoostingRegressor:")
    print(f"Validation MSE: {mse_gb}")
    print(f"Validation R2 Score: {r2_gb}")
    print()
    '''

    st.subheader('Tuned Model Evaluation Code Example')
    st.write("**I could have done this step along wiht the previous one but i prefer to do it separetly and with a neat code beacuse i will use the same variables later for building pipeline!!**")

    st.code(code_to_display, language='python')

    results_dict = {
        'Model': ['DecisionTreeRegressor', 'KNeighborsRegressor', 'RandomForestRegressor', 'GradientBoostingRegressor'],
        'Validation MSE': [364765594, 365017344, 303969273, 370060031],
        'Validation R2 Score': [0.9685716361626315, 0.968549945270658, 0.9738098739045825, 0.9681154651715169]
    }

    results_df = pd.DataFrame(results_dict)

    st.subheader('Model Evaluation Results')
    st.write(" As expected the Hypertuned models did not improve the standard model parms much because they were very high to begin with but I see no harms in improving a little bit more :) ")

    # Display the results table
    st.table(results_df)

    code_to_display = '''
    y_pred_dt = best_dt_model.predict(X_test_transformed)
    y_pred_knn = best_knn_model.predict(X_test_transformed)
    y_pred_rf = best_rf_model.predict(X_test_transformed)
    y_pred_gb = best_gb_model.predict(X_test_transformed)

    # Combine predictions using simple averaging
    y_pred_combined = (y_pred_dt + y_pred_knn + y_pred_rf + y_pred_gb) / 4

    # Evaluate the combined predictions using MSE and R2 score
    mse_combined = mean_squared_error(y_test, y_pred_combined)
    r2_combined = r2_score(y_test, y_pred_combined)

    # Display the combined results
    print("Combined Model:")
    print(f"Test MSE: {mse_combined}")
    print(f"Test R2 Score: {r2_combined}")
    '''

    st.header('Combining and averaging the results from all four models')
    st.subheader("As you can see the individual models on their are very good **but the MSE can be lowered even more when we combine them** and this is why I choose to combine them all together to balance the outputs out")

    st.code(code_to_display, language='python')

    results_dict = {
        'Metric': ['Test MSE', 'Test R2 Score'],
        'Combined Model': [269801000, 0.978538348208262]
    }

    results_df = pd.DataFrame(results_dict)

    # Display the results table
    st.table(results_df)

    st.write("<h2 id='data-testing'>Data Testing and Pipleline building</h2>", unsafe_allow_html=True)

    code_to_display = '''
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.ensemble import VotingRegressor, RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsRegressor
    '''

    st.header('Importing and Setting up the Model pipelines')
    st.subheader("Why am I making pipeline now?!!")
    st.write("Most people usually build pipelines along with the column transformation but in my experience although this is effecient and saves times but it can make it very complex espically if you are working with a team and espically if the team does not have everyone tecnhically sound. IT IS  BETTER  TO BUILD PIPELINE LATER to make sure everyone understands the code and to be fair creating  pipleines later mainly involves copying the same code and editing a little and therefore is not time consuming. Furthurmore, you can always come later and play around with the model without touching the pipline which will be used for the model deployment. It is not the most effeceint strategy I admit but it makes the worklife of the team more **Enjoyable**")

    st.code(code_to_display, language='python')

    code_to_display = '''
    X = df.drop('price', axis=1)  # Drop 'price' column
    y = df['price']  # Target variable

    # Splitting the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Further splitting the training data into training and validation sets
    X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    '''

    st.header('Data Splitting and Preprocessing')

    st.code(code_to_display, language='python')

    code_to_display = '''
    categorical_cols = ['condition', 'model', 'company', 'location']
    numeric_cols = ['mileage', 'engine', 'year_built']

    # Create transformers for the ColumnTransformer
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numeric_transformer = StandardScaler()

    # Create the ColumnTransformer with specified transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols),
            ('num', numeric_transformer, numeric_cols)
        ])

    # Define best hyperparameters
    best_knn_params = {
        'n_neighbors': 7,
        'weights': 'uniform'
    }

    best_rf_params = {
        'max_depth': None,
        'min_samples_leaf': 1,
        'min_samples_split': 5,
        'n_estimators': 200
    }

    best_gb_params = {
        'learning_rate': 0.5,
        'max_depth': 5,
        'min_samples_leaf': 4,
        'min_samples_split': 10,
        'n_estimators': 150
    }

    best_dt_params = {
        'max_depth': 25,
        'min_samples_leaf': 1,
        'min_samples_split': 15
    }

    # Create tuned models
    knn_tuned_model = KNeighborsRegressor(**best_knn_params)
    rf_tuned_model = RandomForestRegressor(**best_rf_params)
    gb_tuned_model = GradientBoostingRegressor(**best_gb_params)
    dt_tuned_model = DecisionTreeRegressor(**best_dt_params)

    # Create pipelines for each model
    knn_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', knn_tuned_model)
    ])

    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', rf_tuned_model)
    ])

    gb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', gb_tuned_model)
    ])

    dt_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', dt_tuned_model)
    ])

    # Fit the pipelines on the training data
    knn_pipeline.fit(X_train_final, y_train_final)
    rf_pipeline.fit(X_train_final, y_train_final)
    gb_pipeline.fit(X_train_final, y_train_final)
    dt_pipeline.fit(X_train_final, y_train_final)

    # Make predictions using the pipelines on the validation set
    knn_val_predictions = knn_pipeline.predict(X_val)
    rf_val_predictions = rf_pipeline.predict(X_val)
    gb_val_predictions = gb_pipeline.predict(X_val)
    dt_val_predictions = dt_pipeline.predict(X_val)

    # Calculate the average validation prediction
    average_val_predictions = (knn_val_predictions + rf_val_predictions + gb_val_predictions + dt_val_predictions) / 4

    # Make predictions using the pipelines on the test set
    knn_test_predictions = knn_pipeline.predict(X_test)
    rf_test_predictions = rf_pipeline.predict(X_test)
    gb_test_predictions = gb_pipeline.predict(X_test)
    dt_test_predictions = dt_pipeline.predict(X_test)

    # Calculate the average test prediction
    average_test_predictions = (knn_test_predictions + rf_test_predictions + gb_test_predictions + dt_test_predictions) / 4
    '''

    st.header('Model Pipelines and Predictions')

    st.code(code_to_display, language='python')

    results_dict = {
        'Metric': ['R-squared score for Average Validation Predictions',
                   'R-squared score for Average Test Predictions'],
        'Value': [0.97539, 0.97858]
    }

    results_df = pd.DataFrame(results_dict)


    st.subheader(" NOTE : It is only now that I am using the TEST SET which makes the authenticity of the model much better. Also note that the code for pipline is very neat which makes the living of the team very easy as they can in the future come back and eidt the pipe and the model building sperately which makes it easy for anyone new or old in the team to work with")
    st.write(" Always Remember you are wokring with a team and the code should be as easy to follow as if an intern was reading")
    st.subheader('Model Evaluation Results')
    # Display the results table
    st.table(results_df)
    # Add content related to Data Testing here

    st.write("<h2 id='data-deployment'>Data Deployment</h2>", unsafe_allow_html=True)

    code_to_display = '''
    import pickle

    # Save the Gradient Boosting pipeline
    with open('gb_pipeline.pkl', 'wb') as gb_file:
        pickle.dump(gb_pipeline, gb_file)

    # Save the KNN pipeline
    with open('knn_pipeline.pkl', 'wb') as knn_file:
        pickle.dump(knn_pipeline, knn_file)

    # Save the Random Forest pipeline
    with open('rf_pipeline.pkl', 'wb') as rf_file:
        pickle.dump(rf_pipeline, rf_file)

    # Save the Decision Tree pipeline
    with open('dt_pipeline.pkl', 'wb') as dt_file:
        pickle.dump(dt_pipeline, dt_file)
    '''

    st.subheader('Saving Model Pipelines')
    st.write(' You can do this using joblib too')

    st.code(code_to_display, language='python')

    code_to_display = '''
    sample_data = {
        'condition': 'Used',
        'mileage': 149999,
        'year_built': 2010,
        'engine': 1495,
        'location': 'Kuala Lumpur',
        'company': 'Perodua',
        'model': 'Alza'
    }

    # Convert the sample data into a DataFrame
    sample_df = pd.DataFrame([sample_data])

    # Use the loaded pipelines to make predictions
    average_predictions = (
        loaded_gb_pipeline.predict(sample_df) +
        loaded_knn_pipeline.predict(sample_df) +
        loaded_rf_pipeline.predict(sample_df) +
        loaded_dt_pipeline.predict(sample_df)  # Adding Decision Tree prediction
    ) / 4

    st.title('Sample Data Prediction')

    st.write("Sample Data:")
    st.write(sample_df)
    st.write("\nAverage Predicted Price:")
    st.write(average_predictions)
    '''

    st.subheader('Using Loaded Pipelines for Predictions')
    st.write("This is a sample of what is going on under the hood of the car price prediction models.")
    st.write("It uses combined results from four models combined. **It may be over kill to be frank, a single model, even without tuning was producing good results, but in order of presentation It would not hurt to go all the way in and fine tune even if it's marginal**")

    st.code(code_to_display, language='python')

    st.subheader(" The model has 97% accuracy")
    st.subheader("Please go and try the model and provide me the feedback and I hope this helped")




# Car Price Predictor Page
if page == "Car price predictor Model 🔦":
    st.header("Car Price Predictor")
    st.subheader("The combined model has accuracy of 97%")
    st.write(" All car models are automatic transmission. If your car model is not here please let me know so I may add!!")
    st.write("input the values and press the **Predict** button and then press **Know more** to see the magic")

    with open('dt_pipeline.pkl', 'rb') as file:
        dt_pipeline = pickle.load(file)

    with open('knn_pipeline.pkl', 'rb') as file:
        knn_pipeline = pickle.load(file)

    with open('gb_pipeline.pkl', 'rb') as file:
        gb_pipeline = pickle.load(file)


    def predict_price(condition, mileage, year_built, engine, location, company, model):
        sample_data = {
            'condition': condition,
            'mileage': mileage,
            'year_built': year_built,
            'engine': engine,
            'location': location,
            'company': company,
            'model': model
        }

        sample_df = pd.DataFrame([sample_data])

        average_predictions = (
                                      gb_pipeline.predict(sample_df) +
                                      knn_pipeline.predict(sample_df) +
                                      dt_pipeline.predict(sample_df)
                              ) / 3

        return average_predictions[0]



    # User inputs
    condition_options = ["New", "Used", "Recon"]
    condition = st.selectbox("Condition", condition_options)

    mileage = st.number_input("Mileage", value=0)

    year_built_options = list(range(2008, 2024))
    year_built = st.selectbox("Year Built", year_built_options)

    location_options = df['location'].unique()
    location = st.selectbox("Location", location_options)

    company_options = df['company'].unique()
    selected_company = st.selectbox("Company", company_options)

    # Filter models based on selected company
    models_for_selected_company = df[df['company'] == selected_company]['model'].unique()
    model = st.selectbox("Model", models_for_selected_company)

    engine = st.number_input("Engine Size in CC", value=1000)



    if st.button("Predict Price"):
        # Get the predicted price
        predicted_price = predict_price(condition, mileage, year_built, engine, location, selected_company, model)

        # Display the predicted price
        prediction_placeholder = st.empty()
        prediction_placeholder.success(f"Predicted Price: RM {predicted_price:.2f}")

    if st.button("Know More"):
        st.write("## Undervalued")
        st.write("The model has a high accuracy of 97% but it only considers 7 features to predict the car's price.")
        st.write("It's possible that some additional features might be influencing the car's value such as:")
        st.write("- Desirable Brand")
        st.write("- Classic or Vintage Status")
        st.write("- Recent Repairs/Restorations")
        st.write("- Popular Colors")
        st.write("- Well-Maintained Interior/Exterior")
        st.write("- Seller's Perception")
        st.write("- Celebrity Ownership")
        st.subheader(
            "If your car has any of the above additional features, make sure to highlight them to potential buyers to avoid underselling.")

        st.write("## Overvalued")
        st.write("The model has a high accuracy of 97% but it only considers 7 features to predict the car's price.")
        st.write("It's possible that some additional features might be influencing the car's value such as:")
        st.write("- Accident History")
        st.write("- Maintenance Records")
        st.write("- Outdated Technology")
        st.write("- Non-Standard Modifications")
        st.write("- High Ownership Count")
        st.write("- Economic Factors")
        st.write("- Worn Interior")
        st.write("- Lack of Features")
        st.subheader( "If your car has any of the above additional features, make sure to provide context to potential buyers to avoid overselling.")

if page == "I am a Recruiter 💳 ":
        lottie_url_hello1 = "https://lottie.host/2221de94-e486-4318-a12c-4684eac24f1b/EoB0dBiLkM.json"
        lottie_url_download1 = "https://lottie.host/2221de94-e486-4318-a12c-4684eac24f1b/EoB0dBiLkM.json"
        lottie_hello1 = load_lottieurl(lottie_url_hello1)
        lottie_download = load_lottieurl(lottie_url_download1)
        st_lottie(lottie_hello1)


          
        st.subheader("Here are my contact details if you want me to make any Improvements or if you wish to contact me!!")
        st.write("**Name:** Muhammd Usman")
        st.write("**Email:** uthmanalfaris@gmail.com")
        st.write("**Phone:** +60 123103231")
        st.write("**LinkedIn:** [My LinkedIn Profile](https://www.linkedin.com/in/muhammed-usman-224188134/)")
