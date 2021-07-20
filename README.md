# MLTrading-Algo
A simple web application to visualize stock prediction algorithms on the S&P 500. The application not only provides nice graphic analysis tools like Candlestick Plots, but also provides an environment to train models including linear regression and RNN's for stock analysis. Can also test these models on past data (for comparison).
# Usage
### Setup Process
1. Download Git Repository to Local Machine
    - (Optional) Create a Virtual Environment for this Project
    - (Optional) Start up the Virtual Environment
2. Install the Required Libraries by Running the Following Command
    - ```
        pip3 install -r requirements.txt
        ```
### Running the Application
1. Use Streamlit to start up the GUI locally at 'localhost: 8501'
    - ```
        streamlit run algo.py
        ```
2. Select the Number of S&P 500 Stocks to Consider in Data Analysis and Press "Fetch Data" so that the application can download appropriate financial data.

3. Select the Appropriate Visualization Tools from Dropdown and Press "Generate Feature Set" to view plots and get training data.

4. Choose an Appropriate Model from Dropdown and Specify the Number of Training Epochs [other parameters are default]. 

5. Press Train Model to Generate/Save Model Checkpoint File and view comparison between predicted values and actual values.