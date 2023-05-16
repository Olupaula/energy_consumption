# Energy Consumption in Steel Industries
This project titled [Energy Consumption](https://en.wikipedia.org/wiki/Electric_energy_consumption) 
in [Steel](https://en.wikipedia.org/wiki/Steel) [Industries](https://en.wikipedia.org/wiki/Industry), 
deals with predicting what amount of energy will be consumed by a steel industry.
It is important to know the internal operations of industries and to estimate what it will cost to run these operations. 
One of the significant things that cannot be overlooked is the amount of electric energy that will be consumed by the 
industry which is directly proportional to the amount that will be spent therein.

There is a need to build a
[model](https://learn.microsoft.com/en-us/windows/ai/windows-ml/what-is-a-machine-learning-model) 
capable of predicting what amount of energy will be consumed by a steel industry, so that aspiring industrialists can 
come to terms with what they can bear and not go bankrupt after a short time as because things fail to go as planned.
A number of 
[regression models](https://learn.microsoft.com/en-us/training/modules/understand-regression-machine-learning/) were 
considered to see which model will be the best for predicting the energy consumption
of steel industries.

**Data Source**: [Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/nursery)

**Utilized Features**:
- Carbon Dioxide, CO2 (tCO2), Emission (ppm, i.e. Parts per million) ( Type: Continuous )
- Lagging Current Power Factor (%, i.e. percentage) ( Type: Continuous )
- Week Status ( Type: Categorical, i.e Week_status = { 'Weekday': 0, 'Weekend': 1 } )
- Day of Week ( Type: Categorical, i.e. Day_of_week = { 'Sunday': 0, 'Others' : 1 } )
- Load Type ( Type: Categorical, i.e Load_Type = { 'light_load': 0, 'other_load_types': 1 } )

**Data Target**:
- amount of energy consumed (Kilowatt hour (KWH)): a continuous variable


**Data Visualization**:

<p>
    <img src="./energy_consumption_images/plot_of_CO2 (tCO2)_against_electric_energy_consumption.png">
    <p>
        There is a predominant positive linear relationship between the amount of carbon dioxide (CO<sub>2</sub>) emission and electric energy 
        consumption. This means that the higher the amount of carbon dioxide emission, the higher the
        amount of electric energy consumption.
    </p>
</p>

<p>
    <img src="./energy_consumption_images/plot_of_Lagging Current Reactive Power kVarh_against_electric_energy_consumption.png">
    <p>
        There is also a positive linear relationship between Lagging Current Reactive Power (kVarh) and the amount of 
        electric energy consumption. This means that the higher the amount of lagging current reactive power, the more 
        amount of electric energy consumption.
    </p>
</p>
<p> 
    <img src="./energy_consumption_images/plot_of_Week Status_against_electric_energy_consumption.png">
    <p>
        The amount of energy consumed is much on weekdays than on weekends
    </p>
</p>


**Regression Techniques used**:
1. [Linear Regression](https://www.oxfordreference.com/display/10.1093/oi/authority.20110803100107226;jsessionid=BAD370C49344F63EAF545090E2E032DE)
2. [K-Nearest Neighbor (KNN)](ttps://online.stat.psu.edu/stat508/lesson/k)
3. [Decision Tree (DT)](https://online.stat.psu.edu/stat857/node/236/)
4. [Bayesian Ridge Regression](https://buildingblock.ai/bayesian-ridge-regression)

**Evaluation Metrics**: 
1. [Coefficient of Determination](https://www.oxfordreference.com/display/10.1093/oi/authority.20110803095621787#:~:text=In%20statistics%2C%20a%20measure%20of,Symbol%3A%20R2.)
2. [Mean Squared Error](https://statisticsbyjim.com/regression/mean-squared-error-mse/#:~:text=The%20calculations%20for%20the%20mean,by%20the%20number%20of%20observations.)

**The best Model**
When the using the metric Mean Squared Error, the model with the lowest mean squared model is the best among all models 
under consideration. However, while using the Coefficient of Determination, the model with the highest Coefficient of 
Variation is preferable. Using both metrics, the linear regression came up as the best model

[View Code on Kaggle](https://www.kaggle.com/oluade111/energy-consumption-notebook/)

[Use API]()


