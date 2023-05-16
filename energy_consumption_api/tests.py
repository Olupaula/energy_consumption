from django.test import TestCase
import requests

endpoint = "http://127.0.0.1:8000/energy_consumption_api"
data_dict = {
        'CO2(tCO2)': 0.02,
        'Lagging_Current_Power_Factor': 68,
        'WeekStatus': 1,
        'Day_of_week': 0,
        'Load_Type': 1
}

predict = requests.post(endpoint, json=data_dict)

energy_consumption_dictionary = predict.json()
print("energy consumption dictionary = ",  energy_consumption_dictionary)

energy_consumed = energy_consumption_dictionary['energy_consumed']
print("energy consumed = ", energy_consumed)
