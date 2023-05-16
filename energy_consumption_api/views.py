from django.shortcuts import render
from rest_framework.views import APIView, Response
from .apps import EnergyConsumptionApiConfig as Ecc


class PredictEnergyConsumptionApi(APIView):
    def post(self, request):
        request = request.data
        carbon = request['CO2(tCO2)']
        lagging = request['Lagging_Current_Power_Factor']
        week_status = request['WeekStatus']
        day_of_week = request['Day_of_week']
        load_type = request['Load_Type']
        predictor = Ecc.predictor

        predict_energy_consumption = predictor.predict([[
            carbon,
            lagging,
            week_status,
            day_of_week,
            load_type
        ]])

        response_dict = {'energy_consumed': predict_energy_consumption[0]}
        return Response(response_dict, status=200)

    def get(self, request):
        context = {'me': 'you'}
        return render(request, 'api_home.html', context)


predict_energy_consumption_api = PredictEnergyConsumptionApi.as_view()

