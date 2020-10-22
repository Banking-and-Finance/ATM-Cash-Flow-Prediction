from flask import Flask, request, render_template, redirect, url_for, session
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from datetime import datetime

app = Flask(__name__)

@app.route('/')
@app.route('/home')
@app.route('/<error>')
def home(error=None):
	return render_template('home.html', error=error)

@app.route('/results', methods=['POST', 'GET'])
def results():
	if request.method == 'POST':
		bs_data = pd.read_csv('static/data/cleaned_bs_data_flask_app.csv')
		bs_data['Transaction Date'] = pd.to_datetime(bs_data['Transaction Date'], format='%Y-%m-%d')

		input_date = datetime.strptime(request.form['date'], '%Y-%m-%d')

		test_row = bs_data[bs_data['Transaction Date'] == input_date].drop('Total amount Withdrawn', axis=1)

		if len(test_row) == 0:
			return redirect(url_for('home', error="error"))

		year = input_date.year
		model = CatBoostRegressor()
		model.load_model('static/models/Best_Catboost_Only_{}.cbm'.format(year))

		withdrawal_predicted = model.predict(test_row)[0]

		return render_template('results.html', test_row=test_row, result=int(withdrawal_predicted))


	return redirect(url_for('home'))













