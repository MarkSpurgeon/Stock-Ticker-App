#Load the packages
import pandas as pd
from flask import Flask,render_template,request,redirect,url_for
from bokeh.embed import components 
from bokeh.models import HoverTool
from bokeh.charts import Scatter
import requests
import json
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
import random

#Connect the app
app = Flask(__name__)

#global app.vars
app.vars = {}

def get_plot(closing_prices, company_name, month, year, line_color="black"):
    
    p = figure(plot_width=600, plot_height=400)
    p.line(list(closing_prices.keys()), list(closing_prices.values()), line_color=line_color)
    p.xaxis.axis_label = 'Day' 
    p.yaxis.axis_label = 'Closing Price (USD)'
    p.title.text = '%s Closing Prices (%s/%s)' % (company_name.upper(), month, year)
    p.add_tools(HoverTool())
    return(p)

@app.route('/',methods=['GET','POST'])
def start1():
    
    if request.method == 'GET':
        return render_template('userinfo_lulu.html')
    else:
        company_name = request.form['name_lulu']
        month = str(request.form['age_lulu'])
        #year = str(request.form['year_lulu'])

        if not company_name:
            company_name = random.choice(['wmt', 'tgt', 'fl', 'hos'])
            
    year = '2018'

    days_str = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
               '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
               '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
    closing_prices = {}
    for day in days_str:
        date_code = '%s%s%s' % (year, month, day)
        url = 'https://api.iextrading.com/1.0/stock/%s/chart/date/%s' % (company_name, date_code)
        data = requests.get(url).json()
        if data:
            if data[-1]['marketClose'] == 0: 
                closing_prices[int(day)] = data[-2]['marketClose']
            else:
                closing_prices[int(day)] = data[-1]['marketClose'] 

    start_price = list(closing_prices.values())[0]
    end_price = list(closing_prices.values())[-1]
    if end_price - start_price > 0: 
        line_color = random.choice(['blue', 'green'])
    else:
        line_color = random.choice(['orange', 'red'])        
            
    #Setup plot
    p = get_plot(closing_prices, company_name, month, year, line_color)
    script, div = components(p)

    #Render the page
    return render_template('home.html', script=script, div=div)    

if __name__ == '__main__':
    app.run(debug=False)