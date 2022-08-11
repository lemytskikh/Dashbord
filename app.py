
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash import dash_table

import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio

import pandas as pd
import math
import requests
import numpy as np


#pio.renderers.default = "browser"

df =  pd.read_csv(r'D:\LEM\Documents\Jupyter\Эксперименты\dash\2. Cars Data1.csv')

description = pd.read_excel(r'D:\LEM\Documents\Jupyter\Эксперименты\dash\описание.xlsx')


df = df.dropna()
df.columns = map(str.lower, df.columns)

df['msrp'] = (df['msrp'].str.replace('$', '', regex=True)
             .str.replace(',', '', regex=True)
             .astype('int64'))
df['invoice'] = (df['invoice'].str.replace('$', '', regex=True)
                .str.replace(',', '', regex=True)
                .astype('int64'))

df['mpg_mean'] = (df['mpg_city'] + df['mpg_highway']) / 2


# преобразуем расход топлива в из галонов в литры
df['mpg_city'] = round(df['mpg_city'] * 3.785)
df['mpg_highway'] = round(df['mpg_highway'] * 3.785)


# SELECTORS
selector = dcc.RangeSlider(
    id = 'range',
    min = min(df['enginesize']),
    max = max(df['enginesize']),
    marks = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4',5:'5', 6:'6', 7:'7', 8:'8', 9:'9'},
    step = 1,
    value = [min(df['enginesize']), max(df['enginesize'])]
    )

#Мощность

hp_bins = [0, 100, 200, 300,500]
hp_labels = ['маленький', 'оптимальный', 'высокий', 'сумашедший']
df['hp_labels'] = pd.cut(df['horsepower'], hp_bins, labels=hp_labels)

#Расход

mh_bins = [0, 50, 100, 200, 250]
mh_labels = ['маленький', 'оптимальный', 'высокий', 'сумашедший']
df['mh_labels'] = pd.cut(df['mpg_highway'], mh_bins, labels=mh_labels)


#Тип машины по расходу и мощности
df['status'] = np.where((df['hp_labels'] == 'оптимальный') & (df['mh_labels'] == 'оптимальный'), 'оптимальный', None)

df.loc[:, 'status'] = np.where((df['hp_labels'] == 'оптимальный') & 
                               (df['mh_labels'].isin(['маленький', 'высокий'])), 'и_так_пойдёт', df['status'])

df.loc[:, 'status'] = np.where((df['mh_labels'] == 'оптимальный') & 
                               (df['hp_labels'].isin(['маленький', 'высокий'])), 'и_так_пойдёт', df['status'])

df['status'] = df.status.fillna('ну_ваще')




options_type = []
name_type = df['type'].unique()

for k in name_type:
    options_type.append({'label':k, 'value':k})

car_type = dcc.Dropdown(id='type',
                        options=options_type,
                        value=name_type,
                        multi=True
                        )

# ТАБЛИЦЫ

tab1_content = [dbc.Row([dbc.Col(html.Div(id='v-w'), md=6), dbc.Col(html.Div(id='h-w'), md=6)], style={'margin-top': 20}),
               dbc.Row([dbc.Col(html.Div(id='p-r'), md=6), dbc.Col(html.Div(id='h-m'), md=6)])]

tab2_content = [dbc.Row(html.Div(id='table1'), style={'margin-top': 20})]

gs = 'https://www.kaggle.com/code/asmaafattah/cars-data/data'
tbl2 = dash_table.DataTable(data = description.to_dict('records'),
                                columns = [{'name':i, 'id':i} for i in description.columns],
                                style_data = {'width':'100px', 'maxWidth':'100px', 'minWidth':'100px'},
                                style_header = {'textAlign':'center'},
                                page_size = 40
                               )
tab3_content = [dbc.Row(html.A('Ссылка на данные', href=gs), style={'margin-top': 20}),
                dbc.Row(html.Div(children=tbl2), style={'margin-top': 20})]



# ДИЗАЙН ГРАФКОВ

ct = go.layout.Template(
    layout = dict(
        font=dict(family='Centry Gothic', size=14),
        legend=dict(orientation='h',
                              title_text='',
                              x = 0,
                              y = 1.1
                             ))

)

color_status_value = ['#fc03ad', '#9aa199', '#39fc03']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

#LAYOUT

app.layout = html.Div([
    dbc.Row([
    dbc.Col(html.H1('Эксперементальный дашборд'),
                 style={'margin-battom':40}),
    dbc.Col([html.P('Автор:_'), 
             html.A('Лемыцких Александр', href='https://t.me/Lemytskikh')
            ], width={'size': 4},
               style={'display' : 'flex', 'position': 'sticky', 'color': 'blue'}
            )]),
              
    # Фильтры
              
    dbc.Row([dbc.Col([html.Div('Выбери объём'),    
                    html.Div(selector)], width={'size': 2}),
            
            dbc.Col([html.Div('car_type'),
                        html.Div(car_type)], width={'size': 6, 'offset': 1}),
             
            dbc.Col(dbc.Button('Применить', id='buton1', n_clicks=0, className='mr-3'),
                    width={'size': 1, 'offset': 1})
             
            ], style={'margin-battom':40}),
    
    #Графики
    
    dbc.Tabs([
        dbc.Tab(tab1_content, label='Графики'),
        dbc.Tab(tab2_content, label='Таблица'),
        dbc.Tab(tab3_content, label='Описание данных')
    ])
              
],
style={'margin-left':'80px', 'margin-right':'80px'}   
)

#CALLBACKS

@app.callback(
    [Output(component_id='v-w', component_property='children'),
     Output(component_id='h-w', component_property='children'),
     Output(component_id='p-r', component_property='children'),
     Output(component_id='h-m', component_property='children'),
     Output(component_id='table1', component_property='children')],
    [Input(component_id='buton1', component_property='n_clicks')],
    [State(component_id='range', component_property='value'),
     State(component_id='type', component_property='value')
    ]
    )


def update_fig1(nc1,some_value, some_type):
    chart_df_fig1 = df[(df['enginesize'] > some_value[0]) & (df['enginesize'] < some_value[1]) &
                       (df['type'].isin(some_type))
                      ]
    
    if len(chart_df_fig1) == 0:
        return html.Div('Выберете больше данных'), html.Div(), html.Div(), html.Div()
    
    fig1 = px.scatter(chart_df_fig1, x='enginesize', y='weight', color='type')
    fig1.update_layout(template=ct, legend_title='Тип автомобиля:')
    fig1.update_xaxes(title_text='Объем двигателя')
    fig1.update_yaxes(title_text='Вес автомобиля')
    html1 = [html.Div('Объем двигателя-Вес автомобиля'),
             dcc.Graph(figure=fig1)]
    
    fig2 = px.scatter(chart_df_fig1, x='horsepower', y='weight', color='status', size='enginesize',
                      color_discrete_sequence=color_status_value)
    fig2.update_layout(template=ct, legend_title='Статус расход-цена:')
    fig2.update_xaxes(title_text='Мощность двигателя')
    fig2.update_yaxes(title_text='Вес автомобиля')
    html2 = [html.Div('Зависимость веса и мощноти двигателя'),
             dcc.Graph(figure=fig2)]
                                                   
    fig3 = px.histogram(chart_df_fig1, x='msrp', color='origin', barmode='overlay')
    fig3.update_layout(template=ct, legend_title='Регион:')
    fig3.update_xaxes(title_text='Цена автомобиля')
    fig3.update_yaxes(title_text='Кол-во автомоблией в ценовом диапазоне')
    html3 = [html.Div('Цена автомобиля взависимости от региона'),
             dcc.Graph(figure=fig3)]
    
    fig4= px.scatter(chart_df_fig1, x='horsepower', y='msrp', color='drivetrain',
                     color_discrete_sequence=color_status_value)
    fig4.update_layout(template=ct, legend_title='Привод автомобиля:')
    fig4.update_xaxes(title_text='Мощность двигателя')
    fig4.update_yaxes(title_text='Цена автомобиля')
    html4 = [html.Div('Зависимость цены от мощности'),
             dcc.Graph(figure=fig4)]
        
        
    # Таблица
    data_table = chart_df_fig1.drop(['mpg_mean', 'hp_labels', 'mh_labels', 'status'], axis=1)
    tbl1 = dash_table.DataTable(data = data_table.to_dict('records'),
                                columns = [{'name':i, 'id':i} for i in data_table.columns],
                                style_data = {'width':'100px', 'maxWidth':'100px', 'minWidth':'100px'},
                                style_header = {'textAlign':'center'},
                                page_size = 40
                               )
    html5 = [html.Div('Исходные данные'), tbl1]
    
    return html1, html2, html3, html4, html5


if __name__ == '__main__':
    app.run_server(debug=True)