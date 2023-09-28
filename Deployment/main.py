# Importación de librerías
import pandas as pd
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import humanize

# Carga de datos histórico
df=pd.read_csv("DataStorage/datasets_company_etl.csv")
# Diccionario de empresas
dic_com={"Apple":"AAPL", "IBM":"IBM", "Google":"GOOG", "Meta":"META", "Amazon":"AMZN", "Tesla": "TSLA", "Microsoft":"MSFT"}

# Función para la extracción de datos
def data_extraction(df, start_date, end_date):
  df_list=[df] 
  for i in dic_com.items():
    url=f"https://query1.finance.yahoo.com/v7/finance/download/{i[1]}?period1={start_date}&period2={end_date}&interval=1d&events=history&includeAdjustedClose=true"
    df=pd.read_csv(url)
    df["id_company"]=i[1]
    df_list.append(df)
  df=pd.concat(df_list).sort_values(by="Date")
  return df.reset_index(drop=True)

start_date=int(pd.to_datetime("2023-09-19").timestamp())# Fecha de inicio de los datos faltantes
end_date=int((datetime.now()+timedelta(days=1)).timestamp())# Se considera un día más de la actualidad para prevenir errores
df2=data_extraction(df, start_date, end_date)
df2["Date"]=pd.to_datetime(df2.Date)

# Función para ajustar la periodicidad
def periodicity_adjustment(dataframe):
  df=dataframe.copy()
  df_list=[]
  for c in df.id_company.unique():
    df2=df[df.id_company==c]
    df2=df2.set_index("Date").asfreq(freq="D", method="bfill").reset_index()
    df_list.append(df2)
  df_final=pd.concat(df_list).sort_values(by="Date").reset_index(drop=True)
  return df_final

df3=periodicity_adjustment(df2)

# Función para hallar el escaldor original y el último bloque de 90 días escalado
def scaler_block(df_original, df_current, company):

  def variable_selection(df, company):
    df2=df.copy()
    df2=df2[df2.id_company==company]
    df2=df2.iloc[:, 1:5].reset_index(drop=True)
    return df2

  df=variable_selection(df_original, company)
  df2=variable_selection(df_current, company)
  
  def data_scaling(set_data):
    set_data2=set_data.copy()
    cols=set_data2.columns
    list_mms=[MinMaxScaler() for c in cols]
    for i, c in enumerate(cols):
      set_data2[c]=list_mms[i].fit_transform(set_data2[[c]])
    return set_data2, list_mms
  
  _, list_mms = data_scaling(df)
  #Hallando los últimos 90 días
  df2=df2.iloc[-90:]
  # Escalando
  for i, c in enumerate(df2.columns):
    df2[c]=list_mms[i].transform(df2[[c]])
  
  set_pred=df2.values.reshape(1, df2.shape[0], df2.shape[1])

  return set_pred, list_mms[1]

# Título
st.markdown("<h1 style='text-align: center;'>Global Stock Predictions</h1>", unsafe_allow_html=True)

# Multiselector de compañías
selected_company=st.sidebar.selectbox("Companies", dic_com.keys(), index=0)
#Botones de métricas
metric=st.sidebar.radio("Metrics", ["Max", "Mean","Min"], index=0)
# Selector de fechas
start_date=pd.to_datetime(st.sidebar.date_input("Start Date", df3.Date.min()))
end_date=pd.to_datetime(st.sidebar.date_input("End Date", df3.Date.max()))
#Multiselector de variables
selected_variable=st.sidebar.multiselect("Variables", df3.columns[1:5], ["High"])

# Métricas
st.markdown(f"#### **{selected_company} Metrics**")
df_metric=df3[(df3.id_company==dic_com[selected_company])&(df3.Date>=start_date)&(df3.Date<=end_date)]
if metric=="Max":
  value=df_metric.max()
elif metric=="Mean":
  value=df_metric.mean()
else:
  value=df_metric.min()

st.markdown("""
<div style="display: flex; justify-content: space-between; background-color: #F0F8FF; padding: 7px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);">
  <div style="background-color: #B0E0E6; width: 130px; padding: 12px; border-radius: 5px; box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);">
    <h2 style="text-align: center; font-size: 20px;">High</h2>
    <p style="font-size: 16px; text-align: center;">{:.2f}</p>
  </div>
  <div style="background-color: #B0E0E6; width: 130px; padding: 12px; border-radius: 5px; box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);">
    <h2 style="text-align: center; font-size: 20px;">Low</h2>
    <p style="font-size: 16px; text-align: center;">{:.2f}</p>
  </div>
  <div style="background-color: #B0E0E6; width: 130px; padding: 12px; border-radius: 5px; box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);">
    <h2 style="text-align: center; font-size: 20px;">Open</h2>
    <p style="font-size: 16px; text-align: center;">{:.2f}</p>
  </div>
  <div style="background-color: #B0E0E6; width: 130px; padding: 12px; border-radius: 5px; box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);">
    <h2 style="text-align: center; font-size: 20px;">Close</h2>
    <p style="font-size: 16px; text-align: center;">{:.2f}</p>
  </div>  
  <div style="background-color: #B0E0E6; width: 130px; padding: 12px; border-radius: 5px; box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);">
    <h2 style="text-align: center; font-size: 20px;">Volume</h2>
    <p style="font-size: 16px; text-align: center;">{}</p>
  </div> 
</div>
""".format(value.High, value.Low, value.Open, value.Close, humanize.intword(value.Volume)), unsafe_allow_html=True)

# Función de pérdida
def RMSE(y_true, y_pred):
  rmse=tf.math.sqrt(tf.math.reduce_mean(tf.square(y_true-y_pred)))
  return rmse
# Función para las predicciones
def predictions(set_pred, model, mms):
  y_pred=model.predict(set_pred, verbose=0)
  y_pred=mms.inverse_transform(y_pred)
  return y_pred.squeeze()

# Predicción de las comapañías seleccionadas

set_pred, mms = scaler_block(df, df3, dic_com[selected_company])
model=load_model(f"DataStorage/model_{selected_company.lower()}.keras", custom_objects={"RMSE":RMSE})
y_pred=predictions(set_pred, model, mms)

# Gráfica de Predicciones
date=[(datetime.now()+timedelta(days=d)).strftime("%Y-%m-%d") for d in range(1, 8)]

fig=px.line(x=date, y=y_pred, markers=True)
fig.update_traces(mode="lines+markers", 
                  line=dict(color="blue", width=3),
                  marker=dict(symbol="square", color="red", size=4))
fig.update_xaxes(title_text="<b>Date</b>", tickfont=dict(size=12), showgrid=True, 
                 gridwidth=0.1, gridcolor="rgba(200, 200, 200, 0.3)", 
                 griddash="dot", showline=False)
fig.update_yaxes(title_text="<b>High Price (USD)</b>", tickfont=dict(size=12), showgrid=True, 
                 gridwidth=0.1, gridcolor="rgba(200, 200, 200, 0.3)", 
                 griddash="dot", showline=False)
fig.update_layout(title=f"<b>{selected_company}'s Maximum Actions for the Next 7 Days</b>", 
                  title_font=dict(size=24),
                  plot_bgcolor="#F0FFFF")

st.plotly_chart(fig)

# Gráfico de velas
df_cand=df3[(df3.id_company==dic_com[selected_company])&(df3.Date>=start_date)&(df3.Date<=end_date)]

fig=go.Figure(data=(go.Candlestick(x=df_cand.Date, 
                                   open=df_cand.Open, 
                                   high=df_cand.High, 
                                   low=df_cand.Low, 
                                   close=df_cand.Close)))

fig.update_traces(increasing_fillcolor="green", decreasing_fillcolor="red", line=dict(width=1))
fig.update_xaxes(title_text="<b>Date</b>", tickfont=dict(size=12), showgrid=True, 
                 gridwidth=0.1, gridcolor="rgba(200, 200, 200, 0.3)", 
                 griddash="dot", showline=False)
fig.update_yaxes(title_text="<b>Price (USD)</b>", tickfont=dict(size=12), showgrid=True, 
                 gridwidth=0.1, gridcolor="rgba(200, 200, 200, 0.3)", 
                 griddash="dot", showline=False)
fig.update_layout(title=f"<b>{selected_company} Stock Candlestick Chart</b>",
                  title_font=dict(size=24),
                  xaxis_rangeslider_visible=False,
                  plot_bgcolor="#F0FFFF")

st.plotly_chart(fig)

# Gráfico de columnas
dic_inverse={v:c for c, v in dic_com.items()}
df_bar=df3.copy()
df_bar["Company"]=df_bar.id_company.apply(lambda x: dic_inverse[x])
df_bar=df_bar.groupby("Company")["Volume"].sum().reset_index().sort_values(by="Volume", ascending=False)

fig=px.bar(df_bar, x="Company", y="Volume", color="Volume", color_continuous_scale=px.colors.sequential.RdBu)
fig.update_xaxes(title_text="<b>Company</b>")
fig.update_yaxes(title_text="<b>Volume</b>", tickfont=dict(size=12), showgrid=True, 
                 gridwidth=0.1, gridcolor="rgba(200, 200, 200, 0.3)", 
                 griddash="dot", showline=False)
fig.update_layout(title=f"<b>Total Volume of Company's Traded Shares</b>",
                  title_font=dict(size=24),
                  plot_bgcolor="#F0FFFF")

st.plotly_chart(fig)

# Gráfica de variables
df_line=df3[(df3.id_company==dic_com[selected_company])&(df3.Date>=start_date)&(df3.Date<=end_date)]

fig=go.Figure()

if len(selected_variable)>0:
  for c in selected_variable:
    fig.add_trace(go.Scatter(x=df_line.Date, y=df_line[c], mode="lines", name=c))

  fig.update_xaxes(title_text="<b>Date</b>", tickfont=dict(size=12), showgrid=True, 
                     gridwidth=0.1, gridcolor="rgba(200, 200, 200, 0.3)", 
                     griddash="dot", showline=False)
  fig.update_yaxes(title_text="<b>Price (USD)</b>", tickfont=dict(size=12), showgrid=True, 
                     gridwidth=0.1, gridcolor="rgba(200, 200, 200, 0.3)", 
                     griddash="dot", showline=False)
  if len(selected_variable)==1:
    fig.update_layout(title=f"<b>{c} {selected_company} Stock Price Over Time</b>",
                      title_font=dict(size=24), 
                      plot_bgcolor="#F0FFFF")
  else:
    fig.update_layout(title=f"<b>{selected_company} Stock Price Over Time</b>", 
                      title_font=dict(size=24),
                      legend_title_text="<b>Variables</b>", 
                      plot_bgcolor="#F0FFFF")

st.plotly_chart(fig)