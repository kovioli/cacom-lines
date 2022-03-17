import dash
import dash_bootstrap_components as dbc

FA = "https://use.fontawesome.com/releases/v5.8.1/css/all.css"
app = dash.Dash(__name__,external_stylesheets=[dbc.themes.CERULEAN, FA],suppress_callback_exceptions=True)
app.title = "Clinical application Project"
server = app.server