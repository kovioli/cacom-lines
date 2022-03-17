from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
import pathlib
from dash.dependencies import Input, Output, State

from app import app
from apps import side_and_nav_bars



# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()

index_content=html.Div(id="page-1-content",className="content-style",
    children=[
   dbc.Card(
    [

        dbc.CardBody(
            [
                html.H1("Introduction", className="card-title"),
                html.P(
                    "The effects of zero gravity on the human body are still unclear."
                    " The physiological effects are being researched at the DLR (German Aerospace Center), "
                    "and to help their work, a parametrised helper tool is developed for the recognition of aponeurosis in the calf muscle.",
                    style={'textAlign': 'justify'},
                    className="card-text",
                ),
            ]
        ),
        dbc.CardImg(src=app.get_asset_url("zero_grav.png"),
                    className="image-center",
                    title='The effects of zero gravity on the human body',
                    bottom=True),

    ],
),

    ],
)
home_layout = html.Div([
    index_content,
    side_and_nav_bars.navbar,
    side_and_nav_bars.sidebar,

])

