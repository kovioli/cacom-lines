import dash_bootstrap_components as dbc
from app import app
from apps import side_and_nav_bars
from dash.dependencies import Input, Output, State
from dash import dcc, html
import cv2
import os
from multi_detection.predict import predict


#path = r'/gui/multi_detection\data\test_set\images\L9GE0MHG.png'
#fig = cv2.imread(path, 1)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
def write_image(path, img):
    # img = img*(2**16-1)
    # img = img.astype(np.uint16)
    # img = img.astype(np.uint8)
    img = cv2.convertScaleAbs(img, alpha=(255.0))
    cv2.imwrite(path, img)

#fig_heatmap=go.Figure(data=go.Heatmap(),layout = go.Layout(height=700))
historical_content = html.Div(id="page-2-content", className="content-style",
children=[
dbc.Card(
    [
     dbc.CardBody(
        [
        html.Div([
            html.H1('Select your images', id="Overview"),
                html.P([" "]),
            ] ),



        html.Div([
            dcc.Upload(
                id='upload-image',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=True
            ),
            html.Div(id='output-image-upload'),
        ]),


        ]),

    ])
])
layout = html.Div([historical_content, side_and_nav_bars.navbar, side_and_nav_bars.sidebar])



def parse_contents(contents, filename, date):
    path = "/Users/zaine/PycharmProjects/cacom-lines/multi_detection/data/test_set/images/"
    new_path = os.path.join(path, filename)
    fig = cv2.imread(new_path, 1)
    current_directory = os.getcwd()
    goal_dir = os.path.join(os.getcwd(), "assets")
    drawing, info = predict(fig)


    os.chdir(goal_dir)
    filename1 = 'savedImage1.png'
    cv2.imshow("prediction", drawing)
    write_image(filename1, drawing)
    #cv2.imwrite(filename1, drawing)

    return html.Div([
        #html.H5(filename),
        #html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents),
        html.Hr(),
        #html.Div('Raw Content'),
        #html.Pre(contents[0:200] + '...', style={
        #    'whiteSpace': 'pre-wrap',
        #    'wordBreak': 'break-all'
        #}),
        html.Div([
            html.H1('display aponeurosis ', id="Overview"),
            html.P([" "]),

        ]),



        dbc.CardImg(src=app.get_asset_url("savedImage1.png"),
                    className="image-center",
                    title='aponeurosis',
                    bottom=True),
    ])

@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):

    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children
