# ----------------------------------- IMPORTS -----------------------------------
# Dash imports
import dash
from dash.dependencies import Input, Output

# Local imports
from js_control import js_control #import custom JavaScript code
from app_layout import layout #import Dash app layout

from app_callbacks import get_callbacks #import Dash callback functions


# ----------------------------------- Initialise Dash App -----------------------------------
app = dash.Dash(__name__) 
app.title = "IML Music Classifier"
app.layout = layout # Import application layout (all Dash/HTML components etc.) from app_layout.py
app.index_string = js_control # Add custom JavaScript code for audio playback

get_callbacks(app) 


app.clientside_callback(
    """
    function(data) {
        if (data && data.audioPath) {
            const event = new CustomEvent('hover-audio', { detail: { audioPath: data.audioPath } });
            window.dispatchEvent(event);
        } else {
            const event = new CustomEvent('unhover-audio');
            window.dispatchEvent(event);
        }
        return null;
    }
    """,
    Output('dummy-output', 'children'),
    Input('hover-audio-path', 'data')
)

# Run app
if __name__ == '__main__':
    app.run(debug=True)
