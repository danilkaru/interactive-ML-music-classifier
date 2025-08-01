# This file contains the JavaScript code which is integrated within the user interface
# This code is responsible for assigning a specific style to the buttons in the UI, 
# as well as handling most of the audio playback on hover. 

js_control = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            button {
                background-color: #4CAF50;
                color: white;
                padding: 5px 10px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            button:hover {
                background-color: #45a049;
            }
            // Reset button design 
            #reset-button {
                background-color: #f44336 !important;
                color: white !important;
            }
            #reset-button:hover {
                background-color: #d32f2f !important;
            }
            // Download button design
            #download-button {
                background-color: #4169E1 !important;
                color: white !important;
            }
            #download-button:hover {
                background-color: #365EC7 !important;
            }
            
            .debug-container {
                background-color: #f8f9fa;
                border: 1px solid #ddd;
                padding: 10px;
                margin-top: 10px;
                font-family: monospace;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
            <script>
                //JS functions for handling audio playback in the UI
                let currentAudio = null;

                function playHoverAudio(audioPath) {
                    stopAudio();  // stop previous audio if any is playing
                    if (audioPath) {
                        currentAudio = new Audio(audioPath);
                        currentAudio.play().catch(err => {
                            console.error("Failed to play:", err);
                        });
                    }
                }

                function playAudio(index, webPath) {
                    console.log('Button clicked - Index:', index, 'Path:', webPath);
                    
                    if (webPath === "Not found") {
                        alert("Audio file not found");
                        return;
                    }
                    
                    stopAudio();  
                    
                    currentAudio = new Audio(webPath);
                    currentAudio.play().catch(err => {
                        console.error("Failed to play button audio:", err);
                        alert("Could not play audio file");
                    });
                }

                function stopAudio() {
                    if (currentAudio) {
                        currentAudio.pause();
                        currentAudio.currentTime = 0;
                        currentAudio = null;
                    }
                }

                // Listen for custom Dash events
                window.addEventListener('hover-audio', function(e) {
                    playHoverAudio(e.detail.audioPath);
                });

                window.addEventListener('unhover-audio', function() {
                    stopAudio();
                });                
            </script>
        </footer>
    </body>
</html>
'''