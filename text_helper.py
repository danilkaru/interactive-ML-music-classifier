# Helper file containing the function which formats the text output
# for displaying the track status statistics in the UI

from dash import html
from collections import defaultdict
import dash_bootstrap_components as dbc

def format_track_stats_component(data_store):
    """
    Function which formats the text output for displaying the track status statistics in the UI
    TO DO: see if function can be further refactored/optimized
    """
    # Initialise empty dictionaries for storing track statuses 
    status_by_class = defaultdict(lambda: defaultdict(int))
    total_by_class = defaultdict(int)
    
    # Count all track statuses per class, for all user annotated data points 
    for idx, label in data_store["assigned_classes"].items():
        # Add 1 to total amount for encountered class label
        total_by_class[label] += 1
        
        # If track is highlighted as problematic in data_store
        if idx in data_store["is_problematic_tracks"]:
            # Retrieve track status as String
            track_status = str(data_store["is_problematic_tracks"][idx])
    
            # Add 1 to track status count for encountered class label
            # NOTE: the problematic track statuses should not be displayed for annotated data points!
            # Look into this when refactoring function
            if "‼️↔️" in track_status: 
                status_by_class[label]["‼️↔️"] += 1
            elif "↔️‼️" in track_status:
                status_by_class[label]["↔️‼️"] += 1
            elif "‼️" in track_status:
                status_by_class[label]["‼️"] += 1
            elif "❔" in track_status: #NOTE: not part of current implementation of UI
                status_by_class[label]["❔"] += 1
            elif "↔️" in track_status:
                status_by_class[label]["↔️"] += 1
            else: 
                status_by_class[label]["👤"] += 1 #Default symbol for user annotated data points
        else: 
            status_by_class[label]["👤"] += 1 #Default symbol for user annotated data points
    
    # Count all track statuses per class, for all unannotated data points 
    for idx, label in data_store["predicted_classes"].items():
        # Add 1 to total amount for encountered class label
        if idx not in data_store["assigned_classes"]:  # Ensure counts for predicted/annotated are mutually exclusive 
            total_by_class[label] += 1
            
            # If track is highlighted as problematic in data_store
            if idx in data_store["is_problematic_tracks"]:
                track_status = str(data_store["is_problematic_tracks"][idx])
                
                # Add 1 to track status count for encountered class label
                if "‼️↔️" in track_status:
                    status_by_class[label]["‼️↔️"] += 1
                elif "↔️‼️" in track_status:
                    status_by_class[label]["↔️‼️"] += 1
                elif "‼️" in track_status:
                    status_by_class[label]["‼️"] += 1
                elif "❔" in track_status:
                    status_by_class[label]["❔"] += 1
                elif "↔️" in track_status:
                    status_by_class[label]["↔️"] += 1
                else: 
                    status_by_class[label]["✅"] += 1 # If not problematic, then track gets ✅ by default
            else:                
                status_by_class[label]["✅"] += 1 # If not problematic, then track gets ✅ by default
    
    # Error handling
    if not total_by_class:
        return "No classified tracks found. Please assign classes or train the classifier."
    

    # Format track status statistics to readable text 
    stats_content = []
    stats_content.append(html.H4("Distribution of Track Statuses per Class"))
    
    table_header = [
        html.Thead(html.Tr([
            html.Th("Class"), 
            html.Th("Total Tracks"),
            html.Th("Status Distribution")
        ]))
    ]
    
    table_rows = []
    for class_name, total in sorted(total_by_class.items()):
        status_distribution = []
        for status, count in status_by_class[class_name].items():
            percentage = (count / total) * 100 if total > 0 else 0
            status_distribution.append(f"{status}: {count} ({percentage:.2f}%)")
        
        row = html.Tr([
            html.Td(class_name),
            html.Td(total),
            html.Td(", ".join(status_distribution))
        ])
        table_rows.append(row)
    
    table_body = [html.Tbody(table_rows)]
    stats_table = dbc.Table(table_header + table_body, bordered=True, hover=True, striped=True)
    
    # Add a legend for describing each of the status symbols 
    legend = html.Div([
        html.H5("Status Symbols:", style={"marginTop": "5px"}),
        html.Ul([
            html.Li("👤 - User annotation"),
            html.Li("✅ - Not marked as problematic"),
            html.Li("‼️ - Marked as problematic by class prediction uncertainty (CPU) heuristic"),
            # html.Li("❔ - Discrepancy between user annotation and MLP prediction"), #not part of current UI implementation
            html.Li("↔️ - Marked as problematic by dataset coverage (DC) heuristic"),            
        ])
    ])
    
    stats_content.append(stats_table)
    stats_content.append(legend)
    
    return stats_content