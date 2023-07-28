"""Plotly Dash HTML layout override."""

html_layout = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        <link rel="icon" type="image/x-icon" href="/static/img/favicon.ico">
        {%css%}
    </head>
    <body>
        <!--[if IE]><script>
        alert("Dash v2.7+ does not support Internet Explorer.
        Please use a newer browser.");
        </script><![endif]-->
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""