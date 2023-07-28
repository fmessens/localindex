"""Routes for parent Flask app."""
from flask import current_app as app
from flask import render_template


@app.route("/")
def home():
    """Landing page."""
    return render_template(
        "index.jinja2",
        title="Dashboard local PDF indexing",
        description="Index local pdf files and search them.",
        template="home-template",
        body="This is a homepage served with Flask.",
    )
