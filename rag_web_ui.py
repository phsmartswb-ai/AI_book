"""Compatibility launcher. Prefer `python app.py`."""

from app import APP_CSS, APP_THEME, app


if __name__ == "__main__":
    app.launch(show_error=True, css=APP_CSS, theme=APP_THEME)
