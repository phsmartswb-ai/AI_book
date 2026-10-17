"""Main application entry point."""

import os

from rag_app.ui import APP_CSS, APP_THEME, build_app


app = build_app()


if __name__ == "__main__":
    configured_port = os.getenv("GRADIO_SERVER_PORT")
    app.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        # None lets Gradio try 7860, then the next available port.
        server_port=int(configured_port) if configured_port else None,
        show_error=True,
        css=APP_CSS,
        theme=APP_THEME,
    )
