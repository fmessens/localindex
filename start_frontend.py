"""Application entry point."""
from dashapp import init_app
from settings import dashapp_port, dashapp_host

dashapp = init_app()

if __name__ == "__main__":
    dashapp.run(host=dashapp_host, 
                port=dashapp_port,
                debug=True)
