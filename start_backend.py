"""Application entry point."""
from backend.indexapp import create_dbapp
from settings import dbapp_port, dbapp_host

dbapp = create_dbapp()

if __name__ == "__main__":
    dbapp.run(host=dbapp_host,
              port=dbapp_port,
              debug=True)
