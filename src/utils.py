import datetime

def get_current_timestamp():
    return str(datetime.datetime.now()).replace(" ", "_").replace(".", ":")
