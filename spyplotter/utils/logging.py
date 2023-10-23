import logging

# create console handler and set level that is printed on console
spyplot_console_log_handler = logging.StreamHandler()
#show up to debug messages
spyplot_console_log_handler.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    
# add formatter to ch
spyplot_console_log_handler.setFormatter(formatter)

def update_logging_level(level):
    spyplot_console_log_handler.setLevel(level)

def setup_log(name='spyplot'):

    # set up logger with name 
    log = logging.getLogger(name)

    # allow debug messages
    log.setLevel(logging.DEBUG)

    # donot duplicate messages
    log.propagate = False  

    # add ch to logger
    log.addHandler(spyplot_console_log_handler)

    return log