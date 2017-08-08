# -*- coding: utf-8 -*-
from . import BASE_DIR

# TODO: if you want to choose another directory, change LOG_DIR path.
LOG_DIR = BASE_DIR + '/log/'


APP_KEY = 'app'
INBOUND_KEY = 'inbound'
OUTBOUND_KEY = 'outbound'

LOGGING = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'verbose': {
            'format': '%(asctime)s [%(levelname)s] [%(module)s] %(process)d %(thread)d %(message)s'
        },
        'normal' : {
            'format': '%(asctime)s [%(levelname)s] [%(module)s] %(message)s'
        },
        'simple': {
            'format': '[%(levelname)s] %(message)s'
        },
    },
    'handlers': {
        'app': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'normal',
            'filename': LOG_DIR + 'application.log'
        },
        'inbound': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'normal',
            'filename': LOG_DIR + 'inbound.log'
        },
        'outbound': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'normal',
            'filename': LOG_DIR + 'outbound.log'
        },
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'simple'
        }
    },
    'loggers': {
        APP_KEY: {
            # 'handlers': ['app'],
            'handlers': ['app', 'console'],
            'level': 'INFO',
            'propagate': False,
        },
        INBOUND_KEY: {
            'handlers': ['inbound'],
            'level': 'DEBUG',
            'backupCount': 4,
            'formatter': 'verbose',
        },
        OUTBOUND_KEY: {
            'handlers': ['outbound'],
            'level': 'DEBUG',
            'backupCount': 4,
            'formatter': 'verbose',
        }
    }
}

CHERRYPY_ACCESS_LOG = LOG_DIR + 'access.log'
CHERRYPY_ERROR_LOG = LOG_DIR + 'error.log'
