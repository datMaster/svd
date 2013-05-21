#-------------------------------------------------
#
# Project created by QtCreator 2013-05-20T13:40:54
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = svd_gnu
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += main.cpp

LIBS += -lgsl -lgslcblas -lm

OTHER_FILES += \
    out \
    temp.txt
