# -*- coding: utf-8 -*-
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import shapefile
# -*- coding: utf-8 -*-
'''
Created on Tue Jun 23 10:30:04 2015
@author: angel.navia@uc3m.es
'''


def clean_name(municipio):
    municipio = municipio.lower()
    if municipio[0:2] == 'a ':
        municipio = municipio[2:len(municipio)]
    if municipio[0:2] == 'o ':
        municipio = municipio[2:len(municipio)]
    if municipio[0:4] == 'las ':
        municipio = municipio[4:len(municipio)]
    if municipio[0:3] == 'el ':
        municipio = municipio[3:len(municipio)]
    if municipio[0:3] == 'la ':
        municipio = municipio[3:len(municipio)]
    if municipio[0:4] == 'los ':
        municipio = municipio[4:len(municipio)]
    if municipio[0:4] == 'els ':
        municipio = municipio[4:len(municipio)]
    if municipio[0:4] == 'les ':
        municipio = municipio[4:len(municipio)]
    if municipio[0:3] == 'os ':
        municipio = municipio[3:len(municipio)]
    if municipio[0:3] == 'as ':
        municipio = municipio[3:len(municipio)]
    municipio = municipio.replace('ñ', '#')
    municipio = municipio.replace('á', 'a')
    municipio = municipio.replace('é', 'e')
    municipio = municipio.replace('í', 'i')
    municipio = municipio.replace('ó', 'o')
    municipio = municipio.replace('ú', 'u')
    municipio = municipio.replace('\xc3\xa1', 'a')
    municipio = municipio.replace('\xc3\xa9', 'e')
    municipio = municipio.replace('\xc3\xad', 'i')
    municipio = municipio.replace('\xc3\xb3', 'o')
    municipio = municipio.replace('\xc3\xba', 'u')
    municipio = municipio.replace('\xc3\xbc', 'u')
    municipio = municipio.replace('\xc3\xb1a', '#')
    municipio = municipio.replace("d'", '')
    municipio = municipio.replace('ç', 's')
    municipio = municipio.replace('Ç', 's')
    municipio = municipio.replace("l'", '')
    municipio = municipio.replace(', a', '')
    municipio = municipio.replace(', las', '')
    municipio = municipio.replace(', los', '')
    municipio = municipio.replace(', les', '')
    municipio = municipio.replace(', els', '')
    municipio = municipio.replace(', la', '')
    municipio = municipio.replace(', el', '')
    municipio = municipio.replace(', as', '')
    municipio = municipio.replace(', os', '')
    municipio = municipio.replace(', o', '')
    #municipio = municipio.replace('la ', '')
    #municipio = municipio.replace('las ', '')
    #municipio = municipio.replace('los ', '')
    municipio = municipio.replace(' de la ', '')
    municipio = municipio.replace(' la ', '')
    municipio = municipio.replace(' del ', '')
    municipio = municipio.replace(' de ', '')
    municipio = municipio.replace(' i la ', '')
    municipio = municipio.replace(' i ', '')
    #municipio = municipio.replace('a ', '')
    municipio = municipio.replace("'", '')
    municipio = municipio.replace('-', '')
    municipio = municipio.replace('/', '')
    municipio = municipio.replace(',', '')
    municipio = municipio.replace(' ', '')
    municipio = municipio.replace('\xc3\x91', '#')
    municipio = municipio.replace('\xd1', '#')
    municipio = municipio.replace('\xc3\xa8', 'e')
    municipio = municipio.replace('\xc3\xa0', 'a')
    municipio = municipio.replace('\xc3\x81', 'a')
    municipio = municipio.replace('\n', '')
    municipio = municipio.replace('\xef\xbb\xbf', '')
    municipio = municipio.replace("l'", '')
    municipio = municipio.replace('\xc3\x87', 's')
    municipio = municipio.replace('\xc3\x89', 'e')
    municipio = municipio.replace('\xc3\x9a', 'u')
    municipio = municipio.replace('\xc3\xb2', 'o')
    municipio = municipio.replace('\xc7', 's')
    municipio = municipio.replace('adejecasco', 'adeje')
    return municipio

'''
for key in gps_coords_orignames.keys():
    if 'vinar' in key:
        key



'''