import ee
import geemap
import sys
import numpy as np

# ee.Authenticate()
ee.Initialize()

'''
File to extract NAIP imagery from Google Earth Engine

Please enter the desired coordinates below
'''

# 30.93605976603369  # 35.236  #  32.60215788172933 # 30.025215750666348
START_NORTH = 32.60215788172933
# 30.98934617733734  # 35.377  # 32.738568513477766  # 30.139294239374312
END_SOUTH = 32.738568513477766
# -89.2068111032426  # -86.06  #  -112.00920345515958 # -84.54255356391563
START_WEST = -112.00920345515958
# -89.14484134616252  # -85.89  # -111.84681179256192  # -84.41724075873985
END_EAST = -111.84681179256192
NR_STEPS = 2

step_size = abs((START_NORTH-END_SOUTH)/NR_STEPS)

north_to_south = np.linspace(START_NORTH, END_SOUTH, NR_STEPS+1)
west_to_east = np.linspace(START_WEST, END_EAST, NR_STEPS+1)

for ns in north_to_south[:len(north_to_south)-1]:
    for we in west_to_east[:len(west_to_east)-1]:
        area = ee.Geometry.Rectangle(
            we,
            ns,
            we+step_size,
            ns+step_size)
        naip = ee.ImageCollection(
            'USDA/NAIP/DOQQ').filterBounds(area).filterDate('2017-01-01', '2018-12-31')

        naip_mosaic = naip.mosaic().visualize(
            bands=['R', 'G', 'B'])

        task = ee.batch.Export.image.toDrive(
            image=naip_mosaic,
            description='NAIP_image_' + str(ns) + '_'+str(we),
            scale=1,
            folder='NAIP',
            region=area,
            maxPixels=10e12,
            fileFormat='GeoTIFF'
        )

        task.start()
        print('task done.')
