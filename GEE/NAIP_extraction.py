import ee
import geemap
import sys
import numpy as np

# ee.Authenticate()
ee.Initialize()

START_NORTH = 30.025215750666348  # 30.93605976603369  # 35.236  #   #
END_SOUTH = 30.139294239374312  # 30.98934617733734  # 35.377  #   #
START_WEST = -84.54255356391563  # -89.2068111032426  # -86.06  #   #
END_EAST = -84.41724075873985  # -89.14484134616252  # -85.89  #   #
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
