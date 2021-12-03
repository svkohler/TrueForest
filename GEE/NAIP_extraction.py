import ee
import sys
import numpy as np

# ee.Authenticate()
ee.Initialize()

start_north = 36.45
end_south = 37.05
start_west = -120.25
end_east = -119.65
nr_steps = 4
step_size = (start_north-end_south)/nr_steps

north_to_south = np.linspace(36.45, 37.05, nr_steps)
west_to_east = np.linspace(-120.25, -119.65, nr_steps)

for ns in north_to_south[2:]:
    for we in west_to_east:
        area = ee.Geometry.Rectangle(
            we,
            ns,
            we+step_size,
            ns+step_size)
        naip = ee.ImageCollection(
            'USDA/NAIP/DOQQ').filterBounds(area).filterDate('2018-01-01', '2018-12-31')

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
