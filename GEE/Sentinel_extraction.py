import ee
import numpy as np

# ee.Authenticate()
ee.Initialize()

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


def maskS2clouds(image):
    qa = image.select('QA60')

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11

    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0) and (
        qa.bitwiseAnd(cirrusBitMask).eq(0))

    return image.updateMask(mask).divide(10000)


for ns in north_to_south[:len(north_to_south)-1]:
    for we in west_to_east[:len(west_to_east)-1]:
        area = ee.Geometry.Rectangle(
            we,
            ns,
            we+step_size,
            ns+step_size)
        sentinel = ee.ImageCollection(
            'COPERNICUS/S2_SR').filterBounds(area).filterDate('2018-01-01', '2018-12-31').filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)).map(maskS2clouds)

        sentinel_SR_RGB = sentinel.select(['B4', 'B3', 'B2']).mosaic(
        ).visualize(bands=['B4', 'B3', 'B2'], min=0.0, max=0.3)
        sentinel_SR_NIR = sentinel.select(['B8']).mosaic(
        ).visualize(bands=['B8'], min=0.0, max=0.4)

        task = ee.batch.Export.image.toDrive(
            image=sentinel_SR_RGB,
            description='Sentinel_image_RGB_' + str(ns) + '_'+str(we),
            scale=10,
            folder='Sentinel',
            region=area,
            maxPixels=10e12,
            fileFormat='GeoTIFF'
        )
        task.start()

        # task = ee.batch.Export.image.toDrive(
        #     image=sentinel_SR_NIR,
        #     description='Sentinel_image_NIL_' + str(ns) + '_'+str(we),
        #     scale=10,
        #     folder='Sentinel',
        #     region=area,
        #     maxPixels=10e12,
        #     fileFormat='GeoTIFF'
        # )

        # task.start()
        print('task done.')
