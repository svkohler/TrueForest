import ee
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


def maskS2clouds(image):
    qa = image.select('QA60')

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11

    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0) and (
        qa.bitwiseAnd(cirrusBitMask).eq(0))

    return image.updateMask(mask).divide(10000)


for ns in north_to_south:
    for we in west_to_east:
        area = ee.Geometry.Rectangle(
            we,
            ns,
            we+step_size,
            ns+step_size)
        sentinel = ee.ImageCollection(
            'COPERNICUS/S2_SR').filterBounds(area).filterDate('2018-01-01', '2018-12-31').filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 1)).map(maskS2clouds)

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

        task = ee.batch.Export.image.toDrive(
            image=sentinel_SR_NIR,
            description='Sentinel_image_NIL_' + str(ns) + '_'+str(we),
            scale=10,
            folder='Sentinel',
            region=area,
            maxPixels=10e12,
            fileFormat='GeoTIFF'
        )

        task.start()
        print('task done.')
