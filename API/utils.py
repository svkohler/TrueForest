from doctest import register_optionflag
from unittest.mock import _SentinelObject
import ee
import geemap
import os
import numpy as np
from PIL import Image
from skimage import io


def get_sat_image(coordinates, PATH):
    ee.Initialize()

    bottom_left = ee.Geometry.Point(
        [coordinates['bottom_left_long'], coordinates['bottom_left_lat']])
    top_right = ee.Geometry.Point(
        [coordinates['top_right_long'], coordinates['top_right_lat']])

    area = ee.Geometry.Rectangle([bottom_left, top_right])

    sentinel = ee.ImageCollection(
        'COPERNICUS/S2_SR').filterBounds(area).filterDate('2018-01-01', '2018-12-31').filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 1)).map(maskS2clouds)

    sentinel_SR_RGB = sentinel.select(['B4', 'B3', 'B2']).mosaic(
    ).visualize(bands=['B4', 'B3', 'B2'], min=0.0, max=0.3)

    geemap.ee_export_image(
        sentinel_SR_RGB, filename=PATH+'satellite.tif', scale=10, region=area, file_per_band=False)

    sentinel_rgb = Image.fromarray(
        np.uint8(io.imread(PATH+'satellite.tif')))

    sentinel_rgb = sentinel_rgb.resize((224, 224))

    sentinel_rgb.save(PATH+'satellite.png', format="png")

    os.remove(PATH+'satellite.tif')


def maskS2clouds(image):
    qa = image.select('QA60')

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11

    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0) and (
        qa.bitwiseAnd(cirrusBitMask).eq(0))

    return image.updateMask(mask).divide(10000)
