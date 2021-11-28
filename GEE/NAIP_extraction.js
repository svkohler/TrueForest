//var area= ee.Geometry.Rectangle(-120.25, 36.45, -119.65,37.05)
var area = ee.Geometry.Rectangle(-120.25, 36.45, -119.65,37.05) //Extracted Area

// Load four 2012 NAIP quarter quads, different locations.
var naip2018 = ee.ImageCollection('USDA/NAIP/DOQQ')
  .filterBounds(area)
  .filterDate('2018-01-01', '2018-12-31');

// Spatially mosaic the images in the collection and display.
var naip_2018_mosaic = naip2018.mosaic();

// Export a cloud-optimized GeoTIFF.
Export.image.toDrive({
  image: naip_2018_mosaic,
  description: 'NAIP_image',
  folder: 'NAIP',
  scale: 1,
  region: area,
  maxPixels: 10e12,
  fileFormat: 'GeoTIFF',
  formatOptions: {
    cloudOptimized: true
  }
})