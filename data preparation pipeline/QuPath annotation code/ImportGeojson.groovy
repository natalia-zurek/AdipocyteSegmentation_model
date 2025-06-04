// import geojson file with the same name, useful when there are a lot of files to import

def imageData = getCurrentImageData()
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
clearAnnotations()
def geojson_path = 'path/to/geojson'
importObjectsFromFile("${geojson_path}/${name}.geojson")
