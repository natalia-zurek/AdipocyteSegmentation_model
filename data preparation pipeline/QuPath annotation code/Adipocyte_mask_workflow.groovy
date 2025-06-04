//select all annotations and fill holes + delete small fragments
selectAnnotations();
runPlugin('qupath.lib.plugins.objects.RefineAnnotationsPlugin', '{"minFragmentSizePixels":100.0,"maxHoleSizePixels":-1.0}')
runPlugin('qupath.lib.plugins.objects.FillAnnotationHolesPlugin', '{}')
// Assign one class to all annotations
def fat = getPathClass('Adipocyte')
getAnnotationObjects().eachWithIndex { annotation , i ->
      annotation.setPathClass(fat)

}
fireHierarchyUpdate()

def imageData = getCurrentImageData()

// Define output path (relative to project)
def outputDir = buildFilePath(PROJECT_BASE_DIR, 'adipocyte masks')
mkdirs(outputDir)
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
def path = buildFilePath(outputDir, name + ".tif")

//if (fileExists(path)) { print("File exists. Skipping.") }
//else {
// Define how much to downsample during export (may be required for large images)
double downsample = 1

// Create an ImageServer where the pixels are derived from annotations
def labelServer = new LabeledImageServer.Builder(imageData)
  .backgroundLabel(0, ColorTools.BLACK) // Specify background label (usually 0 or 255)
  .downsample(downsample)    // Choose server resolution; this should match the resolution at which tiles are exported
  .addLabel('Adipocyte', 1)      // Choose output labels (the order matters!)
  .multichannelOutput(false) // If true, each label refers to the channel of a multichannel binary image (required for multiclass probability)
  .build()

// Write the image
writeImage(labelServer, path)
print("Done")
//}