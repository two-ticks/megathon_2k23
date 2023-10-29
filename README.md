# megathon_2k23

# Figma File
[Presentation](https://www.figma.com/file/De9cz2hzi7hEg3G7IDDz7A/RaitaluZone?type=design&node-id=25%3A2808&mode=design&t=STb02jCBdTrbqWQC-1)

# Google Earth Engine - Identification of Paddy Cultivation 

```js
var dist = table

var sentinel1 = ee.ImageCollection('COPERNICUS/S1_GRD')
                    .filterDate('2021-06-01', '2021-06-15')
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                    .filter(ee.Filter.eq('instrumentMode', 'IW'))
                    .filter(ee.Filter.or((ee.Filter.eq('orbitProperties_pass', 'ASCENDING'), ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))))
                    .filterBounds(dist);
var sentinel2 = ee.ImageCollection('COPERNICUS/S1_GRD')
                    .filterDate('2021-06-16', '2021-06-30')
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                    .filter(ee.Filter.eq('instrumentMode', 'IW'))
                    .filter(ee.Filter.or((ee.Filter.eq('orbitProperties_pass', 'ASCENDING'), ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))))
                     .filterBounds(dist);
var sentinel3 = ee.ImageCollection('COPERNICUS/S1_GRD')
                    .filterDate('2021-07-01', '2021-07-15')
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                    .filter(ee.Filter.eq('instrumentMode', 'IW'))
                    .filter(ee.Filter.or((ee.Filter.eq('orbitProperties_pass', 'ASCENDING'), ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))))
                     .filterBounds(dist);
var sentinel4 = ee.ImageCollection('COPERNICUS/S1_GRD')
                    .filterDate('2021-07-16', '2021-07-31')
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                    .filter(ee.Filter.eq('instrumentMode', 'IW'))
                    .filter(ee.Filter.or((ee.Filter.eq('orbitProperties_pass', 'ASCENDING'), ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))))
                     .filterBounds(dist);
                    
var sentinel5 = ee.ImageCollection('COPERNICUS/S1_GRD')
                    .filterDate('2021-08-01', '2021-08-15')
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                    .filter(ee.Filter.eq('instrumentMode', 'IW'))
                    .filter(ee.Filter.or((ee.Filter.eq('orbitProperties_pass', 'ASCENDING'), ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))))
                     .filterBounds(dist);
var sentinel6 = ee.ImageCollection('COPERNICUS/S1_GRD')
                    .filterDate('2021-08-16', '2021-08-31')
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                    .filter(ee.Filter.eq('instrumentMode', 'IW'))
                    .filter(ee.Filter.or((ee.Filter.eq('orbitProperties_pass', 'ASCENDING'), ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))))
                     .filterBounds(dist);   
var sentinel7 = ee.ImageCollection('COPERNICUS/S1_GRD')
                    .filterDate('2021-09-01', '2021-09-15')
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                    .filter(ee.Filter.eq('instrumentMode', 'IW'))
                    .filter(ee.Filter.or((ee.Filter.eq('orbitProperties_pass', 'ASCENDING'), ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))))
                     .filterBounds(dist); 
var sentinel8 = ee.ImageCollection('COPERNICUS/S1_GRD')
                    .filterDate('2021-09-16', '2021-09-30')
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                    .filter(ee.Filter.eq('instrumentMode', 'IW'))
                    .filter(ee.Filter.or((ee.Filter.eq('orbitProperties_pass', 'ASCENDING'), ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))))
                    .filterBounds(dist);  
var sentinel9 = ee.ImageCollection('COPERNICUS/S1_GRD')
                    .filterDate('2021-10-01', '2021-10-15')
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                    .filter(ee.Filter.eq('instrumentMode', 'IW'))
                    .filter(ee.Filter.or((ee.Filter.eq('orbitProperties_pass', 'ASCENDING'), ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))))
                    .filterBounds(dist);
var sentinel10 = ee.ImageCollection('COPERNICUS/S1_GRD')
                    .filterDate('2021-10-16', '2021-10-31')
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                    .filter(ee.Filter.eq('instrumentMode', 'IW'))
                    .filter(ee.Filter.or((ee.Filter.eq('orbitProperties_pass', 'ASCENDING'), ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))))
                    .filterBounds(dist);
                    
var image1 = sentinel1.select('VH').mean().rename('VH1');
var image2 = sentinel2.select('VH').mean().rename('VH2');
var image3 = sentinel3.select('VH').mean().rename('VH3');
var image4 = sentinel4.select('VH').mean().rename('VH4');
var image5 = sentinel5.select('VH').mean().rename('VH5');
var image6 = sentinel6.select('VH').mean().rename('VH6');
var image7 = sentinel7.select('VH').mean().rename('VH7');
var image8 = sentinel8.select('VH').mean().rename('VH8');
var image9 = sentinel9.select('VH').mean().rename('VH9');
var image10 = sentinel10.select('VH').mean().rename('VH10');

var stacked = image1.addBands([image2,image3,image4,image5,image6,image7,image8,image9,image10]).clip(dist);
print(stacked);

var stacked_scaled = stacked.multiply(10).add(350).uint8();
var bands = ['VH2', 'VH5', 'VH9'];
var display = {bands: bands, min: 0, max: 220};

Map.addLayer(stacked_scaled, display, 'stacked');
// Map.setCenter(77.00,29.68,8);
var collection = ee.ImageCollection('COPERNICUS/S2_SR') 
                   .filterDate('2021-06-5', '2021-08-30')
                   .filterBounds(dist);
var im = collection.median().clip(dist);
var S2_bands = ['B8', 'B4', 'B3'];
var S2_display = {bands: S2_bands, min: 100, max: 4000};
Map.addLayer(im, S2_display, 'im');

var gt1 =  rice1.merge(rice2).merge(rice3).merge(urban).merge(water).merge(other);

var training = stacked_scaled.sampleRegions({
  collection: gt1,
  properties: ['class'],
  scale: 10
});
// Make a Random Forest classifier and train it.
var classifier = ee.Classifier.smileRandomForest(10)
    .train({
      features: training,
      classProperty: 'class',
      
    });
var classified = stacked_scaled.classify(classifier);

var masked = classified.updateMask(classified.gt(0).and(classified.lt(6)));

Map.addLayer(masked,
             {min: 1, max:3 , palette: ['orange','magenta','yellow']},
             'classification - Kharif');

var areaImage = ee.Image.pixelArea().addBands(masked);
var areas = areaImage.reduceRegion({
  reducer: ee.Reducer.sum().group({
    groupField: 1,
    groupName: 'class',
  }),
  geometry: dist,
  scale: 100,
  maxPixels: 1e10,
  bestEffort: true,
  //tileScale: 8
});

// Print the area calculations.
print('##### AREA SQ. METERS #####');
print(areas);

// Get a confusion matrix representing resubstitution accuracy.
var trainAccuracy = classifier.confusionMatrix();
print('Resubstitution error matrix: ', trainAccuracy);
print('Training overall accuracy: ', trainAccuracy.accuracy());

// Sample the input with a different random seed to get validation data.
var validation = stacked_scaled.sampleRegions({
  collection: gt1,
  properties: ['class'],
  scale: 10
});

// Classify the validation data.
var validated = validation.classify(classifier);

// Get a confusion matrix representing expected accuracy.
var testAccuracy = validated.errorMatrix('class', 'classification');
print('Validation error matrix: ', testAccuracy);
print('Validation overall accuracy: ', testAccuracy.accuracy());
          
var bandInfo = {
  'VH1': {v: 1, f: 'June_1FN'},
  'VH2': {v: 2, f: 'June_2FN'},
  'VH3': {v: 3, f: 'July_1FN'},
  'VH4': {v: 4, f: 'July_2FN'},
  'VH5': {v: 5, f: 'Aug_1FN'},
  'VH6': {v: 6, f: 'Aug_2FN'},
  'VH7': {v: 7, f: 'Sep_1FN'},
  'VH8': {v: 8, f: 'Sep_2FN'},
  'VH9': {v: 9, f: 'Oct_1FN'},
  'VH10': {v: 10, f: 'Oct_2FN'}

  
};

var xPropVals = [];    // List to codify x-axis band names as values.
var xPropLabels = [];  // Holds dictionaries that label codified x-axis values.
for (var key in bandInfo) {
  xPropVals.push(bandInfo[key].v);
  xPropLabels.push(bandInfo[key]);
}
var regionsBand =
    gt1
        .reduceToImage({properties: ['class'], reducer: ee.Reducer.first()})
         .rename('class');

var sentinelSrClass = stacked_scaled.addBands(regionsBand);
       // print(sentinelSrClass);
var chart = ui.Chart.image
                .byClass({
                  image: sentinelSrClass,
                  classBand: 'class',
                  region: gt1,
                  reducer: ee.Reducer.mean(),
                  scale: 10,
                //  classLabels: ['Mustard', 'Wheat'],
                  xLabels: xPropVals
                })
                .setChartType('ScatterChart')
                .setOptions({
                  title: 'Temporal Signatures - Backscatter',
                  hAxis: {
                    title: 'Dates',
                    titleTextStyle: {italic: false, bold: true},
                    viewWindow: {min: bands[0], max: bands[9]},
                    ticks: xPropLabels
                  },
                  vAxis: {
                    title: 'Backscatter(Scaled)',
                    titleTextStyle: {italic: false, bold: true},
                    viewWindow: {min: 0, max: 250},
                  },
                  colors: ['red', 'blue', 'grey', 'green', 'yellow', 'magenta', 'cyan', 'green'],
                  pointSize: 0,
                  lineSize: 2,
                  curveType: 'function'
                });
print(chart);     
        
// Export.image.toDrive({
//     image: masked, // <--
//     description: 'image',
//     scale: 1000,
//     region: dist
// });



// NDVI

// Define ROI
var roi= table; //geometry_roi;
Map.centerObject(roi,10);

//Date
var startDate = ee.Date('2019-01-01');
var endDate =  ee.Date('2020-12-31');

// Create image collection of S-2 imagery for the perdiod 2019-2020
var S2 = ee.ImageCollection('COPERNICUS/S2')
         //filter start and end date
         .filter(ee.Filter.date(startDate, endDate))
         .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than',100)
         //filter according to drawn boundary
         .filterBounds(roi)


print(S2.limit(10))
print(S2.aggregate_array('SPACECRAFT_NAME'))
// Function to calculate and add an NDVI band
var addNDVI = function(image) {
 return image.addBands(image.normalizedDifference(['B8', 'B4'] )); //'B8', 'B4'
};  
  
// Add NDVI band to image collection
var S2 = S2.map(addNDVI).select(['nd']);
print('S2',S2.limit(10)) ;
var NDVI=S2.select('nd');

// For month
var month = 1;

// Calculating number of intervals
var months = endDate.difference(startDate,'month').divide(month).toInt();
// Generating a sequence 
var sequence = ee.List.sequence(0, months); 
// print(sequence)

var sequence_s1 = sequence.map(function(num){
    num = ee.Number(num);
    var Start_interval = startDate.advance(num.multiply(month), 'month');
  
    var End_interval = startDate.advance(num.add(1).multiply(month), 'month');
    var subset = NDVI.filterDate(Start_interval,End_interval);
    return subset.max().set('system:time_start',Start_interval);
});

// print('sequence_s1',sequence_s1)
var byMonthYear = ee.ImageCollection.fromImages(sequence_s1);

// print('byMonthYear',byMonthYear)
var multibandNDVI = byMonthYear.toBands().clip(roi);
// print('multiband', multibandNDVI);



var bandsName=['2019-01','2019-02','2019-03','2019-04','2019-05','2019-06',
               '2019-07','2019-08','2019-09','2019-10','2019-11','2019-12',
               '2020-01','2020-02','2020-03','2020-04','2020-05','2020-06',
               '2020-07','2020-08','2020-09','2020-10','2020-11','2020-12']

var multiband1_ndvi = multibandNDVI.rename(bandsName).clip(roi);//(monList)//
//

//s1
var sentinel1_vh = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .select('VH')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.eq('resolution_meters', 10))
  .filter(ee.Filter.date(startDate, endDate))
  .filter(ee.Filter.bounds(roi))

// print('s1',sentinel1_vh);

// For month
var month = 1;

// Calculating number of intervals
var months = endDate.difference(startDate,'month').divide(month).toInt();
// Generating a sequence 
var sequence = ee.List.sequence(0, months); 
// print(sequence)

var sequence_s1 = sequence.map(function(num){
    num = ee.Number(num);
    var Start_interval = startDate.advance(num.multiply(month), 'month');
  
    var End_interval = startDate.advance(num.add(1).multiply(month), 'month');
    var subset = sentinel1_vh.filterDate(Start_interval,End_interval);
    return subset.median().set('system:time_start',Start_interval);
});

// print('sequence_s1',sequence_s1)
var byMonthYearS1 = ee.ImageCollection.fromImages(sequence_s1);
var multibands1 = byMonthYearS1.toBands().clip(roi);



var multibands1 = multibands1.rename(bandsName).clip(roi);//.rename(monLists1).clip(roi);//

var isNDVIValid = multiband1_ndvi.gte(0).and(multiband1_ndvi.lte(1));
print("isNDVIValid", isNDVIValid)
// Map.addLayer(multiband1_ndvi ,  {min: 0.2, max: 0.8}, 'NDVI', 0);

// Set specific NDVI thresholds for different paddy growth stages
var plantingThreshold = 0.2;
var growthThreshold = 0.5;
var harvestThreshold = 0.2;

// Classify NDVI into different growth stages
var plantingStage = multiband1_ndvi.gt(plantingThreshold);
var growthStage = multiband1_ndvi.gt(growthThreshold).and(multiband1_ndvi.lte(harvestThreshold).not());
var harvestStage = multiband1_ndvi.lte(harvestThreshold);

Map.addLayer(plantingStage ,  {min: 0, max: 1}, 'planting', 0);
Map.addLayer(growthStage ,  {min: 0, max: 1}, 'growth', 0);
Map.addLayer(harvestStage ,  {min: 0, max: 1}, 'harvest', 0);
```
