import os
import processing
from qgis.PyQt.QtCore import QCoreApplication, QVariant
from PyQt5.QtGui import QColor
from qgis.core import (
    QgsApplication, QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterString,
    QgsProcessingParameterNumber,
    QgsProcessingParameterEnum,
    QgsProcessingOutputVectorLayer,
    QgsProject, QgsProcessingProvider,
    QgsVectorLayer, QgsFeature, QgsGeometry, QgsField,
    QgsFields, QgsWkbTypes, QgsCoordinateReferenceSystem,
    QgsGraduatedSymbolRenderer, QgsRendererRange, QgsMarkerSymbol,
    QgsProcessingParameterField, QgsGradientColorRamp, QgsClassificationEqualInterval, QgsClassificationPrettyBreaks,
    QgsRendererRangeLabelFormat, QgsStyle, QgsProcessingParameterVectorLayer 
)
import geopandas as gpd
import pandas as pd
import shapely.geometry
import numpy as np
import h3
import osmnx as ox
from shapely.geometry import Polygon
from shapely.geometry import mapping

# настройка кэша
new_cache_path = os.path.expanduser(f"D:\karta\osm_cache")
os.makedirs(new_cache_path, exist_ok=True)
ox.settings.cache_folder = new_cache_path
ox.settings.use_cache = True

def get_city_data(city_name):
    city_boundary = ox.geocode_to_gdf(city_name)
    buildings = ox.geometries_from_place(city_name, tags={'building': True})
    roads = ox.graph_from_place(city_name, network_type='all')
    roads_gdf = ox.graph_to_gdfs(roads, nodes=False, edges=True)
    return city_boundary, buildings, roads_gdf

def get_hex_grid(gdf_bbox, gdf_build, res, column=None):
    for idx, row in gdf_bbox.iterrows():
        minx, miny, maxx, maxy = row.geometry.bounds

    cellsize = 0.0001
    gdf_build_clip = gdf_build.cx[minx:maxx, miny:maxy]
    poly = shapely.geometry.box(minx - cellsize, miny - cellsize, maxx + cellsize, maxy + cellsize)
    hex_polygons = []
    h3_indexes = h3.polyfill(mapping(poly), res, geo_json_conformant=True)

    for h in h3_indexes:
        hex_geo = h3.h3_to_geo_boundary(h, geo_json=True)
        hex_polygons.append(Polygon(hex_geo))

    hex_gdf = gpd.GeoDataFrame(geometry=hex_polygons, crs='EPSG:4326')

    if gdf_build_clip.empty:
        print("Нет зданий в границах области!")
        return hex_gdf 

    gdf_build_clip['h3_index'] = gdf_build_clip.geometry.centroid.apply(
        lambda geom: str(h3.geo_to_h3(geom.y, geom.x, res)) if geom else None
    )
    
    # with column
    if column and column in gdf_build_clip.columns:
        aggr = gdf_build_clip.groupby('h3_index')[column].mean().reset_index(name='count')
    else:
        aggr = gdf_build_clip.groupby('h3_index').size().reset_index(name='count')
    
    valid_h3_indexes = set(h3_indexes)
    aggr = aggr[aggr['h3_index'].isin(valid_h3_indexes)]
    aggr['h3_index'] = aggr['h3_index'].astype(str)
    h3_areas_km2 = {index: h3.hex_area(resolution=res, unit='km^2') for index in valid_h3_indexes}
    
    aggr['normalized_count'] = aggr.apply(
        lambda row: row['count'] / h3_areas_km2.get(row['h3_index'], 1), axis=1
    )
    
    # building density
    if column == 'r_floors' and column in gdf_build_clip.columns:
        gdf_build_clip = gdf_build_clip.to_crs(3857)
        gdf_bbox = gdf_bbox.to_crs(3857)
        city_area = gdf_bbox.geometry.area / (10 ** 6)
        gdf_build_clip['building_area'] = gdf_build_clip.geometry.area
        building_stats = gdf_build_clip.groupby('h3_index').agg(
            total_building_area=('building_area', 'sum'), 
            mean_floors=('r_floors', 'mean')
        ).reset_index()
        building_stats['total_building_area_km2'] = building_stats['total_building_area'] / (10 ** 6)
        building_stats['building_coverage_ratio'] = building_stats.apply(
            lambda row: (row['total_building_area_km2'] / city_area) * 100, axis=1
        )
        building_stats['building_density'] = building_stats.apply(
            lambda row: row['mean_floors'] * row['building_coverage_ratio'], axis=1
        )

        aggr = aggr.merge(building_stats, on='h3_index', how='left')
        aggr['normalized_count'] = aggr['building_density'] 
        gdf_build_clip = gdf_build_clip.to_crs(4326)
        gdf_bbox = gdf_bbox.to_crs(4326)

    h3_df = pd.DataFrame({'h3_index': [str(h) for h in valid_h3_indexes]})
    h3_df = h3_df.merge(aggr, on='h3_index', how='left').fillna(0)

    h3_df['geometry'] = h3_df['h3_index'].apply(lambda h: Polygon(h3.h3_to_geo_boundary(h, geo_json=True)))
    gdf_h3 = gpd.GeoDataFrame(h3_df, geometry='geometry', crs='EPSG:4326')

    return gdf_h3[gdf_h3['count'] > 0]

def get_square_grid(gdf_bbox, cellsize, gdf_build, column=None):    
    gdf_bbox = gdf_bbox.to_crs(54009)
    for idx, row in gdf_bbox.iterrows():
        minx, miny, maxx, maxy = row.geometry.bounds
        
    gdf_build = gdf_build.to_crs(54009)
    gdf_build_clip = gdf_build.cx[minx:maxx, miny:maxy]
    buffered_bounds = shapely.geometry.box(minx - cellsize, miny - cellsize, maxx + cellsize, maxy + cellsize)
    x_coords = np.arange(buffered_bounds.bounds[0], buffered_bounds.bounds[2], cellsize)
    y_coords = np.arange(buffered_bounds.bounds[1], buffered_bounds.bounds[3], cellsize)
    gridcells = []
    for x in x_coords:
        for y in y_coords:
            gridcells.append(shapely.geometry.box(x, y, x + cellsize, y + cellsize))
    grid = gpd.GeoDataFrame({'geometry': gridcells}, crs=gdf_build.crs)
    
    # with column
    if column and column in gdf_build_clip.columns:
        grid['count'] = grid.geometry.apply(
            lambda cell: gdf_build_clip[gdf_build_clip.geometry.centroid.within(cell)][column].mean()
        )
    else:
        grid['count'] = grid.geometry.apply(
            lambda cell: gdf_build_clip.geometry.centroid.within(cell).sum()
        )
    
    grid['normalized_count'] = grid['count'] / grid.geometry.area * (10**6) # km2
    
    # building density
    if column == 'r_floors' and column in gdf_build_clip.columns:
        gdf_build_clip['building_area'] = gdf_build_clip.geometry.area
        city_area = gdf_bbox.geometry.area
        grid['total_building_area'] = grid.geometry.apply(
            lambda cell: gdf_build_clip[gdf_build_clip.geometry.centroid.within(cell)]['building_area'].sum()
        )
        grid['mean_floors'] = grid.geometry.apply(
            lambda cell: gdf_build_clip[gdf_build_clip.geometry.centroid.within(cell)]['r_floors'].mean()
        )
        grid['building_coverage_ratio'] = grid.apply(
            lambda row: (row['total_building_area'] / city_area) * 100, axis=1
        )
        grid['building_density'] = grid.apply(
            lambda row: row['mean_floors'] * row['building_coverage_ratio'], axis=1
        )
        grid['normalized_count'] = grid['building_density']
    
    grid = grid[grid['count'] != 0]
    grid = grid.to_crs(4326)
    
    return grid

def calculate_square_cell_size(resolution):
    hex_area = h3.hex_area(resolution, unit='m^2') 
    square_side = 2 * np.sqrt((2 / np.sqrt(3)) * hex_area) 
    return square_side

class CreateGridAlgorithm(QgsProcessingAlgorithm):
    CITY = 'CITY'
    GRID_TYPE = 'GRID_TYPE'
    CELL_SIZE = 'CELL_SIZE'
    RESOLUTION = 'RESOLUTION'
    BUILDINGS_LAYER = 'BUILDINGS_LAYER'
    AGGREGATION_COLUMN = 'AGGREGATION_COLUMN'
    OUTPUT_HEX = 'OUTPUT_HEX'
    OUTPUT_SQUARE = 'OUTPUT_SQUARE'
    
    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterString(
                self.CITY,
                self.tr('City Name')
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.GRID_TYPE,
                self.tr('Grid Type'),
                options=[self.tr('Hexagonal'), self.tr('Square'), self.tr('All')]
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.RESOLUTION,
                self.tr('Resolution (for Hexagonal Grid)'),
                type=QgsProcessingParameterNumber.Integer,
                minValue=1,
                maxValue=15,
                defaultValue=8
            )
        )

        default_cell_size = calculate_square_cell_size(8) 

        self.cell_size_param = QgsProcessingParameterNumber(
            self.CELL_SIZE,
            self.tr('Cell Size'),
            type=QgsProcessingParameterNumber.Double,
            minValue=0,
            defaultValue=default_cell_size,
            optional=True 
        )
        self.addParameter(self.cell_size_param)
        
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.BUILDINGS_LAYER,
                self.tr('Buildings Layer'),
                optional=True,
                types=[QgsProcessing.TypeVectorPolygon] 
            )
        )
        
        self.addParameter(
            QgsProcessingParameterField(
                self.AGGREGATION_COLUMN,
                self.tr('Aggregation Column'),
                parentLayerParameterName=self.BUILDINGS_LAYER,  # Привязываем к слою BUILDINGS_LAYER
                optional=True
            )
        )
        
        self.addOutput(
            QgsProcessingOutputVectorLayer(
                self.OUTPUT_HEX,
                self.tr('Hexagonal Grid Output')
            )
        )
        self.addOutput(
            QgsProcessingOutputVectorLayer(
                self.OUTPUT_SQUARE,
                self.tr('Square Grid Output')
            )
        )

    def name(self):
        return 'create_grid'

    def displayName(self):
        return self.tr('Create Grid')

    def group(self):
        return self.tr('Custom Tools')

    def groupId(self):
        return 'customtools'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return CreateGridAlgorithm()

    def create_qgis_layer(self, gdf, layer_name):
        features = []
        fields = QgsFields()
        fields.append(QgsField("id", QVariant.Int))
        fields.append(QgsField("normalized_count", QVariant.Double))

        for idx, row in gdf.iterrows():
            feat = QgsFeature(fields)
            feat.setId(int(idx))
            feat.setGeometry(QgsGeometry.fromWkt(row.geometry.wkt))
            feat.setAttributes([int(idx), float(row['normalized_count'])])
            features.append(feat)

        layer = QgsVectorLayer("Polygon?crs=EPSG:4326", layer_name, "memory")
        provider = layer.dataProvider()
        provider.addAttributes(fields)
        layer.updateFields()
        provider.addFeatures(features)
        layer.updateExtents()
        return layer

    def apply_graduated_style(self, layer, field_name):
        renderer = QgsGraduatedSymbolRenderer()
        renderer.setClassAttribute(field_name)

        ramp_name = 'Viridis'
        color_ramp = QgsStyle().defaultStyle().colorRamp(ramp_name)

        num_classes = 5
        classification_method = QgsClassificationPrettyBreaks()

        format = QgsRendererRangeLabelFormat()
        format.setFormat("%1 - %2")
        format.setPrecision(2)
        format.setTrimTrailingZeroes(True)

        renderer.setClassificationMethod(classification_method)
        renderer.setLabelFormat(format)
        renderer.updateClasses(layer, num_classes)
        renderer.updateColorRamp(color_ramp)

        layer.setRenderer(renderer)
        layer.triggerRepaint()
    
    def processAlgorithm(self, parameters, context, feedback):           
        city_name = self.parameterAsString(parameters, self.CITY, context)
        resolution = self.parameterAsInt(parameters, self.RESOLUTION, context)
        cell_size = self.parameterAsDouble(parameters, self.CELL_SIZE, context)
        grid_type = self.parameterAsEnum(parameters, self.GRID_TYPE, context)
        column = self.parameterAsString(parameters, self.AGGREGATION_COLUMN, context)
        buildings_layer = self.parameterAsVectorLayer(parameters, self.BUILDINGS_LAYER, context)

        if not cell_size or cell_size < 1:
            cell_size = calculate_square_cell_size(resolution)
        
        if buildings_layer:
            gdf_build = gpd.GeoDataFrame.from_features(buildings_layer.getFeatures(), crs=buildings_layer.crs().toWkt())
            gdf_build = gdf_build.to_crs(4326)
            city_boundary, gdf_build_2, roads_gdf = get_city_data(city_name)
        else:
            city_boundary, gdf_build, roads_gdf = get_city_data(city_name)
                
        output_layers = {}
        
        if grid_type in [0, 2]:  
            gdf_h3 = get_hex_grid(city_boundary, gdf_build, resolution, column)
            layer = self.create_qgis_layer(gdf_h3, 'Hexagonal Grid')
            self.apply_graduated_style(layer, 'normalized_count')
            QgsProject.instance().addMapLayer(layer)
            output_layers[self.OUTPUT_HEX] = layer

        if grid_type in [1, 2]: 
            grid = get_square_grid(city_boundary, cell_size, gdf_build, column)
            layer = self.create_qgis_layer(grid, 'Square Grid')
            self.apply_graduated_style(layer, 'normalized_count')
            QgsProject.instance().addMapLayer(layer)
            output_layers[self.OUTPUT_SQUARE] = layer
            
        return output_layers

class CustomProcessingProvider(QgsProcessingProvider):
    def loadAlgorithms(self):
        self.addAlgorithm(CreateGridAlgorithm())

def register_algorithm():
    provider = CustomProcessingProvider()
    QgsApplication.processingRegistry().addProvider(provider)

register_algorithm()