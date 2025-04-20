import os
from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (
    QgsApplication,
    QgsProcessingAlgorithm,
    QgsProcessingParameterVectorLayer,
    QgsProcessingParameterCrs,
    QgsProcessingOutputVectorLayer,
    QgsProject, QgsGraduatedSymbolRenderer,
    QgsVectorLayer, QgsClassificationEqualInterval,
    QgsFeature, QgsRendererRangeLabelFormat,
    QgsGeometry, QgsStyle,
    QgsField, QgsProcessingParameterField,
    QgsFields,
    QgsWkbTypes,
    QgsCoordinateReferenceSystem,
    QgsProcessing,
    QgsProcessingProvider
)
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiPolygon, GeometryCollection, mapping
from shapely.ops import polygonize, unary_union, split, nearest_points
import osmnx as ox
import networkx as nx

def remove_dead_ends_extend(roads):
    G = nx.Graph()
    node_to_index = {}
    for idx, row in roads.iterrows():
        line = row.geometry
        coords = list(line.coords)
        start, end = Point(coords[0]), Point(coords[-1])
        G.add_node(start)
        G.add_node(end)
        node_to_index[start] = idx
        node_to_index[end] = idx
        G.add_edge(start, end, idx=idx)
    dead_ends = [node for node in G.nodes if G.degree(node) == 1]

    new_lines = []
    
    for idx, row in roads.iterrows():
        new_lines.append(row.geometry)
    
    for node in dead_ends:
        road_idx = node_to_index[node]
        road_segment = roads.loc[road_idx].geometry
        extended_line = extend_to_nearest(road_segment, node, roads)

        if extended_line:
            new_lines.append(extended_line)
    
    return gpd.GeoDataFrame(geometry=new_lines, crs=roads.crs)

def extend_to_nearest(line, dead_end, roads):
    coords = list(line.coords)
    if dead_end == Point(coords[0]):
        direction = (coords[1][0] - coords[0][0], coords[1][1] - coords[0][1]) 
    else:
        direction = (coords[-2][0] - coords[-1][0], coords[-2][1] - coords[-1][1])
    
    max_distance = 500  
    extended_point = Point(dead_end.x + direction[0] * max_distance, dead_end.y + direction[1] * max_distance)
    extended_line = LineString([dead_end, extended_point])
    nearest_road = None
    min_dist = float("inf")
    
    for idx, other_road in roads.iterrows():
        if other_road.geometry != line:
            nearest = nearest_points(extended_line, other_road.geometry)[1]
            distance = dead_end.distance(nearest)
            if distance < min_dist:
                min_dist = distance
                nearest_road = nearest
    
    if nearest_road:
        return LineString([dead_end, nearest_road])
    
    return None

def split_line_by_intersections(lines):
    new_segments = []
    for line in lines.geometry:
        if line.geom_type == "MultiLineString":
            for sub_line in line.geoms:
                new_segments.extend(split_line_segments(sub_line, lines))
        else:
            new_segments.extend(split_line_segments(line, lines))
    
    return gpd.GeoDataFrame(geometry=new_segments, crs=lines.crs)

def split_line_segments(line, lines):
    intersection_points = []
    
    for other_line in lines.geometry:
        if line != other_line:
            intersection = line.intersection(other_line)
            if intersection.geom_type == 'Point':
                intersection_points.append(intersection)
            elif intersection.geom_type == 'MultiPoint':
                intersection_points.extend(intersection.geoms)

    if not intersection_points:
        return [line] 
    
    multi_point = MultiPoint(intersection_points) if len(intersection_points) > 1 else intersection_points[0]
    split_lines = split(line, multi_point)

    return list(split_lines.geoms)

def remove_dead_end_segments_in_blocks(merged_big_roads_poly, buffer_size=6):
    buffered_polygons = merged_big_roads_poly.buffer(buffer_size)
    negative_buffered_polygons = buffered_polygons.buffer(-buffer_size)
    processed_polygons = gpd.GeoDataFrame(geometry=negative_buffered_polygons, crs=merged_big_roads_poly.crs)
    processed_polygons = processed_polygons[processed_polygons.geometry.is_valid]
    
    return processed_polygons

def remove_holes(geom):
    if geom.geom_type == "Polygon":
        return Polygon(geom.exterior)
    elif geom.geom_type == "MultiPolygon":
        return MultiPolygon([Polygon(p.exterior) for p in geom.geoms])
    return geom

def get_blocks(gdf_bound, gdf_build_clip, gdf_roads_clip, gdf_waterway, gdf_railroads, prj, column=None):
    # roads selection
    val_big_roads = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential', 'unclassified', 'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link']
    big_roads_clip = gdf_roads_clip[gdf_roads_clip['highway'].isin(val_big_roads)]
    # roads splitting
    big_roads_clip = pd.concat([big_roads_clip, gdf_railroads], ignore_index=True)
    split_big_roads = split_line_by_intersections(big_roads_clip)
    big_roads_clip = remove_dead_ends_extend(split_big_roads)
    # get 1 step blocks
    big_roads_buff = big_roads_clip.geometry.buffer(5)
    big_roads_buff = gpd.GeoDataFrame(geometry = big_roads_buff, crs = big_roads_clip.crs)
    # polygonize big roads
    split_big_roads = split_big_roads.to_crs(prj)
    split_big_roads_poly = list(polygonize(split_big_roads.geometry))
    split_big_roads_poly = gpd.GeoDataFrame(geometry = split_big_roads_poly, crs = split_big_roads.crs)
    big_roads_poly_buff = split_big_roads_poly.overlay(big_roads_buff, how = 'difference')
    merged_polygons = unary_union(big_roads_poly_buff.geometry)
    if merged_polygons.geom_type == "Polygon":
        merged_big_roads_poly = gpd.GeoDataFrame(geometry = [merged_polygons], crs = big_roads_poly_buff.crs)
    elif merged_polygons.geom_type == "MultiPolygon":
        merged_big_roads_poly = gpd.GeoDataFrame(geometry = list(merged_polygons.geoms), crs = big_roads_poly_buff.crs)   
    # 1.5 steps blocks 
    # remove inner deadends parts
    merged_big_roads_poly = remove_dead_end_segments_in_blocks(merged_big_roads_poly, buffer_size=6)
    # 2 step blocks
    # deleting poligons without houses
    gdf_build_clip = gdf_build_clip[gdf_build_clip.geometry.area > 100]
    gdf_build_clip = gdf_build_clip[gdf_build_clip.is_valid]
    gdf_build_clip["geometry"] = gdf_build_clip["geometry"].apply(lambda geom: geom.buffer(0) if not geom.is_valid else geom)
    for geom_block in merged_big_roads_poly.geometry:
        res = gdf_build_clip.clip(geom_block)
        if len(res.geometry) == 0:
            ind = merged_big_roads_poly[merged_big_roads_poly.geometry.apply(lambda x: x.equals(geom_block))].index
            merged_big_roads_poly.drop(index = ind, inplace = True)
        else:
            continue
    # 3 step blocks with waterways
    waterway_buff = gdf_waterway.geometry.buffer(13, resolution=50, cap_style='square')
    waterway_buff = gpd.GeoDataFrame(geometry=waterway_buff, crs=gdf_waterway.crs)
    merged_big_roads_poly2 = merged_big_roads_poly.overlay(waterway_buff, how='difference')
    
    for idx, row in merged_big_roads_poly2.iterrows():
        if row.geometry.area > 500000 or row.geometry.area < 50:  
            merged_big_roads_poly2.drop(idx, inplace=True)

    merged_big_roads_poly2 = merged_big_roads_poly2[~merged_big_roads_poly2.geometry.is_empty]
    
    # remove small holes
    merged_big_roads_poly2["geometry"] = merged_big_roads_poly2["geometry"].apply(lambda g: g.buffer(0))
    merged_big_roads_poly2["geometry"] = merged_big_roads_poly2["geometry"].apply(remove_holes)
    merged_big_roads_poly2.reset_index(drop = True, inplace = True)
    
    # with column
    merged_big_roads_poly2['block_area'] = merged_big_roads_poly2.geometry.area
    
    if column and column in gdf_build_clip.columns:
        merged_big_roads_poly2['count'] = merged_big_roads_poly2.geometry.apply(
            lambda geom: gdf_build_clip[gdf_build_clip.geometry.centroid.within(geom)][column].mean()
        )
        merged_big_roads_poly2['normalized_count'] = merged_big_roads_poly2['count']  / merged_big_roads_poly2.geometry.area * (10**6)
    else:
        merged_big_roads_poly2['count'] = merged_big_roads_poly2.geometry.apply(
            lambda geom: gdf_build_clip.geometry.centroid.within(geom).sum()
        )
        merged_big_roads_poly2['normalized_count'] = merged_big_roads_poly2['count'] / merged_big_roads_poly2.geometry.area * (10**6) 
        
    if column == 'r_floors' and column in gdf_build_clip.columns:
        # FAR (Floor Area Ratio)
        merged_big_roads_poly2['total_floor_area'] = merged_big_roads_poly2.geometry.apply(
            lambda geom: (
                gdf_build_clip[gdf_build_clip.geometry.centroid.within(geom)]['r_floors'] * 
                gdf_build_clip[gdf_build_clip.geometry.centroid.within(geom)].geometry.area
            ).sum()
        )
        merged_big_roads_poly2['far'] = (
            merged_big_roads_poly2['total_floor_area'] / merged_big_roads_poly2['block_area']
        )
        
        merged_big_roads_poly2['normalized_count'] = merged_big_roads_poly2['far']
            
    merged_big_roads_poly2 = merged_big_roads_poly2.to_crs(4326)

    return merged_big_roads_poly2

class CreateBlocksAlgorithm(QgsProcessingAlgorithm):
    BOUNDARY_LAYER = 'BOUNDARY_LAYER'
    ROADS_LAYER = 'ROADS_LAYER'
    RAILROADS_LAYER = 'RAILROADS_LAYER'
    BUILDINGS_LAYER = 'BUILDINGS_LAYER'
    WATERWAY_LAYER = 'WATERWAY_LAYER'
    AGGREGATION_COLUMN = 'AGGREGATION_COLUMN'
    OUTPUT = 'OUTPUT'

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.BOUNDARY_LAYER,
                self.tr('Boundary Layer'),
                types=[QgsProcessing.TypeVectorPolygon]
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.ROADS_LAYER,
                self.tr('Roads Layer'),
                types=[QgsProcessing.TypeVectorLine]
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.RAILROADS_LAYER,
                self.tr('Railroads Layer'),
                types=[QgsProcessing.TypeVectorLine]
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.BUILDINGS_LAYER,
                self.tr('Buildings Layer'),
                types=[QgsProcessing.TypeVectorPolygon]
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.WATERWAY_LAYER,
                self.tr('Waterway Layer'),
                types=[QgsProcessing.TypeVectorLine]
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.AGGREGATION_COLUMN,
                self.tr('Aggregation Column (optional)'),
                parentLayerParameterName=self.BUILDINGS_LAYER,
                optional=True,
                type=QgsProcessingParameterField.Numeric 
            )
        )
        
        self.addOutput(
            QgsProcessingOutputVectorLayer(
                self.OUTPUT,
                self.tr('Output Blocks Layer')
            )
        )
        
        self.addParameter(
            QgsProcessingParameterCrs(
                'TARGET_CRS',
                self.tr('Target CRS'),
                defaultValue='EPSG:4326'
            )
        )
        
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
        classification_method = QgsClassificationEqualInterval()

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
        boundary_layer = self.parameterAsVectorLayer(parameters, self.BOUNDARY_LAYER, context)
        roads_layer = self.parameterAsVectorLayer(parameters, self.ROADS_LAYER, context)
        railroads_layer = self.parameterAsVectorLayer(parameters, self.RAILROADS_LAYER, context)
        buildings_layer = self.parameterAsVectorLayer(parameters, self.BUILDINGS_LAYER, context)
        waterway_layer = self.parameterAsVectorLayer(parameters, self.WATERWAY_LAYER, context)
        target_crs = self.parameterAsCrs(parameters, 'TARGET_CRS', context)
        column = self.parameterAsString(parameters, self.AGGREGATION_COLUMN, context)
        prj = target_crs.authid() 

        gdf_bound = gpd.GeoDataFrame.from_features(boundary_layer.getFeatures())
        gdf_roads = gpd.GeoDataFrame.from_features(roads_layer.getFeatures())
        gdf_build = gpd.GeoDataFrame.from_features(buildings_layer.getFeatures())
        gdf_waterway = gpd.GeoDataFrame.from_features(waterway_layer.getFeatures())
        gdf_railroads = gpd.GeoDataFrame.from_features(railroads_layer.getFeatures())
        
        def ensure_crs(gdf, crs):
            if gdf.crs is None:
                gdf.set_crs(crs, inplace=True)

        ensure_crs(gdf_bound, 'EPSG:4326')  
        ensure_crs(gdf_roads, 'EPSG:4326')
        ensure_crs(gdf_build, 'EPSG:4326')
        ensure_crs(gdf_waterway, 'EPSG:4326')
        ensure_crs(gdf_railroads, 'EPSG:4326')

        gdf_bound = gdf_bound.to_crs(prj)
        gdf_roads = gdf_roads.to_crs(prj)
        gdf_build = gdf_build.to_crs(prj)
        gdf_waterway = gdf_waterway.to_crs(prj)
        gdf_railroads = gdf_railroads.to_crs(prj)
        
        result = get_blocks(gdf_bound, gdf_build, gdf_roads, gdf_waterway, gdf_railroads, prj, column)
        output_layers = {}
        layer = self.create_qgis_layer(result, 'Block Layer')
        self.apply_graduated_style(layer, 'normalized_count')
        QgsProject.instance().addMapLayer(layer)
        output_layers[self.OUTPUT] = layer
        
        return output_layers

    def name(self):
        return 'create_blocks'

    def displayName(self):
        return self.tr('Create Blocks')

    def group(self):
        return self.tr('Custom Tools')

    def groupId(self):
        return 'customtools'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return CreateBlocksAlgorithm()

class CustomProcessingProvider(QgsProcessingProvider):
    def loadAlgorithms(self):
        self.addAlgorithm(CreateBlocksAlgorithm())

def register_algorithm():
    provider = CustomProcessingProvider()
    QgsApplication.processingRegistry().addProvider(provider)

register_algorithm()