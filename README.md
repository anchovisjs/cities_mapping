# cities_mapping

# QGIS Grids Generator (hexagonal and square)

This Python module is designed for use in **QGIS** as a custom processing algorithm for generating **hexagonal** and **square** grids with aggregation of building data. 

## Features

- Automatically fetches **city boundaries, buildings, and road networks** using **OSMnx**.
- Generates **H3 hexagonal grids** or regular **square grids** over the city area.
- Aggregates values from a specified attribute column (e.g., `r_floors`) per grid cell.
- Computes **normalized metrics** (e.g., buildings per km², building density index).
- Applies **graduated styling** to output layers for effective visualization.
- Fully integrated into **QGIS Processing Toolbox** via a custom plugin interface.

---

## Installation and Usage

1. Place the provided Python file into your QGIS Python scripts directory or load it manually.
2. Open **QGIS**.
3. Ensure the following Python libraries are installed in your environment:
   - `osmnx`
   - `geopandas`
   - `shapely`
   - `numpy`
   - `h3`
4. Run the `register_algorithm()` function or restart QGIS to auto-load the tool.
5. Open the **Processing Toolbox** and locate **"Create Grid"** under the **Custom Tools** section.

---

## Parameters

| Parameter | Description |
|----------|-------------|
| `City Name` | Name of the city for fetching OSM data (e.g., `"Kazan, Russia"`) |
| `Grid Type` | Select between Hexagonal, Square, or Both |
| `Resolution` | H3 resolution (integer from 1 to 15) |
| `Cell Size` | Cell size in meters for square grids (optional, calculated automatically if omitted) |
| `Buildings Layer` | (Optional) Custom building layer to override OSM data |
| `Aggregation Column` | Column name for aggregation (e.g., `r_floors`) |

---

## Output Layers

- **Hexagonal Grid Output**: H3-based grid layer with aggregated values.
- **Square Grid Output**: Regular square mesh with the same computed attributes.

Each output contains the `normalized_count` field, representing:
- Object count (if no aggregation column is specified),
- Average value of the specified attribute (if provided),
- **Building density index** (when using `r_floors`).

---

## How It Works

1. The algorithm loads urban geodata using OSMnx.
2. It creates the grid over the city's bounding box.
3. Buildings are spatially assigned to grid cells using their centroids.
4. Aggregated statistics and density metrics are computed.
5. Results are visualized in QGIS with automatic styling.

---


# QGIS Urban Block Generator

This plugin provides a custom QGIS Processing algorithm for generating **urban blocks** based on roads, buildings, waterways, natural objects and railroads infrastructure.

## Features

- Uses road geometry to define urban blocks via **polygonization** of road networks.
- Integrates **railways** and **waterways** into block segmentation.
- Automatically removes dead-end street segments and internal artifacts.
- Performs **area-based filtering** of blocks (e.g., excludes blocks without buildings).
- Aggregates building data by block (e.g., counts, average floors, FAR).
- Outputs styled block polygons with normalized density metrics.

---

## Installation and Usage

1. Copy the Python file into your QGIS Python script folder.
2. Launch **QGIS**.
3. Ensure the following Python libraries are installed:
   - `geopandas`
   - `shapely`
   - `pandas`
   - `osmnx`
   - `networkx`
4. Call `register_algorithm()` or restart QGIS to auto-load the tool.
5. In the **Processing Toolbox**, find the algorithm under **Custom Tools** as **"Create Blocks"**.

---

## Parameters

| Parameter | Description |
|-----------|-------------|
| `Boundary Layer` | Polygon layer defining city or district boundaries |
| `Roads Layer` | Line layer containing road geometry |
| `Railroads Layer` | Line layer with rail infrastructure |
| `Buildings Layer` | Polygon layer with building footprints |
| `Waterway Layer` | Line layer of rivers, canals, etc. |
| `Aggregation Column` | Optional: building attribute to aggregate (e.g., `r_floors`) |
| `Target CRS` | Coordinate system for processing geometry (default: EPSG:4326) |

---

## Output

- A polygon layer representing **urban blocks**.
- Each block includes calculated values:
  - `normalized_count`: buildings per square kilometer or normalized attribute
  - `far`: Floor Area Ratio (if `r_floors` provided)

---

## How It Works

1. Roads and rails are merged and split at intersections.
2. Dead-end road segments are extended to improve connectivity.
3. Polygons are formed by **polygonizing** road networks.
4. Polygons without buildings or too large/small are removed.
5. Waterways are buffered and subtracted from block geometry.
6. Metrics are calculated per block (building count, average, FAR).
7. Final layer is styled with graduated color ramp for visualization.
