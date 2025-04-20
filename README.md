# cities_mapping

# QGIS Grid Generator: Hexagonal and Square Grid Aggregation Tool

This Python module is designed for use in **QGIS** as a custom processing algorithm for generating **hexagonal** and **square** grids with aggregation of building data. It enables spatial analysis of urban areas, particularly for visualizing and quantifying building density and structural characteristics such as average number of floors.

## 📦 Features

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

## 📝 Parameters

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

