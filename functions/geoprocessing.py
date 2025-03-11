import os
import requests

import cv2
import math
import numpy as np
import geopandas as gpd

import rasterio
from rasterio.merge import merge
from rasterio.transform import from_bounds

import folium

from shapely.geometry import box, Point





# ----------------------------------------------
# 1. CONSTANTS AND HELPERS
# ----------------------------------------------
tms_url = "http://www.google.cn/maps/vt?lyrs=s@189&gl=cn&x={x}&y={y}&z={z}"
# "https://c.tile.openstreetmap.org/{z}/{x}/{y}.png" #"http://ecn.t3.tiles.virtualearth.net/tiles/a%7Bq%7D.jpeg?g=1"
TILE_SIZE = 256  # Standard tile size in pixels


def get_tile_url(x, y, z):
    return tms_url.format(x=x, y=y, z=z)


def latlon_to_xyz(lat, lon, zoom):
    """Convert latitude and longitude to XYZ tile indices."""
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y, zoom


def xyz_to_bounds(x, y, z):
    """Get lat/lon bounds for a given XYZ tile."""
    n = 2 ** z
    lon_min = x / n * 360.0 - 180.0
    lon_max = (x + 1) / n * 360.0 - 180.0
    lat_min = math.degrees(
        math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon_min, lat_min, lon_max, lat_max


# ----------------------------------------------
# 2. DEFINE AOI
# ----------------------------------------------
def define_aoi_by_point(coordinate, buffer_meters):
    point = gpd.GeoSeries([Point(coordinate)], crs="EPSG:4326")
    point_projected = point.to_crs("EPSG:3857")
    buffer_projected = point_projected.buffer(buffer_meters, cap_style=3)
    buffer_wgs84 = buffer_projected.to_crs("EPSG:4326")
    return gpd.GeoDataFrame(geometry=buffer_wgs84, crs="EPSG:4326")

def define_aoi_by_wkt(wkt_string): 
    return gpd.GeoDataFrame(geometry=gpd.GeoSeries.from_wkt([wkt_string]), crs="EPSG:4326")

def get_xyz_tiles(aoi_gdf, zoom):
    """Retrieve all XYZ tiles covering an AOI at a given zoom level."""
    if aoi_gdf.empty or aoi_gdf.geometry.is_empty.any():
        print("AOI is empty. No tiles to fetch.")
        return []

    # Extract bounding box (min lon, min lat, max lon, max lat)
    minx, miny, maxx, maxy = aoi_gdf.total_bounds  # Bounding box of AOI

    # Ensure correct lat/lon ordering
    x_min, y_max, _ = latlon_to_xyz(maxy, minx, zoom)  # Top-left corner
    x_max, y_min, _ = latlon_to_xyz(miny, maxx, zoom)  # Bottom-right corner

    # Ensure tile indices are correctly ordered
    x_min, x_max = sorted([x_min, x_max])
    y_min, y_max = sorted([y_min, y_max])

    # Validate tile range
    if x_min > x_max or y_min > y_max:
        print(
            f"Warning: Computed tile range is invalid. x: [{x_min}, {x_max}], y: [{y_min}, {y_max}]")
        return []

    tiles = [(x, y, zoom) for x in range(x_min, x_max + 1)
             for y in range(y_min, y_max + 1)]

    if not tiles:
        print("No tiles found within AOI bounds.")

    return tiles

# ----------------------------------------------
# 3. TILE DOWNLOADING
# ----------------------------------------------
def download_xyz_tile(x, y, z, output_dir):
    os.makedirs(output_dir, exist_ok = True)
    url = get_tile_url(x, y, z)
    file_path = os.path.join(output_dir, f"{x}_{y}_{z}.png")
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"Saved: {file_path}")
    else:
        print(
            f"Failed to download {url} - Status Code: {response.status_code}")

def download_xyz_tile_as_geotiff(x, y, z, output_dir, verbose=True):
    """Download a tile and save as a correctly georeferenced GeoTIFF."""
    os.makedirs(output_dir, exist_ok = True)
    url = get_tile_url(x, y, z)
    file_path = os.path.join(output_dir, f"{x}_{y}_{z}.tif")
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        image = np.asarray(bytearray(response.content), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)  # Preserve all channels
        # Convert to RGB
        if image.shape[-1] == 3:  # RGB case
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        elif image.shape[-1] == 4:  # RGBA case
            image[:, :, :3] = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)

        image = np.flipud(image)  # Flip image vertically
        lon_min, lat_min, lon_max, lat_max = xyz_to_bounds(x, y, z)
        transform = from_bounds(lon_min, lat_max, lon_max, lat_min, image.shape[1], image.shape[0])

        # Save as GeoTIFF
        with rasterio.open(
            file_path,
            "w",
            driver="GTiff",
            height=image.shape[0],
            width=image.shape[1],
            count=image.shape[-1],  # Number of bands (3 for RGB, 4 for RGBA)
            dtype=image.dtype,
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            for i in range(image.shape[-1]):  # Write all channels
                dst.write(image[:, :, i], i + 1)
        if verbose:
            print(f"Saved corrected GeoTIFF: {file_path}")
    
    else:
        print(f"Failed to download {url} - Status Code: {response.status_code}")


def xyz_to_quadkey(x, y, z):
    quadkey = ""
    for i in range(z, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (x & mask) != 0:
            digit += 1
        if (y & mask) != 0:
            digit += 2
        quadkey += str(digit)
    return quadkey

def download_quadkey_tile(x, y, z, output_dir, verbose=True):
    os.makedirs(output_dir, exist_ok = True)
    quadkey = xyz_to_quadkey(x, y, z)
    url = f"http://ecn.t3.tiles.virtualearth.net/tiles/a{quadkey}.jpeg?g=1"
    file_path = os.path.join(output_dir, f"{x}_{y}_{z}.tif")

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        image = np.asarray(bytearray(response.content), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)  # Preserve all channels
        if image.shape[-1] == 3:  # RGB case
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[-1] == 4:  # RGBA case
            image[:, :, :3] = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)

        image = np.flipud(image)
        lon_min, lat_min, lon_max, lat_max = xyz_to_bounds(x, y, z)
        transform = from_bounds(lon_min, lat_max, lon_max, lat_min, image.shape[1], image.shape[0])
        with rasterio.open(
            file_path,
            "w",
            driver="GTiff",
            height=image.shape[0],
            width=image.shape[1],
            count=image.shape[-1],  # Number of bands (3 for RGB, 4 for RGBA)
            dtype=image.dtype,
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            for i in range(image.shape[-1]):  # Write all channels
                dst.write(image[:, :, i], i + 1)

        if verbose:
            print(f"Saved tile as GeoTIFF: {file_path}")

    else:
        print(f"Failed to download {url} - Status Code: {response.status_code}")

def merge_tiles(tile_paths, output_path):
    """Merge multiple GeoTIFF tiles into a single raster."""
    src_files_to_mosaic = []

    for tile_path in tile_paths:
        src = rasterio.open(tile_path)
        src_files_to_mosaic.append(src)

    mosaic, out_trans = merge(src_files_to_mosaic)

    out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "crs": src.crs
    })

    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    for src in src_files_to_mosaic:
        src.close()

    print(f"Merged tiles saved to: {output_path}")

def crop_raster_to_aoi(raster_path, aoi_gdf, output_path):
    """Crop a raster to the boundaries of an AOI."""
    with rasterio.open(raster_path) as src:
        # Ensure the AOI is in the same CRS as the raster
        aoi_gdf = aoi_gdf.to_crs(src.crs)

        # Mask the raster using the AOI geometry
        out_image, out_transform = mask(src, aoi_gdf.geometry, crop=True)
        out_meta = src.meta.copy()

        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "crs": src.crs
        })

        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)

    print(f"Cropped raster saved to: {output_path}")

# ----------------------------------------------
# VISUALIZE AOI
# ----------------------------------------------
def visualize_aoi_and_point(aoi_gdf, point_coords=None, zoom_start=15):
    if point_coords:
        center = point_coords[::-1]  # Convert to (lat, lon) for folium
    else:
        center = [aoi_gdf.geometry.centroid.y.mean(
        ), aoi_gdf.geometry.centroid.x.mean()]
    f = folium.Figure(width=600, height=400)
    m = folium.Map(location=center, zoom_start=zoom_start,
                   tiles="Cartodb Positron").add_to(f)
    if not aoi_gdf.empty:
        for _, row in aoi_gdf.iterrows():
            folium.GeoJson(
                row["geometry"],
                name="AOI",
                style_function=lambda x: {
                    "fillColor": 'cadetblue',
                    "color": 'cadetblue',
                    "weight": 2,
                    "fillOpacity": 0.3,
                },
            ).add_to(m)
    if point_coords:
        folium.Marker(
            location=point_coords[::-1],  # Convert to (lat, lon)
            popup="Approximate Location",
            icon=folium.Icon(color='cadetblue', icon='camera'),
        ).add_to(m)
    folium.LayerControl().add_to(m)
    return m

# ----------------------------------------------
# DOWNLOAD TILES
# ----------------------------------------------
def download_quadkey_tile(x, y, z, output_dir, verbose=True):
    os.makedirs(output_dir, exist_ok = True)
    quadkey = xyz_to_quadkey(x, y, z)
    url = f"http://ecn.t3.tiles.virtualearth.net/tiles/a{quadkey}.jpeg?g=1"
    file_path = os.path.join(output_dir, f"{x}_{y}_{z}.tif")

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        image = np.asarray(bytearray(response.content), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)  # Preserve all channels
        if image.shape[-1] == 3:  # RGB case
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[-1] == 4:  # RGBA case
            image[:, :, :3] = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)

        # Calculate bounds and transform
        lon_min, lat_min, lon_max, lat_max = xyz_to_bounds(x, y, z)
        transform = from_bounds(lon_min, lat_min, lon_max, lat_max, image.shape[1], image.shape[0])

        # Save the image without flipping
        with rasterio.open(
            file_path,
            "w",
            driver="GTiff",
            height=image.shape[0],
            width=image.shape[1],
            count=image.shape[-1],  # Number of bands (3 for RGB, 4 for RGBA)
            dtype=image.dtype,
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            for i in range(image.shape[-1]):  # Write all channels
                dst.write(image[:, :, i], i + 1)

        if verbose:
            print(f"Saved tile as GeoTIFF: {file_path}")