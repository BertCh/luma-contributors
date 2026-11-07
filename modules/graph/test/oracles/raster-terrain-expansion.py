# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent library primitives for bounded raster/terrain/geometry expansion.

Inputs are exactly representable float32, planar pixel units, full domains. SciPy median uses
nearest border and its upper median. Nodata fill reference is nearest original donor cKDTree,
not GDAL's quadrant IDW. Flow references use NetworkX on explicit directed topology. Alpha hull
is a composition of SciPy Delaunay + radius selection + Shapely union (not GEOS concave_hull).
No GPU throughput or full-library parity is inferred from these small deterministic fixtures.
"""
import json
from pathlib import Path
import numpy as np
import scipy
from scipy import ndimage
from scipy.spatial import Delaunay, cKDTree
import shapely
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
import skimage
from skimage.segmentation import watershed
from skimage.morphology import reconstruction
import networkx as nx

rank_source = np.array([[1, 2, 3], [4, 99, 6], [7, 8, 9]], dtype=np.float32)
hole_source = np.ones((5, 5), dtype=bool)
hole_source[1:4, 1:4] = False
hole_source[0, 0] = False
marker = np.array([[0, 0, 0], [0, 3, 0], [0, 0, 0]], dtype=np.float32)
mask = np.array([[1, 2, 1], [2, 3, 2], [1, 2, 1]], dtype=np.float32)
flow_graph = nx.DiGraph([(0, 1), (1, 2), (2, 5), (3, 4), (4, 5)])
flow_distances = [nx.shortest_path_length(flow_graph, cell, 5) for cell in range(6)]
coverage_polygon = Polygon([(0.25, 0.25), (2.75, 0.25), (2.75, 2.75), (0.25, 2.75)], holes=[[(1,1),(2,1),(2,2),(1,2)]])
coverage = [coverage_polygon.intersection(box(column,row,column+1,row+1)).area for row in range(3) for column in range(3)]
segmentation_source = np.array([[0,1,2,1,0],[0,1,2,1,0],[0,1,2,1,0]], dtype=np.float32)
segmentation_markers = np.zeros((3,5),dtype=np.uint32)
segmentation_markers[1,0] = 1
segmentation_markers[1,4] = 2
medial_source=np.ones((3,3),dtype=bool)
medial_distances=ndimage.distance_transform_edt(np.pad(medial_source,1))[1:-1,1:-1]
alpha_positions=np.array([[0,0],[2,0],[2,2],[0,2],[1,0.5]],dtype=np.float32)
alpha_triangles=[]
for simplex in Delaunay(alpha_positions).simplices:
    triangle=alpha_positions[simplex].astype(float)
    area=Polygon(triangle).area
    lengths=[np.linalg.norm(triangle[vertex]-triangle[(vertex+1)%3]) for vertex in range(3)]
    radius=np.prod(lengths)/(4*area)
    if radius<=1.25:
        alpha_triangles.append(Polygon(triangle))
alpha_geometry=unary_union(alpha_triangles)
cdt_polygon=Polygon([(0,0),(4,0),(4,3),(2,1),(0,3)])
cdt_triangles=shapely.constrained_delaunay_triangles(cdt_polygon)
fixture={
 'versions': {'numpy':np.__version__,'scipy':scipy.__version__,'shapely':shapely.__version__,'scikitImage':skimage.__version__,'networkx':nx.__version__},
 'median':{'width':3,'height':3,'values':rank_source.ravel().tolist(),'expected':ndimage.median_filter(rank_source,size=3,mode='nearest').ravel().tolist()},
 'holes':{'width':5,'height':5,'values':hole_source.astype(int).ravel().tolist(),'expected':ndimage.binary_fill_holes(hole_source).astype(int).ravel().tolist()},
 'reconstruction':{'marker':marker.ravel().tolist(),'mask':mask.ravel().tolist(),'expected':reconstruction(marker,mask,method='dilation',footprint=ndimage.generate_binary_structure(2,1)).ravel().tolist()},
 'flow':{'width':3,'height':2,'directions':[1,1,4,1,1,0],'expected':flow_distances},
 'fractional':{'width':3,'height':3,'positions':[coordinate for ring in [coverage_polygon.exterior,*coverage_polygon.interiors] for point in list(ring.coords)[:-1] for coordinate in point],'ringOffsets':[0,4,8],'expected':coverage,'area':coverage_polygon.area},
 'segmentation':{'width':5,'height':3,'values':segmentation_source.ravel().tolist(),'markers':segmentation_markers.ravel().tolist(),'expected':watershed(segmentation_source,segmentation_markers,connectivity=1).ravel().tolist()},
 'medial':{'width':3,'height':3,'values':medial_source.astype(int).ravel().tolist(),'expectedRadius':medial_distances.ravel().tolist()},
 'alpha':{'positions':alpha_positions.ravel().tolist(),'maximumCircumradius':1.25,'area':alpha_geometry.area,'perimeter':alpha_geometry.length},
 'constrained':{'positions':[coordinate for point in list(cdt_polygon.exterior.coords)[:-1] for coordinate in point],'area':cdt_polygon.area,'triangleCount':len(cdt_triangles.geoms)}
}
Path(__file__).with_suffix('.json').write_text(json.dumps(fixture,indent=2)+'\n')
