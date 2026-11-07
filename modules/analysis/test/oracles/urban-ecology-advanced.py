# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Direct pinned upstream urban/decomposition references, not copied GPU formulas.
Run in isolated Python3.12 with momepy0.10.0, segregation2.5.3, numpy2.2.6,
scipy1.15.3 and shapely2.1.2. Geometry uses a projected metre CRS.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import momepy
import segregation
from segregation.singlegroup import Dissim
from segregation.decomposition import DecomposeSegregation
from segregation.inference.comparative import _generate_counterfactual

assert momepy.__version__ == '0.10.0'
assert segregation.__version__ == '2.5.3'
assert np.__version__ == '2.2.6'
assert shapely.__version__ == '2.1.2'

def flatten_segments(polygons):
    segments=[];owners=[]
    for owner,polygon in enumerate(polygons):
        coordinates=list(polygon.exterior.coords)
        for first,second in zip(coordinates[:-1],coordinates[1:]):
            segments.extend([*first,*second]);owners.append(owner)
    return segments,owners

buildings=gpd.GeoDataFrame(geometry=[shapely.box(1,1,2,2),shapely.box(5,1,6,2)],crs=3857)
enclosure=shapely.Polygon([(0,0),(8,0),(8,3),(3,3),(3,8),(0,8)])
tessellation=momepy.morphological_tessellation(buildings,clip=enclosure,shrink=0,segment=1,simplify=False)
segments,owners=flatten_segments(buildings.geometry)
output={'versions':{'momepy':momepy.__version__,'segregation':segregation.__version__,'numpy':np.__version__,'shapely':shapely.__version__,'pandas':pd.__version__},'tessellation':{'segments':segments,'owners':owners,'enclosure':np.array(enclosure.exterior.coords)[:-1].reshape(-1).tolist(),'areas':tessellation.geometry.area.tolist(),'areaUnion':tessellation.geometry.union_all().area}}
holed=shapely.Polygon([(0,0),(8,0),(8,8),(0,8)],holes=[[(3,3),(3,5),(5,5),(5,3)]])
holed_cells=momepy.morphological_tessellation(buildings,clip=holed,shrink=0,segment=1,simplify=False)
output['holedTessellation']={'segments':segments,'owners':owners,'enclosure':np.concatenate((np.array(holed.exterior.coords)[:-1],np.array(holed.interiors[0].coords)[:-1])).reshape(-1).tolist(),'areas':holed_cells.geometry.area.tolist()}
streets=gpd.GeoDataFrame(geometry=[shapely.LineString([(-4,0),(4,0)])],crs=3857)
facades=gpd.GeoDataFrame(geometry=[shapely.box(-10,2,10,3),shapely.box(-10,-4,10,-3)],crs=3857)
profile=momepy.street_profile(streets,facades,distance=1,tick_length=20,height=pd.Series([6.,9.]))
output['streetProfile']={key:float(value) for key,value in profile.iloc[0].items()}
adjacent_buildings=gpd.GeoSeries([shapely.box(0,0,1,2),shapely.box(1,0,2,2),shapely.box(2,2,3,3)],crs=3857)
output['sharedWalls']={'lengths':momepy.shared_walls(adjacent_buildings,strict=True).tolist()}
first=np.array([[10.,30.],[20.,20.],[20.,20.],[0.,0.]])
second=np.array([[30.,10.],[10.,30.],[5.,35.],[0.,0.]])
def frame(population):return pd.DataFrame({'focal':population[:,0],'total':population.sum(axis=1)})
def dissimilarity(population):return float(Dissim(frame(population),'focal','total').statistic)
output['counterfactual']={'baseline':first.reshape(-1).tolist(),'final':second.reshape(-1).tolist(),'methods':{}}
for method in ['composition','dual_composition','share']:
    first_counter,second_counter=_generate_counterfactual(frame(first),frame(second),'focal','total','focal','total',counterfactual_approach=method)
    values=[]
    for counter in [first_counter,second_counter]:values.append(np.column_stack((counter.counterfactual_group_pop,counter.counterfactual_total_pop-counter.counterfactual_group_pop)).reshape(-1).tolist())
    decomposition=DecomposeSegregation(Dissim(frame(first),'focal','total'),Dissim(frame(second),'focal','total'),counterfactual_approach=method)
    output['counterfactual']['methods'][method]={'populations':values,'decomposition':[dissimilarity(second)-dissimilarity(first),-float(decomposition.c_a),-float(decomposition.c_s),0.]}
swaps=[];distribution=[]
for replicate in range(16):
    mask=np.array([(replicate>>unit)&1 for unit in range(4)],dtype=bool)
    swaps.extend(mask.astype(int).tolist())
    distribution.append(dissimilarity(np.where(mask[:,None],first,second))-dissimilarity(np.where(mask[:,None],second,first)))
output['randomization']={'swaps':swaps,'distribution':distribution,'pValue':(1+sum(abs(value)>=abs(dissimilarity(second)-dissimilarity(first)) for value in distribution))/17}
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import scipy
assert scipy.__version__ == '1.15.3'
resistance_edges=np.array([[0,1],[1,2],[2,5],[0,3],[3,4],[4,5],[1,4],[2,3],[0,1],[7,8],[5,6]])
conductances=np.array([1.,3.,.5,2.,1.5,4.,.2,5.,.3,7.,0.])
rows=[];columns=[];weights=[]
for (first_node,second_node),weight in zip(resistance_edges,conductances):
 if first_node>=6 or second_node>=6:continue
 rows.extend([first_node,second_node,first_node,second_node]);columns.extend([first_node,second_node,second_node,first_node]);weights.extend([weight,weight,-weight,-weight])
laplacian=coo_matrix((weights,(rows,columns)),shape=(6,6)).tocsr()
right=np.zeros(5);right[0]=1
potentials=np.concatenate((spsolve(laplacian[:5,:5],right),[0]))
currents=[float(weight*(potentials[first_node]-potentials[second_node])) if first_node<6 and second_node<6 else (0 if weight==0 else None) for (first_node,second_node),weight in zip(resistance_edges,conductances)]
output['sparseResistance']={'edges':resistance_edges.reshape(-1).tolist(),'conductances':conductances.tolist(),'potentials':potentials.tolist()+[None,None,None],'currents':currents,'resistance':float(potentials[0]),'scipy':scipy.__version__}
Path(__file__).with_name('fixtures').joinpath('urban-ecology-advanced.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
from shapely.affinity import rotate
from libpysal.graph import Graph
structure_geometry=gpd.GeoSeries(shapely.orient_polygons([shapely.box(0,0,2,2),shapely.box(2,0,4,2),shapely.Polygon([(6,0),(10,0),(10,4),(6,4)],holes=[[(7,1),(7,3),(9,3),(9,1)]])]),crs=3857)
structure_graph=Graph.build_contiguity(structure_geometry,rook=False)
angles=[10,30,80]
orientation_geometry=gpd.GeoSeries([rotate(shapely.box(-2,-1,2,1),angle) for angle in angles],crs=3857)
cardinal=momepy.orientation(orientation_geometry)
alignment=momepy.alignment(cardinal,structure_graph)
positions=[];rings=[0];polygons=[0];features=[0]
for polygon in structure_geometry:
 for ring in [polygon.exterior,*polygon.interiors]:
  positions.extend(np.array(ring.coords)[:-1].reshape(-1).tolist());rings.append(len(positions)//2)
 polygons.append(len(rings)-1);features.append(len(polygons)-1)
output['structures']={'positions':positions,'ringOffsets':rings,'polygonOffsets':polygons,'featureOffsets':features,'orientations':[np.deg2rad(angle) for angle in angles],'perimeterWall':momepy.perimeter_wall(gpd.GeoDataFrame(geometry=structure_geometry),graph=structure_graph,buffer=0).tolist(),'cardinal':cardinal.tolist(),'alignment':[None if pd.isna(value) else value for value in alignment]}
Path(__file__).with_name('fixtures').joinpath('urban-ecology-advanced.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
