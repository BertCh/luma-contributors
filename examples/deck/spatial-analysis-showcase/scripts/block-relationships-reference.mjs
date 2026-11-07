// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {readFile, writeFile} from 'node:fs/promises';
import {addMetersToLngLat, getDistanceScales} from '@math.gl/web-mercator';

// GEOS is the independent planar geometry oracle. The only shared math is the local display-frame
// projection used by the showcase; projected source coordinates are rounded to float32 where the
// application uploads them, while data.extent retains its original double-precision values.
const python = process.argv[2] ?? '/tmp/ahk-showcase-venv/bin/python';
const outputFile = process.argv[3] ?? '/tmp/block-relationships-reference.json';
const sourceIds = [342227, 815550, 815551];
const sourceRows = [21, 1118, 1119];
const addresses = ['16 W Erie St', '14 W Erie St', '22 W Erie St'];
const origin = [-87.62887, 41.89431];
const collection = JSON.parse(
  await readFile(
    new URL('../public/data/chicago-buildings/buildings.geojson', import.meta.url),
    'utf8'
  )
);
const scales = getDistanceScales({longitude: origin[0], latitude: origin[1], highPrecision: true});
const metersPerDegree = scales.unitsPerDegree.map(
  (value, axis) => value * scales.metersPerUnit[axis]
);
function project(longitude, latitude) {
  let horizontal = (longitude - origin[0]) * metersPerDegree[0];
  let vertical = (latitude - origin[1]) * metersPerDegree[1];
  const [estimatedLongitude, estimatedLatitude] = addMetersToLngLat(origin, [horizontal, vertical]);
  horizontal += (longitude - estimatedLongitude) * metersPerDegree[0];
  vertical += (latitude - estimatedLatitude) * metersPerDegree[1];
  return [horizontal, vertical];
}

const inputFeatures = sourceIds.map((sourceId, sourceIndex) => {
  const feature = collection.features.find(item => Number(item.properties.bldg_id) === sourceId);
  assert.ok(feature, `Missing archived source building ${sourceId}`);
  assert.equal(Number(feature.properties.bldg_id), sourceId);
  assert.equal(collection.features.indexOf(feature), sourceRows[sourceIndex]);
  const geometryType = feature.geometry.type;
  assert.ok(geometryType === 'Polygon' || geometryType === 'MultiPolygon');
  const polygons = geometryType === 'Polygon' ? [feature.geometry.coordinates] : feature.geometry.coordinates;
  const projectedDouble = polygons.map(polygon =>
    polygon
      .filter(ring => ring.length >= 4)
      .map(ring => {
        const projectedRing = ring.map(([longitude, latitude]) => project(longitude, latitude));
        const first = projectedRing[0];
        const last = projectedRing.at(-1);
        if (first[0] !== last[0] || first[1] !== last[1]) projectedRing.push(first);
        return projectedRing;
      })
      .filter(ring => ring.length >= 4)
  );
  const uploaded = projectedDouble.map(polygon =>
    polygon.map(ring => ring.map(([positionX, positionY]) => [Math.fround(positionX), Math.fround(positionY)]))
  );
  return {
    sourceId,
    sourceRow: sourceRows[sourceIndex],
    address: addresses[sourceIndex],
    sourceOriginalBuildingId: feature.properties.orig_bldg_,
    sourceFootprint: feature.properties.footprint_,
    sourceStatus: feature.properties.bldg_statu,
    geometryType,
    projectedDouble,
    polygons: uploaded
  };
});

const pythonProgram = String.raw`import json, math, sys
import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box, MultiPoint
from shapely.ops import unary_union

rows=json.load(sys.stdin)
assert shapely.__version__.startswith('2.1.2'), 'Reference expects Shapely 2.1.2'
assert shapely.geos_version_string.startswith('3.13.1'), 'Reference expects GEOS 3.13.1'

def polygon_from_rings(rings):
    return Polygon(rings[0], rings[1:])

def area_and_bounds(geometry):
    return {'area': geometry.area, 'bounds': list(geometry.bounds)}

def bfs(adjacency, start):
    distances={start:0}
    queue=[start]
    for vertex in queue:
        for neighbor in adjacency[vertex]:
            if neighbor not in distances:
                distances[neighbor]=distances[vertex]+1
                queue.append(neighbor)
    return distances

def graph_metrics(adjacency):
    vertex_count=len(adjacency)
    metrics=[]
    betweenness=[0.0]*vertex_count
    for source in range(vertex_count):
        predecessors=[[] for _ in range(vertex_count)]
        path_counts=[0]*vertex_count
        path_counts[source]=1
        distances=[-1]*vertex_count
        distances[source]=0
        stack=[]
        queue=[source]
        for vertex in queue:
            stack.append(vertex)
            for neighbor in adjacency[vertex]:
                if distances[neighbor] < 0:
                    distances[neighbor]=distances[vertex]+1
                    queue.append(neighbor)
                if distances[neighbor] == distances[vertex]+1:
                    path_counts[neighbor]+=path_counts[vertex]
                    predecessors[neighbor].append(vertex)
        dependency=[0.0]*vertex_count
        while stack:
            vertex=stack.pop()
            for predecessor in predecessors[vertex]:
                dependency[predecessor] += (path_counts[predecessor]/path_counts[vertex])*(1+dependency[vertex])
            if vertex != source:
                betweenness[vertex] += dependency[vertex]
        reachable=[distance for index,distance in enumerate(distances) if index != source and distance >= 0]
        metrics.append({
            'degree': len(adjacency[source]),
            'reachableVertices': len(reachable),
            'hopDistanceSum': sum(reachable),
            'closeness': (len(reachable)/sum(reachable)) if reachable else None,
            'meanHopDistance': (sum(reachable)/len(reachable)) if reachable else None
        })
    # Undirected Brandes accumulates each unordered pair at both endpoints.
    betweenness=[value/2 for value in betweenness]
    normalizer=((vertex_count-1)*(vertex_count-2)/2) if vertex_count > 2 else 0
    normalized=[value/normalizer if normalizer else None for value in betweenness]
    reachable_counts=[item['reachableVertices'] for item in metrics]
    integration=[]
    for index,item in enumerate(metrics):
        reachable=item['reachableVertices']
        if reachable <= 1 or item['meanHopDistance'] is None:
            integration.append(None)
        else:
            # Hillier-Hanson real relative asymmetry and integration for the reachable component.
            relative_asymmetry=2*(item['meanHopDistance']-1)/(reachable-1)
            component_size=reachable+1
            diamond_normalizer=2*(component_size*(math.log2((component_size+2)/3)-1)+1)/((component_size-1)*(component_size-2))
            integration.append((diamond_normalizer/relative_asymmetry) if relative_asymmetry > 0 else None)
    return metrics, betweenness, normalized, integration

buildings=[]
for row in rows:
    double_polygons=row['projectedDouble']
    polygons=[polygon_from_rings(rings) for rings in row['polygons']]
    geometry=polygons[0] if len(polygons)==1 else MultiPolygon(polygons)
    assert geometry.is_valid and geometry.area > 0
    extent_coords=[coordinate for polygon in double_polygons for ring in polygon for coordinate in ring]
    double_bounds=[min(point[0] for point in extent_coords), min(point[1] for point in extent_coords), max(point[0] for point in extent_coords), max(point[1] for point in extent_coords)]
    buildings.append({'sourceId':row['sourceId'],'sourceRow':row['sourceRow'],'address':row['address'],'geometryType':row['geometryType'],'polygonCount':len(polygons),'holeCount':sum(len(polygon.interiors) for polygon in polygons),'valid':geometry.is_valid,'area':geometry.area,'float32Bounds':list(geometry.bounds),'doubleExtentBounds':double_bounds,'geometry':geometry})

assert [item['sourceId'] for item in buildings]==[342227,815550,815551]
assert [item['sourceRow'] for item in buildings]==[21,1118,1119]
assert [item['polygonCount'] for item in buildings]==[1,1,1]
assert [item['holeCount'] for item in buildings]==[0,0,0]

shared=[]
adjacency=[set() for _ in buildings]
for left in range(len(buildings)):
    for right in range(left+1,len(buildings)):
        intersection=buildings[left]['geometry'].boundary.intersection(buildings[right]['geometry'].boundary)
        line_parts=[]
        def collect_lines(geometry):
            if geometry.geom_type=='LineString' and geometry.length>0:
                line_parts.append(geometry)
            elif hasattr(geometry,'geoms'):
                for part in geometry.geoms: collect_lines(part)
        collect_lines(intersection)
        shared_length=sum(line.length for line in line_parts)
        shared.append({'leftSourceId':buildings[left]['sourceId'],'rightSourceId':buildings[right]['sourceId'],'sharedBoundaryLength':shared_length,'intersectionType':intersection.geom_type})
        if shared_length>0:
            adjacency[left].add(right); adjacency[right].add(left)
components=[]
unvisited=set(range(len(buildings)))
while unvisited:
    seed=min(unvisited); component=sorted(bfs(adjacency,seed)); unvisited.difference_update(component)
    components.append([buildings[index]['sourceId'] for index in component])

query_half_extents=sorted(set([6,12,18,30,20]+list(range(8,49,4))))
bounds_relations={}
for relation in ['intersects','within','contains']:
    cases=[]
    for half_extent in query_half_extents:
        query=box(-half_extent,-half_extent,half_extent,half_extent)
        selected=[]
        rows_for_case=[]
        for building in buildings:
            candidate=box(*building['float32Bounds'])
            match={'intersects':candidate.intersects(query),'within':candidate.within(query),'contains':candidate.contains(query)}[relation]
            rows_for_case.append({'sourceId':building['sourceId'],'matches':match,'bbox':building['float32Bounds']})
            if match: selected.append(building['sourceId'])
        cases.append({'halfExtent':half_extent,'queryBounds':[-half_extent,-half_extent,half_extent,half_extent],'perSource':rows_for_case,'selectedSourceIds':selected})
    bounds_relations[relation]=cases

group_union=unary_union([building['geometry'] for building in buildings])
minimum_x=min(building['doubleExtentBounds'][0] for building in buildings); minimum_y=min(building['doubleExtentBounds'][1] for building in buildings)
maximum_x=max(building['doubleExtentBounds'][2] for building in buildings); maximum_y=max(building['doubleExtentBounds'][3] for building in buildings)
probe_left=minimum_x-4; probe_right=maximum_x+4; probe_bottom=minimum_y-4; probe_top=maximum_y+4
probe_points=[]
for index in range(4):
    fraction=(index+0.5)/4
    x=probe_left+fraction*(probe_right-probe_left)
    probe_points.extend([[np.float32(x).item(),np.float32(probe_bottom).item()],[np.float32(x).item(),np.float32(probe_top).item()]])
for index in range(4):
    fraction=(index+0.5)/4
    y=probe_bottom+fraction*(probe_top-probe_bottom)
    probe_points.extend([[np.float32(probe_left).item(),np.float32(y).item()],[np.float32(probe_right).item(),np.float32(y).item()]])
obstacles=[building['geometry'] for building in buildings]
assert all(not group_union.covers(Point(point)) for point in probe_points)
visibility=[]
for maximum_distance in [25,40,55,100]:
    edges=[]
    adjacency_matrix=[[False]*len(probe_points) for _ in probe_points]
    for left in range(len(probe_points)):
        for right in range(left+1,len(probe_points)):
            first=Point(probe_points[left]); second=Point(probe_points[right])
            segment=LineString([probe_points[left],probe_points[right]])
            if first.distance(second) <= maximum_distance and not any(segment.intersects(obstacle) for obstacle in obstacles):
                edges.append([left,right]); adjacency_matrix[left][right]=True; adjacency_matrix[right][left]=True
    adjacency_list=[[neighbor for neighbor,connected in enumerate(row) if connected] for row in adjacency_matrix]
    metrics,choice,normalized_choice,integration=graph_metrics(adjacency_list)
    distances=[bfs(adjacency_list,index) for index in range(len(probe_points))]
    visibility.append({'maximumDistance':maximum_distance,'edges':edges,'denseAdjacency':adjacency_matrix,'vertices':[{'index':index,'position':point,'degree':metrics[index]['degree'],'reachableVertices':metrics[index]['reachableVertices'],'hopDistanceSum':metrics[index]['hopDistanceSum'],'closeness':metrics[index]['closeness'],'meanHopDistance':metrics[index]['meanHopDistance'],'choice':choice[index],'normalizedChoice':normalized_choice[index],'hillierHansonIntegration':integration[index]} for index,point in enumerate(probe_points)],'graphMetrics':{'degree': [item['degree'] for item in metrics],'reachableVertices':[item['reachableVertices'] for item in metrics],'hopDistanceSum':[item['hopDistanceSum'] for item in metrics],'closeness':[item['closeness'] for item in metrics],'meanHopDistance':[item['meanHopDistance'] for item in metrics],'choice':choice,'normalizedChoice':normalized_choice,'hillierHansonIntegration':integration}})

enclosure_bounds=[np.float32(minimum_x-18).item(),np.float32(minimum_y-18).item(),np.float32(maximum_x+18).item(),np.float32(maximum_y+18).item()]
enclosure=box(*enclosure_bounds)
buffer_cases=[]
inset_owners=[]
for building in buildings:
    inset=building['geometry'].buffer(-0.4,join_style='mitre',mitre_limit=2)
    inset_owners.append(inset)
    buffer_cases.append({'sourceId':building['sourceId'],'sourceArea':building['area'],'insetArea':inset.area,'insetIsEmpty':inset.is_empty,'insetIsValid':inset.is_valid,'insetGeometryType':inset.geom_type,'insetBounds':list(inset.bounds) if not inset.is_empty else None})

def unique_boundary_samples(geometry,spacing):
    lines=[]
    def collect(geom):
        if geom.geom_type=='LineString' and geom.length>0: lines.append(geom)
        elif hasattr(geom,'geoms'):
            for part in geom.geoms: collect(part)
    collect(geometry.boundary)
    samples=[]
    for line in lines:
        coordinates=list(line.coords)
        for start,end in zip(coordinates,coordinates[1:]):
            segment=LineString([start,end])
            count=math.ceil(segment.length/spacing)
            samples.extend([segment.interpolate(index/count,normalized=True).coords[0] for index in range(count)])
    return samples

sampling=[]
for inset_label,owners in [('source', [building['geometry'] for building in buildings]),('inset-0.4m',inset_owners)]:
    for spacing in [4,8,16]:
        owner_samples=[unique_boundary_samples(owner,spacing) if not owner.is_empty else [] for owner in owners]
        required_count=sum(len(samples) for samples in owner_samples)
        for capacity in [16,128]:
            samples=[]
            remaining=capacity
            # Match the documented sequential source-owner truncation used by bounded recipe input.
            for samples_for_owner in owner_samples:
                taken=samples_for_owner[:remaining]; samples.append(taken); remaining-=len(taken)
            sampling.append({'geometry':inset_label,'spacing':spacing,'capacity':capacity,'requiredSampleCount':required_count,'writtenSampleCount':min(required_count,capacity),'perOwnerRequiredCounts':[len(items) for items in owner_samples],'perOwnerWrittenCounts':[len(items) for items in samples]})

duplicate_samples=[]
for label,owners in [('source',[building['geometry'] for building in buildings]),('inset-0.4m',inset_owners)]:
    sites=[]
    for owner_index,owner in enumerate(owners):
        if owner.is_empty: continue
        for coordinate in unique_boundary_samples(owner,8): sites.append((owner_index,coordinate))
    duplicate_count=0
    for left in range(len(sites)):
        for right in range(left+1,len(sites)):
            if sites[left][0]!=sites[right][0] and Point(sites[left][1]).equals(Point(sites[right][1])):
                duplicate_count+=1
    duplicate_samples.append({'geometry':label,'spacing':8,'siteCount':len(sites),'crossOwnerDuplicateSitePairCount':duplicate_count})

# Optional GEOS ordered Voronoi cell construction and clipping to the analyst enclosure.
voronoi=[]
for spacing in [4,8,16]:
    owners=[]
    site_coordinates=[]
    for owner_index,owner in enumerate(inset_owners):
        coords=unique_boundary_samples(owner,spacing) if not owner.is_empty else []
        owners.append(coords); site_coordinates.extend(coords)
    sites=MultiPoint(site_coordinates)
    diagram=shapely.voronoi_polygons(sites,extend_to=enclosure,ordered=True)
    unioned=[None]*len(buildings)
    cell_areas=[0.0]*len(buildings)
    for cell_index,cell in enumerate(diagram.geoms):
        clipped=cell.intersection(enclosure)
        if clipped.is_empty: continue
        owner_index=next(index for index,coordinates in enumerate(owners) if any(Point(coordinate).equals(Point(site_coordinates[cell_index])) for coordinate in coordinates))
        unioned[owner_index]=clipped if unioned[owner_index] is None else unioned[owner_index].union(clipped)
        cell_areas[owner_index]+=clipped.area
    owner_geometries=[item if item is not None else Polygon() for item in unioned]
    overlap_area=sum(owner_geometries[left].intersection(owner_geometries[right]).area for left in range(len(owner_geometries)) for right in range(left+1,len(owner_geometries)))
    voronoi.append({'spacing':spacing,'siteCount':len(site_coordinates),'enclosureArea':enclosure.area,'ownerAreas':[{'sourceId':building['sourceId'],'unionArea':owner_geometries[index].area,'sumCellArea':cell_areas[index]} for index,building in enumerate(buildings)],'totalOwnerUnionArea':sum(item.area for item in owner_geometries),'overlapArea':overlap_area,'allOwnersCoveredEnclosure':unary_union(owner_geometries).symmetric_difference(enclosure).area < 1e-7})

result={'reference':'independent CPU geometry reference; no GPU values','versions':{'shapely':shapely.__version__,'geos':shapely.geos_version_string,'numpy':np.__version__},'coordinateFrame':{'origin':[-87.62887,41.89431],'units':'local display-frame metres','sourceCoordinates':'double projected for data extent; geometry/probes/enclosure rounded to float32 before GEOS','geosBufferNote':'GEOS negative mitre buffer with mitre limit 2; GPUBufferSurface can differ slightly from float32 geometry processing and vertex order'},'sourceBuildings':[{'sourceId':item['sourceId'],'sourceRow':item['sourceRow'],'address':item['address'],'sourceOriginalBuildingId':row['sourceOriginalBuildingId'],'sourceFootprint':row['sourceFootprint'],'sourceStatus':row['sourceStatus'],'geometryType':item['geometryType'],'polygonCount':item['polygonCount'],'holeCount':item['holeCount'],'valid':item['valid'],'area':item['area'],'float32Bounds':item['float32Bounds'],'doubleExtentBounds':item['doubleExtentBounds']} for item,row in zip(buildings,rows)],'sharedBoundaries':shared,'graphAdjacency':{'edges':[[buildings[left]['sourceId'],buildings[right]['sourceId']] for left in range(len(buildings)) for right in adjacency[left] if left<right],'components':components,'note':'Positive-length mapped boundary contact only; no inference about physical party walls.'},'boundsFilter':{'semantics':'closed AABB intersects/within/contains','relations':bounds_relations},'visibility':{'probes':probe_points,'probeCount':len(probe_points),'obstacles':'all closed source polygon exteriors; segment intersection blocks visibility','maximumDistanceGraphs':visibility,'metricDefinitions':{'choice':'undirected unnormalized Brandes betweenness; unordered endpoint pairs excluded','normalizedChoice':'choice divided by ((n-1)(n-2)/2)','closeness':'reachable vertex count divided by sum of hop distances','hillierHansonIntegration':'inverse real relative asymmetry within the reachable component; null for undefined/isolate'}},'enclosure':{'bounds':list(enclosure.bounds),'area':enclosure.area},'negativeBuffer':{'distance':-0.4,'joinStyle':'mitre','mitreLimit':2,'perBuilding':buffer_cases,'totalInsetArea':sum(item.area for item in inset_owners)},'boundarySampling':sampling,'duplicateSites':duplicate_samples,'voronoi':voronoi}
print(json.dumps(result,allow_nan=False))`;

const reference = JSON.parse(
  execFileSync(python, ['-c', pythonProgram], {input: JSON.stringify(inputFeatures), encoding: 'utf8'})
);
assert.deepEqual(reference.sourceBuildings.map(item => item.sourceId), sourceIds);
assert.deepEqual(reference.sourceBuildings.map(item => item.sourceRow), sourceRows);
assert.ok(reference.sourceBuildings.every(item => item.valid && item.area > 0));
assert.equal(reference.visibility.probeCount, 16);
await writeFile(outputFile, JSON.stringify(reference, null, 2) + '\n');
console.log(JSON.stringify({outputFile, versions: reference.versions, sourceBuildings: reference.sourceBuildings, sharedBoundaries: reference.sharedBoundaries, components: reference.graphAdjacency.components, visibilityEdges: reference.visibility.maximumDistanceGraphs.map(item => ({maximumDistance: item.maximumDistance, edgeCount: item.edges.length})), insetAreas: reference.negativeBuffer.perBuilding.map(item => ({sourceId: item.sourceId, area: item.insetArea})), voronoi: reference.voronoi}));
