# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Direct ESMF Cartesian conservative fixtures, pinned ESMPy/ESMF 8.9.1, NumPy 2.2.6.
Create with conda-forge python=3.12 numpy=2.2.6 esmpy=8.9.1; no production imports.
Extensive source totals are converted to density before ESMF and integrated over destination area.
"""
import json
from pathlib import Path
import numpy as np
import esmpy as ESMF
assert ESMF.__version__ == '8.9.1'
assert np.__version__ == '2.2.6'


def grid(specification, validity=None):
    width, height = specification['width'], specification['height']
    result = ESMF.Grid(np.array([width, height], dtype=np.int32), coord_sys=ESMF.CoordSys.CART,
                       staggerloc=[ESMF.StaggerLoc.CENTER, ESMF.StaggerLoc.CORNER])
    for location, count, offset in [(ESMF.StaggerLoc.CENTER, 0, .5), (ESMF.StaggerLoc.CORNER, 1, 0)]:
        for dimension, length in enumerate([width, height]):
            coordinates = specification['origin'][dimension] + (np.arange(length+count)+offset)*specification['spacing'][dimension]
            result.get_coords(dimension, staggerloc=location)[...] = coordinates[:, None] if dimension == 0 else coordinates[None, :]
    if validity is not None:
        result.add_item(ESMF.GridItem.MASK)
        result.get_item(ESMF.GridItem.MASK)[...] = np.array(validity).reshape((width,height), order='F')
    return result

source = dict(width=4,height=3,origin=[0,0],spacing=[1.25,.75])
cases=[]
for name, destination, validity in [
    ('coarsen',dict(width=2,height=2,origin=[0,0],spacing=[2.5,1.125]),[1]*12),
    ('partial-mask',dict(width=4,height=3,origin=[-1.25,-.375],spacing=[1.875,1.125]),[1,0,1,1,1,1,0,1,1,1,1,1]),
    ('disjoint',dict(width=2,height=2,origin=[10,10],spacing=[1,1]),[1]*12)
]:
    source_grid=grid(source,validity); destination_grid=grid(destination)
    source_field=ESMF.Field(source_grid); destination_field=ESMF.Field(destination_grid)
    destination_fraction=ESMF.Field(destination_grid)
    source_area=np.prod(source['spacing']); destination_area=np.prod(destination['spacing'])
    for normalization,norm in [('destination-area',ESMF.NormType.DSTAREA),('covered-area',ESMF.NormType.FRACAREA),('extensive',ESMF.NormType.DSTAREA)]:
        route=ESMF.Regrid(source_field,destination_field,src_mask_values=np.array([0],dtype=np.int32),
                          regrid_method=ESMF.RegridMethod.CONSERVE,norm_type=norm,
                          unmapped_action=ESMF.UnmappedAction.IGNORE,dst_frac_field=destination_fraction)
        covered=destination_fraction.data.copy().flatten(order='F')*destination_area
        dates=[]
        for date in range(2):
            values=np.array([float((index*7+date*3)%17)-4 for index in range(12)])
            source_field.data[...]=(values/source_area if normalization=='extensive' else values).reshape((4,3),order='F')
            destination_field.data[...]=0
            route(source_field,destination_field,zero_region=ESMF.Region.SELECT)
            expected=destination_field.data.copy().flatten(order='F')
            if normalization=='extensive': expected*=destination_area
            dates.append(dict(values=values.tolist(),expected=[float(value) if area>0 else None for value,area in zip(expected,covered)]))
        cases.append(dict(name=name,source=source,destination=destination,validity=validity,normalization=normalization,coveredArea=covered.tolist(),dates=dates))
        route.destroy()
    destination_fraction.destroy(); source_field.destroy(); destination_field.destroy(); source_grid.destroy(); destination_grid.destroy()
output={'versions':{'esmpy':'8.9.1','numpy':np.__version__},'cases':cases}
path=Path(__file__).parent/'fixtures/raster-esmf.json'
path.write_text(json.dumps(output,indent=2)+'\n')
print(f'Wrote {len(cases)} cases with two dates each to {path}')
