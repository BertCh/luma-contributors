#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Write an independent exposed-face oracle for the WorldCover patch-metrics window."""

import json
from pathlib import Path

import rasterio

SOURCE = Path('examples/deck/spatial-analysis-showcase/public/data/dixie-fire/worldcover.png')
ORACLE = Path('dev-docs/research/showcase-overhaul-2026-10-25/earth-source-oracles.json')
SOURCE_ROW = 506
SOURCE_COLUMN = 555
GRID_SIZE = 64
CELL_SIZE_METERS = 20
GROUPS = {
    'tree': {'label': 'tree cover + mangroves', 'classes': {10, 95}},
    'woody': {'label': 'tree + shrub + mangroves', 'classes': {10, 20, 95}},
    'broad': {
        'label': 'woody + grass + crops + wetland + moss/lichen',
        'classes': {10, 20, 30, 40, 90, 95, 100},
    },
}


with rasterio.open(SOURCE) as source:
    classes = source.read(1)[
        SOURCE_ROW : SOURCE_ROW + GRID_SIZE,
        SOURCE_COLUMN : SOURCE_COLUMN + GRID_SIZE,
    ]

records = []
for group_id, group in GROUPS.items():
    selected = [[int(value) in group['classes'] for value in row] for row in classes]
    visited = [[False] * GRID_SIZE for _ in range(GRID_SIZE)]
    component_count = 0
    for row in range(GRID_SIZE):
        for column in range(GRID_SIZE):
            if not selected[row][column] or visited[row][column]:
                continue
            component_count += 1
            pending = [(row, column)]
            visited[row][column] = True
            while pending:
                current_row, current_column = pending.pop()
                for row_offset in (-1, 0, 1):
                    for column_offset in (-1, 0, 1):
                        if row_offset == 0 and column_offset == 0:
                            continue
                        neighbor_row = current_row + row_offset
                        neighbor_column = current_column + column_offset
                        if (
                            0 <= neighbor_row < GRID_SIZE
                            and 0 <= neighbor_column < GRID_SIZE
                            and selected[neighbor_row][neighbor_column]
                            and not visited[neighbor_row][neighbor_column]
                        ):
                            visited[neighbor_row][neighbor_column] = True
                            pending.append((neighbor_row, neighbor_column))

    outer_faces = 0
    interior_exposed_faces = 0
    selected_cells = sum(value for row in selected for value in row)
    for row in range(GRID_SIZE):
        for column in range(GRID_SIZE):
            if not selected[row][column]:
                continue
            for row_offset, column_offset in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor_row = row + row_offset
                neighbor_column = column + column_offset
                if not (0 <= neighbor_row < GRID_SIZE and 0 <= neighbor_column < GRID_SIZE):
                    outer_faces += 1
                elif not selected[neighbor_row][neighbor_column]:
                    interior_exposed_faces += 1

    for count_raster_border in (True, False):
        exposed_faces = interior_exposed_faces + (outer_faces if count_raster_border else 0)
        records.append(
            {
                'group': group_id,
                'label': group['label'],
                'countRasterBorder': count_raster_border,
                'selectedCells': selected_cells,
                'components8': component_count,
                'interiorExposedFaces': interior_exposed_faces,
                'outerExposedFaces': outer_faces,
                'exposedFaces': exposed_faces,
                'perimeterMeters': exposed_faces * CELL_SIZE_METERS,
            }
        )

document = json.loads(ORACLE.read_text())
document['habitatPatchPerimeter'] = {
    'method': (
        'Independent exposed-face count over the shipped ESA WorldCover crop at source row 506, '
        'column 555 (64×64, 20 m). Selected classes match the scene tree/woody/broad groups. '
        'Eight-neighbor flood fill independently checks component count. Each selected pixel '
        'contributes one face for a 4-neighbor outside the selected mask; raster-exterior faces '
        'are included only when countRasterBorder is true. No GPU output is used.'
    ),
    'sourceCrop': {
        'firstRow': SOURCE_ROW,
        'lastRow': SOURCE_ROW + GRID_SIZE - 1,
        'firstColumn': SOURCE_COLUMN,
        'lastColumn': SOURCE_COLUMN + GRID_SIZE - 1,
        'cellSizeMeters': CELL_SIZE_METERS,
    },
    'records': records,
}
ORACLE.write_text(json.dumps(document, indent=2) + '\n')
print(json.dumps({'records': len(records), 'sourceCrop': document['habitatPatchPerimeter']['sourceCrop']}, indent=2))
