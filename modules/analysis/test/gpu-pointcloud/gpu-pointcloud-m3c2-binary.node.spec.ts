// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {spawnSync} from 'node:child_process';
import {mkdtempSync, readFileSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {expect, it} from 'vitest';
import oracle from '../oracles/fixtures/pointcloud-m3c2-binary.json';

// Opt in with a PDAL build that provides filters.m3c2. An explicitly configured executable must
// succeed: a missing stage or executable fails instead of being converted into golden-fixture evidence.
const binary = process.env['PDAL_M3C2_BINARY'];
it.skipIf(!binary)(
  'executes the PDAL M3C2 binary from CSV inputs through output against NumPy planes',
  () => {
    const directory = mkdtempSync(join(tmpdir(), 'pc06-m3c2-'));
    try {
      const cloudNames = ['reference', 'comparison', 'cores'];
      const clouds = [oracle.reference, oracle.comparison, [oracle.core]];
      const readers = cloudNames.map((name, index) => {
        const filename = join(directory, `${name}.csv`);
        writeFileSync(filename, `X,Y,Z\n${clouds[index].map(row => row.join(',')).join('\n')}\n`);
        return {type: 'readers.text', filename, tag: name};
      });
      const output = join(directory, 'change.csv');
      const pipeline = join(directory, 'pipeline.json');
      writeFileSync(
        pipeline,
        JSON.stringify({
          pipeline: [
            ...readers,
            {
              type: 'filters.m3c2',
              inputs: cloudNames,
              normal_radius: oracle.normalRadius,
              cyl_radius: oracle.radius,
              cyl_halflen: oracle.halfDepth,
              orientation: 'up',
              min_points: 2
            },
            {
              type: 'writers.text',
              filename: output,
              format: 'csv',
              precision: 12,
              write_header: true
            }
          ]
        })
      );
      const result = spawnSync(binary!, ['pipeline', pipeline], {encoding: 'utf8', timeout: 60000});
      expect(result.error, result.stderr).toBeUndefined();
      expect(result.status, result.stderr).toBe(0);
      const lines = readFileSync(output, 'utf8').trim().split(/\r?\n/);
      expect(lines).toHaveLength(2);
      const [header, values] = lines;
      const names = header.split(',').map(name => name.replaceAll('"', '').toLowerCase());
      const row = values.split(',').map(Number);
      const getValue = (name: string) => {
        const column = names.indexOf(name);
        expect(column, `missing binary output ${name}`).toBeGreaterThanOrEqual(0);
        return row[column];
      };
      expect(getValue('m3c2_distance')).toBeCloseTo(oracle.expected.distance, 7);
      expect(getValue('m3c2_uncertainty')).toBeCloseTo(oracle.expected.uncertainty, 7);
      expect(getValue('m3c2_significant')).toBe(oracle.expected.significant);
      for (let epoch = 0; epoch < 2; epoch++) {
        expect(getValue(`m3c2_count${epoch + 1}`)).toBe(oracle.expected.counts[epoch]);
        expect(getValue(`m3c2_std_dev${epoch + 1}`)).toBeCloseTo(
          oracle.expected.populationRoughness[epoch],
          7
        );
      }
    } finally {
      rmSync(directory, {recursive: true, force: true});
    }
  }
);
