// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Independent CPU queue BFS and Brandes oracle for small unweighted undirected fixtures. */
export function computeVisibilityMetricsOracle(neighbors: readonly (readonly number[])[]) {
  const count = neighbors.length;
  const betweenness = Array<number>(count).fill(0);
  const metrics: number[] = [];
  const maximumDepth: number[] = [];
  for (let source = 0; source < count; source++) {
    const distance = Array<number>(count).fill(-1);
    const paths = Array<number>(count).fill(0);
    const predecessors = Array.from({length: count}, () => [] as number[]);
    const dependency = Array<number>(count).fill(0);
    const queue = [source];
    distance[source] = 0;
    paths[source] = 1;
    for (let head = 0; head < queue.length; head++) {
      const vertex = queue[head];
      for (const neighbor of neighbors[vertex]) {
        if (distance[neighbor] < 0) {
          distance[neighbor] = distance[vertex] + 1;
          queue.push(neighbor);
        }
        if (distance[neighbor] === distance[vertex] + 1) {
          paths[neighbor] += paths[vertex];
          predecessors[neighbor].push(vertex);
        }
      }
    }
    for (let reverse = queue.length - 1; reverse >= 0; reverse--) {
      const vertex = queue[reverse];
      for (const predecessor of predecessors[vertex])
        dependency[predecessor] += (paths[predecessor] / paths[vertex]) * (1 + dependency[vertex]);
      if (vertex !== source) betweenness[vertex] += dependency[vertex] * 0.5;
    }
    const reached = queue.length - 1;
    const total = queue.reduce((sum, vertex) => sum + distance[vertex], 0);
    metrics.push(neighbors[source].length, reached, total, total > 0 ? reached / total : NaN);
    maximumDepth.push(Math.max(0, ...distance));
  }
  return {
    metrics,
    betweenness,
    maximumDepth,
    normalizedChoice: betweenness.map(value =>
      count > 2 ? (2 * value) / ((count - 1) * (count - 2)) : NaN
    )
  };
}

export const VISIBILITY_METRIC_GRAPH_FIXTURES = [
  {
    name: 'diamond',
    coordinates: [0, 0, 1, 1, 1, -1, 2, 0],
    maximumDistance: 1.5,
    neighbors: [
      [1, 2],
      [0, 3],
      [0, 3],
      [1, 2]
    ],
    expectedChoice: [0.5, 0.5, 0.5, 0.5]
  },
  {
    name: 'cycle',
    coordinates: [
      1,
      0,
      0.5,
      Math.sqrt(3) / 2,
      -0.5,
      Math.sqrt(3) / 2,
      -1,
      0,
      -0.5,
      -Math.sqrt(3) / 2,
      0.5,
      -Math.sqrt(3) / 2
    ],
    maximumDistance: 1.1,
    neighbors: [
      [1, 5],
      [0, 2],
      [1, 3],
      [2, 4],
      [3, 5],
      [0, 4]
    ],
    expectedChoice: [2, 2, 2, 2, 2, 2]
  },
  {
    name: 'complete',
    coordinates: [0, 0, 1, 0, 0, 1, 1, 1],
    maximumDistance: 2,
    neighbors: [
      [1, 2, 3],
      [0, 2, 3],
      [0, 1, 3],
      [0, 1, 2]
    ],
    expectedChoice: [0, 0, 0, 0]
  },
  {
    name: 'disconnected',
    coordinates: [0, 0, 1, 0, 2, 0, 10, 0, 11, 0, 20, 0],
    maximumDistance: 1.1,
    neighbors: [[1], [0, 2], [1], [4], [3], []],
    expectedChoice: [0, 1, 0, 0, 0, 0]
  }
] as const;
