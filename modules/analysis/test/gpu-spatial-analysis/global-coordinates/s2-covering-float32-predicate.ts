// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Arithmetic transcription, not GPU qualification. Expected regions are analytic grid boxes. */
export class Float32CoveringPredicate {
  constructor(
    readonly reciprocal: boolean,
    readonly stableEdges: boolean,
    readonly integerWinding = false
  ) {}
  round = Math.fround;
  dot(first: number[], second: number[]): number {
    const round = this.round;
    return round(
      round(round(first[0] * second[0]) + round(first[1] * second[1])) + round(first[2] * second[2])
    );
  }
  cross(first: number[], second: number[]): number[] {
    const round = this.round;
    return [
      round(round(first[1] * second[2]) - round(first[2] * second[1])),
      round(round(first[2] * second[0]) - round(first[0] * second[2])),
      round(round(first[0] * second[1]) - round(first[1] * second[0]))
    ];
  }
  divide(first: number, second: number): number {
    return this.reciprocal
      ? this.round(first * this.round(1 / second))
      : this.round(first / second);
  }
  normalize(point: number[]): number[] {
    const length = this.round(Math.sqrt(this.dot(point, point)));
    return point.map(value => this.divide(value, length));
  }
  edgeCross(first: number[], second: number[]): number[] {
    if (!this.stableEdges) return this.cross(first, second);
    const sign = this.dot(first, second) >= 0 ? 1 : -1;
    return this.cross(
      first,
      second.map((value, axis) => this.round(value - sign * first[axis]))
    );
  }
  length(point: number[]): number {
    return this.round(Math.sqrt(this.dot(point, point)));
  }
  onArc(
    first: number[],
    second: number[],
    point: number[],
    normal = this.normalize(this.edgeCross(first, second))
  ): boolean {
    return (
      Math.abs(this.dot(normal, point)) <= 1e-6 &&
      this.dot(this.edgeCross(first, point), normal) >= -1e-6 &&
      this.dot(this.edgeCross(point, second), normal) >= -1e-6
    );
  }
  arcsIntersect(first: number[], second: number[], third: number[], fourth: number[]): boolean {
    const crossing = this.cross(this.edgeCross(first, second), this.edgeCross(third, fourth));
    if (this.length(crossing) > this.round(1e-6 * 1e-6)) {
      const point = this.normalize(crossing);
      const opposite = point.map(value => -value);
      return (
        (this.onArc(first, second, point) && this.onArc(third, fourth, point)) ||
        (this.onArc(first, second, opposite) && this.onArc(third, fourth, opposite))
      );
    }
    return this.onArc(first, second, third) || this.onArc(first, second, fourth);
  }
  contains(ring: number[][], query: number[]): [boolean, boolean] {
    const point = this.normalize(query);
    let winding = 0;
    let boundary = false;
    for (let edge = 0; edge < ring.length; edge++) {
      const first = ring[edge];
      const second = ring[(edge + 1) % ring.length];
      boundary ||= this.onArc(first, second, query);
      const tangent = (vertex: number[]) =>
        this.cross(
          point,
          vertex.map((value, axis) =>
            this.round(value - (this.dot(point, vertex) >= 0 ? point[axis] : -point[axis]))
          )
        );
      const firstTangent = tangent(first);
      const secondTangent = tangent(second);
      const numerator = this.dot(point, this.cross(firstTangent, secondTangent));
      const denominator = this.dot(firstTangent, secondTangent);
      if (this.integerWinding) {
        const magnitudes = point.map(Math.abs);
        const axis =
          magnitudes[0] <= magnitudes[1] && magnitudes[0] <= magnitudes[2]
            ? [1, 0, 0]
            : magnitudes[1] <= magnitudes[2]
              ? [0, 1, 0]
              : [0, 0, 1];
        const firstHeight = this.dot(firstTangent, axis);
        const secondHeight = this.dot(secondTangent, axis);
        if (firstHeight <= 0 && secondHeight > 0 && numerator > 0) winding++;
        if (firstHeight > 0 && secondHeight <= 0 && numerator < 0) winding--;
      } else if (Math.abs(numerator) + Math.abs(denominator) > 0)
        winding = this.round(
          winding + this.round(Math.atan2(numerator, denominator === 0 ? 0 : denominator))
        );
    }
    return [winding > (this.integerWinding ? 0 : Math.PI), boundary];
  }
  classify(ring: number[][], vertices: number[][]): [boolean, boolean] {
    const normals = vertices.map((first, edge) =>
      this.normalize(this.edgeCross(first, vertices[(edge + 1) % 4]))
    );
    let intersects = false;
    let contained = true;
    let touched = false;
    for (const point of ring)
      if (normals.every(normal => this.dot(normal, point) >= -1e-6)) {
        intersects = true;
        touched = true;
      }
    for (let edge = 0; edge < 4; edge++) {
      const first = vertices[edge];
      const second = vertices[(edge + 1) % 4];
      const [inside, boundary] = this.contains(ring, first);
      intersects ||= inside || boundary;
      contained &&= inside && !boundary;
      for (let vertex = 0; vertex < ring.length; vertex++)
        if (this.arcsIntersect(first, second, ring[vertex], ring[(vertex + 1) % ring.length])) {
          touched = true;
          intersects = true;
        }
    }
    return [intersects, contained && !touched];
  }
  validate(ring: number[][]): boolean {
    const center = this.normalize(
      ring.reduce(
        (sum, point) => sum.map((value, axis) => this.round(value + point[axis])),
        [0, 0, 0]
      )
    );
    for (let edge = 0; edge < ring.length; edge++) {
      const next = (edge + 1) % ring.length;
      if (
        this.dot(center, ring[edge]) <= 1e-6 ||
        this.length(this.edgeCross(ring[edge], ring[next])) <= 1e-6
      )
        return false;
      for (let other = edge + 1; other < ring.length; other++) {
        const otherNext = (other + 1) % ring.length;
        if (other === next || otherNext === edge) continue;
        if (this.arcsIntersect(ring[edge], ring[next], ring[other], ring[otherNext])) return false;
      }
    }
    return true;
  }
  cellVertices(first: number, second: number): number[][] {
    const surface = (coordinate: number) => {
      const value = this.round(coordinate / 32768);
      return this.divide(this.round(this.round(this.round(4 * value) * value) - 1), 3);
    };
    return [
      [first, second],
      [first + 1, second],
      [first + 1, second + 1],
      [first, second + 1]
    ].map(([horizontal, vertical]) => this.normalize([1, surface(horizontal), surface(vertical)]));
  }
}

/** Binary64 construction followed by f32 packing, matching the input buffer rather than cell WGSL. */
export function createFineCoveringRectangle(first: number, second: number): number[][] {
  const surface = (coordinate: number) => (4 * (coordinate / 32768) ** 2 - 1) / 3;
  return [
    [first - 0.4, second - 0.4],
    [first + 1.4, second - 0.4],
    [first + 1.4, second + 1.4],
    [first - 0.4, second + 1.4]
  ].map(([horizontal, vertical]) => {
    const point = [1, surface(horizontal), surface(vertical)];
    return point.map(value => Math.fround(value / Math.hypot(...point)));
  });
}
