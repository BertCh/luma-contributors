// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {KeySpaceId} from '../types';

/** CPU keys in one declared key space. */
export type KeySet = ReadonlySet<number>;

/** A declared mapping between key spaces, never inferred from geometry. */
export type KeyRelation = {
  id: string;
  from: KeySpaceId;
  to: KeySpaceId;
  cardinality: '1:1' | '1:N' | 'N:1' | 'N:M';
  kind: 'function' | 'buffer';
  timeDependent: boolean;
  available: boolean;
  /** Capture failure, when unavailable. */
  reason?: string;
  /** Missing capture evidence for a traversal. Check before using images or preimages. */
  getCoverageIssue?(
    keys: ReadonlySet<number>,
    direction: 'forward' | 'inverse'
  ): string | undefined;
  forward(keys: Iterable<number>): Set<number>;
  inverse(keys: Iterable<number>): Set<number>;
  preimageSize?(key: number): number;
};

/** One directed traversal of a relation. */
export type KeyRelationStep = {relation: KeyRelation; direction: 'forward' | 'inverse'};

/** Explicit distinction between missing evidence and disconnected key spaces. */
export type KeyMappingResult =
  | {status: 'ok'; keys: Set<number>}
  | {status: 'unknown'; reason: string}
  | {status: 'unlinked'};

/** Creates a function relation, scanning the declared domain when no inverse is supplied. */
export function createFunctionRelation(props: {
  id: string;
  from: KeySpaceId;
  to: KeySpaceId;
  cardinality: KeyRelation['cardinality'];
  map(key: number): number | readonly number[];
  inverseMap?(key: number): number | readonly number[];
  fromCount: number;
  timeDependent?: boolean;
  getCoverageIssue?: KeyRelation['getCoverageIssue'];
}): KeyRelation {
  const inDomain = (key: number) => Number.isInteger(key) && key >= 0 && key < props.fromCount;
  return {
    id: props.id,
    from: props.from,
    to: props.to,
    cardinality: props.cardinality,
    kind: 'function',
    timeDependent: props.timeDependent ?? false,
    available: true,
    getCoverageIssue: props.getCoverageIssue,
    forward(keys) {
      const requested = new Set(keys);
      requireMappingCoverage(props.getCoverageIssue?.(requested, 'forward'));
      const result = new Set<number>();
      for (const key of requested) {
        if (inDomain(key)) for (const target of asKeys(props.map(key))) result.add(target);
      }
      return result;
    },
    inverse(keys) {
      const result = new Set<number>();
      const targets = new Set(keys);
      requireMappingCoverage(props.getCoverageIssue?.(targets, 'inverse'));
      if (props.inverseMap) {
        for (const target of targets) {
          for (const key of asKeys(props.inverseMap(target))) if (inDomain(key)) result.add(key);
        }
      } else {
        for (let key = 0; key < props.fromCount; key++) {
          if (asKeys(props.map(key)).some(target => targets.has(target))) result.add(key);
        }
      }
      return result;
    }
  };
}

/** Creates an immutable CSR relation; its reverse index is built only on first use. */
export function createCSRRelation(props: {
  id: string;
  from: KeySpaceId;
  to: KeySpaceId;
  offsets: ArrayLike<number>;
  targets: ArrayLike<number>;
  toCount: number;
  /** Absolute source row of the first captured offset; defaults to zero. */
  firstKey?: number;
  /** Absolute target-list slot of the first captured target; defaults to zero. */
  targetFirstKey?: number;
  /** Full source domain size; omitted only when the supplied offsets are the complete domain. */
  fromCount?: number;
  timeDependent?: boolean;
}): KeyRelation {
  const firstKey = props.firstKey ?? 0;
  const targetFirstKey = props.targetFirstKey ?? 0;
  const fromCount = props.fromCount ?? firstKey + Math.max(0, props.offsets.length - 1);
  const getRowIssue = (row: number) => {
    if (row < 0 || row + 1 >= props.offsets.length)
      return 'relation source rows not fully captured';
    const first = props.offsets[row];
    const end = props.offsets[row + 1];
    if (!Number.isSafeInteger(first) || !Number.isSafeInteger(end) || first < 0 || end < first)
      return 'relation offsets invalid';
    // A captured empty row requires no target-list evidence.
    if (first !== end && (first < targetFirstKey || end > targetFirstKey + props.targets.length))
      return 'relation target span not fully captured';
    return undefined;
  };
  const getCoverageIssue: NonNullable<KeyRelation['getCoverageIssue']> = (keys, direction) => {
    if (!keys.size) return undefined;
    if (direction === 'inverse') {
      if (
        !Number.isSafeInteger(fromCount) ||
        firstKey !== 0 ||
        props.offsets.length < fromCount + 1
      )
        return 'relation inverse source coverage incomplete';
      for (let row = 0; row < fromCount; row++) {
        const issue = getRowIssue(row);
        if (issue) return issue;
      }
    } else {
      for (const key of keys) {
        if (!Number.isInteger(key) || key < 0 || key >= fromCount) continue;
        const issue = getRowIssue(key - firstKey);
        if (issue) return issue;
      }
    }
    return undefined;
  };
  const startOffset = (row: number) => Math.max(targetFirstKey, props.offsets[row]);
  const endOffset = (row: number) =>
    Math.min(targetFirstKey + props.targets.length, props.offsets[row + 1]);
  let reverse: Map<number, Set<number>> | undefined;
  const getReverse = () => {
    if (!reverse) {
      reverse = new Map();
      for (let row = 0; row + 1 < props.offsets.length; row++) {
        for (let offset = startOffset(row); offset < endOffset(row); offset++) {
          const target = props.targets[offset - targetFirstKey];
          if (target >= 0 && target < props.toCount) {
            const rows = reverse.get(target) ?? new Set<number>();
            rows.add(firstKey + row);
            reverse.set(target, rows);
          }
        }
      }
    }
    return reverse;
  };
  return {
    id: props.id,
    from: props.from,
    to: props.to,
    cardinality: 'N:M',
    kind: 'buffer',
    available: true,
    getCoverageIssue,
    timeDependent: props.timeDependent ?? false,
    forward(keys) {
      const requested = new Set(keys);
      requireMappingCoverage(getCoverageIssue(requested, 'forward'));
      const result = new Set<number>();
      for (const key of requested) {
        const row = key - firstKey;
        if (!Number.isInteger(row) || row < 0 || row + 1 >= props.offsets.length) continue;
        for (let offset = startOffset(row); offset < endOffset(row); offset++) {
          const target = props.targets[offset - targetFirstKey];
          if (target >= 0 && target < props.toCount) result.add(target);
        }
      }
      return result;
    },
    inverse(keys) {
      const requested = new Set(keys);
      requireMappingCoverage(getCoverageIssue(requested, 'inverse'));
      const result = new Set<number>();
      const index = getReverse();
      for (const key of requested) for (const row of index.get(key) ?? []) result.add(row);
      return result;
    },
    preimageSize: key => {
      requireMappingCoverage(getCoverageIssue(new Set([key]), 'inverse'));
      return getReverse().get(key)?.size ?? 0;
    }
  };
}

/** Represents an uncaptured time-dependent relation without guessing its mapping. */
export function createUnavailableRelation(props: {
  id: string;
  from: KeySpaceId;
  to: KeySpaceId;
  cardinality: KeyRelation['cardinality'];
  reason: string;
}): KeyRelation {
  const unavailable = (): Set<number> => {
    // This time-dependent mapping was not captured; callers must check available first.
    assert(false);
    return new Set();
  };
  return {
    ...props,
    kind: 'buffer',
    timeDependent: true,
    available: false,
    forward: unavailable,
    inverse: unavailable
  };
}

/** Bidirectional relation graph with deterministic shortest-path lookup. */
export class KeyRelationGraph {
  private relations = new Map<string, KeyRelation>();

  /** Adds or replaces the relation with this id. */
  add(relation: KeyRelation): void {
    this.relations.set(relation.id, relation);
  }

  /** Removes a relation by id (no-op when absent). */
  remove(relationId: string): void {
    this.relations.delete(relationId);
  }

  /** Every relation, in insertion order. */
  list(): KeyRelation[] {
    return [...this.relations.values()];
  }

  /** Finds a shortest path, including unavailable edges so missing captures remain visible. */
  findPath(from: KeySpaceId, to: KeySpaceId): KeyRelationStep[] | undefined {
    const queue: {keySpace: KeySpaceId; path: KeyRelationStep[]}[] = [{keySpace: from, path: []}];
    const visited = new Set([from]);
    for (let index = 0; index < queue.length; index++) {
      const {keySpace, path} = queue[index];
      if (keySpace === to) return path;
      for (const relation of this.relations.values()) {
        const direction = relation.from === keySpace ? 'forward' : 'inverse';
        if (relation.from !== keySpace && relation.to !== keySpace) continue;
        const next = direction === 'forward' ? relation.to : relation.from;
        if (visited.has(next)) continue;
        visited.add(next);
        queue.push({keySpace: next, path: [...path, {relation, direction}]});
      }
    }
    return undefined;
  }

  /** Maps keys along the shortest declared path. */
  mapKeys(keys: Iterable<number>, from: KeySpaceId, to: KeySpaceId): KeyMappingResult {
    const path = this.findPath(from, to);
    if (!path) return {status: 'unlinked'};
    let mapped = new Set(keys);
    for (const {relation, direction} of path) {
      if (!relation.available)
        return {
          status: 'unknown',
          reason: relation.reason ?? `Relation ${relation.id} was not captured`
        };
      const issue = relation.getCoverageIssue?.(mapped, direction);
      if (issue) return {status: 'unknown', reason: issue};
      mapped = relation[direction](mapped);
    }
    return {status: 'ok', keys: mapped};
  }
}

function asKeys(keys: number | readonly number[]): readonly number[] {
  return typeof keys === 'number' ? [keys] : keys;
}

function requireMappingCoverage(issue: string | undefined): void {
  if (issue) throw new Error(`Cannot map incomplete relation: ${issue}`);
}
