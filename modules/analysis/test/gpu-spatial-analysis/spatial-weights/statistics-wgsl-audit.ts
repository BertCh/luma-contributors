// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  Assign,
  Break,
  Call,
  CallExpr,
  Continue,
  For,
  Function as WGSLFunction,
  If,
  Let,
  Loop,
  Node as WGSLNode,
  Return,
  Var,
  VariableExpr,
  While
} from 'wgsl_reflect';
import {WgslParser} from 'wgsl_reflect';
import {
  findConstantNonFiniteBitcasts,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

/** Visit statement containers too: Node.search omits several of them. */
function visitNodes(
  value: unknown,
  callback: (node: WGSLNode, ancestors: readonly WGSLNode[]) => void
): void {
  const visited = new Set<WGSLNode>();
  const visit = (child: unknown, ancestors: readonly WGSLNode[] = []): void => {
    if (Array.isArray(child)) {
      for (const item of child) visit(item, ancestors);
    } else if (child instanceof WGSLNode && !visited.has(child)) {
      visited.add(child);
      callback(child, ancestors);
      for (const nested of Object.values(child)) visit(nested, [...ancestors, child]);
    }
  };
  visit(value);
}

/** Focused source checks; these supplement, rather than replace, real WebGPU compilation. */
export function findStatisticsNonFiniteConstants(source: string): string[] {
  const code = stripWGSLComments(source);
  const findings = findConstantNonFiniteBitcasts(code).map(finding => finding.detail);
  // `nan` is a legal variable name, used by existing kernels. Only flag undeclared tokens.
  const declared = new Set(
    [...code.matchAll(/\b(?:let|var|const|override|fn)\s*(?:<[^>]*>)?\s+(\w+)/g)].map(
      match => match[1]
    )
  );
  for (const match of code.matchAll(/\b(?:nan|inf|infinity)\b/gi))
    if (!declared.has(match[0])) findings.push(`nonfinite token ${match[0]}`);
  for (const match of code.matchAll(
    /(?<![\w.])(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[fh]?(?![\w.])/g
  )) {
    const value = Number(match[0].replace(/[fh]$/, ''));
    if (!Number.isFinite(value) || !Number.isFinite(Math.fround(value)))
      findings.push(`nonfinite float literal ${match[0]}`);
  }
  return findings;
}

/**
 * Catch varying returns, branches, loop bounds and exits around barriers, including calls to
 * barrier-containing helpers. This deliberately does not claim full WGSL uniformity analysis.
 */
export function findStatisticsBarrierRisks(source: string): string[] {
  // The parser mistakes these legal contextual names for keywords.
  const statements = new WgslParser().parse(
    source.replace(/\bblock\b/g, 'block_').replace(/(?<!<\s*)\bworkgroup\b(?!\s*>)/g, 'workgroup_')
  );
  const functions = statements.filter(
    (statement): statement is WGSLFunction => statement instanceof WGSLFunction
  );
  const barrierNames = new Set(['workgroupBarrier', 'storageBarrier']);
  for (let changed = true; changed; ) {
    changed = false;
    for (const declaration of functions) {
      visitNodes(declaration.body, node => {
        if (
          (node instanceof Call || node instanceof CallExpr) &&
          barrierNames.has(node.name) &&
          !barrierNames.has(declaration.name)
        ) {
          barrierNames.add(declaration.name);
          changed = true;
        }
      });
    }
  }
  const moduleNonuniform = statements
    .filter(
      (statement): statement is Var => statement instanceof Var && statement.storage !== 'uniform'
    )
    .map(statement => statement.name);
  const findings: string[] = [];
  for (const declaration of functions) {
    if (!barrierNames.has(declaration.name)) continue;
    const nonuniform = new Set(moduleNonuniform);
    for (const argument of declaration.args) {
      const builtin = argument.attributes?.find(attribute => attribute.name === 'builtin')?.value;
      if (builtin !== 'workgroup_id' && builtin !== 'num_workgroups') nonuniform.add(argument.name);
    }
    const dependencies = (expression: unknown): string[] => {
      const names = new Set<string>();
      visitNodes(expression, node => {
        if (node instanceof VariableExpr && nonuniform.has(node.name)) names.add(node.name);
      });
      return [...names];
    };
    // Track immutable aliases and local mutations to a fixed point. Constant induction variables
    // stay uniform; lane- or storage-derived bounds become nonuniform.
    for (let changed = true; changed; ) {
      changed = false;
      visitNodes(declaration.body, (node, ancestors) => {
        let name: string | undefined;
        let value: unknown;
        if (node instanceof Let || node instanceof Var) {
          name = node.name;
          value = node.value;
        }
        if (node instanceof Assign && node.variable instanceof VariableExpr) {
          name = node.variable.name;
          value = node.value;
        }
        if (
          name &&
          !nonuniform.has(name) &&
          (dependencies(value).length ||
            (node instanceof Assign &&
              ancestors.some(
                parent => parent instanceof If && dependencies(parent.condition).length
              )))
        ) {
          nonuniform.add(name);
          changed = true;
        }
      });
    }
    const hasBarrier = (value: unknown): boolean => {
      let found = false;
      visitNodes(value, node => {
        if ((node instanceof Call || node instanceof CallExpr) && barrierNames.has(node.name))
          found = true;
      });
      return found;
    };
    let lastBarrier = 0;
    visitNodes(declaration.body, node => {
      if ((node instanceof Call || node instanceof CallExpr) && barrierNames.has(node.name))
        lastBarrier = Math.max(lastBarrier, node.line);
    });
    visitNodes(declaration.body, (node, ancestors) => {
      if (node instanceof If) {
        let exits = false;
        const enclosingLoop = [...ancestors]
          .reverse()
          .find(
            parent => parent instanceof For || parent instanceof While || parent instanceof Loop
          );
        visitNodes([node.body, node.else, node.elseif], (child, nested) => {
          if (child instanceof Return) exits = true;
          // A break/continue in a completed inner loop reconverges before the later barrier.
          if (
            (child instanceof Break || child instanceof Continue) &&
            enclosingLoop &&
            hasBarrier(enclosingLoop) &&
            !nested.some(
              parent => parent instanceof For || parent instanceof While || parent instanceof Loop
            )
          )
            exits = true;
        });
        if (hasBarrier([node.body, node.else, node.elseif]) || (exits && node.line <= lastBarrier))
          for (const name of dependencies(node.condition))
            findings.push(`${declaration.name}:${node.line}: varying barrier branch/exit: ${name}`);
      }
      if ((node instanceof For || node instanceof While) && hasBarrier(node.body))
        for (const name of dependencies(node.condition))
          findings.push(`${declaration.name}:${node.line}: varying barrier loop: ${name}`);
    });
  }
  return findings;
}
