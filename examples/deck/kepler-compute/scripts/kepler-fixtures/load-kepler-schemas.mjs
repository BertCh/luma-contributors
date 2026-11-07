// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import Module, {createRequire} from 'node:module';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../../..');
const require = createRequire(path.join(repositoryRoot, 'package.json'));

/** Modules that need a GPU or browser at import time; schemas only touch layer class metadata. */
const STUBBED_PACKAGE_PATTERN = /^(@deck\.gl|@luma\.gl|@loaders\.gl|@math\.gl|@flowmap\.gl|@kepler\.gl\/deckgl-)/;

function createStub() {
  const target = function stub() {};
  const stub = new Proxy(target, {
    get: (_, key) => {
      if (key === '__esModule') return true;
      if (key === 'default') return stub;
      if (key === Symbol.toPrimitive) return () => '';
      if (key === 'prototype') return target.prototype;
      return stub;
    },
    apply: () => stub,
    construct: () => ({})
  });
  return stub;
}

/** Loads the installed Kepler CJS schema package in node, stubbing GPU/browser-only dependencies. */
export function loadKeplerSchemas() {
  const originalLoad = Module._load;
  const stub = createStub();
  Module._load = function loadModule(request, ...rest) {
    if (STUBBED_PACKAGE_PATTERN.test(request)) return stub;
    return originalLoad.call(this, request, ...rest);
  };
  try {
    return require('@kepler.gl/schemas');
  } finally {
    Module._load = originalLoad;
  }
}
