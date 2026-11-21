// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Extracts the user-facing operations of @luma-contributors/analysis into
// src/catalog/generated/inventory.json, using the TypeScript compiler API.
//
// Usage (from the website directory, Node 22.6+):
//   node --experimental-strip-types scripts/extract-inventory.ts
//
// An export is user-facing when it is a non-abstract class named GPU*, a function named
// add*Recipe / add*ToGraph / create* / make*, or any class or function listed as an operator in the
// spatial-analysis ontology. Analysis-core internals (kernels, contracts, graph support,
// ring assembly), low-level helpers, WGSL builders, types and constants are skipped.
// Eligible declarations without qualified metadata fail extraction after actual re-exports are
// deduplicated. Function overloads are retained, including literal mode discriminants.
//
// Operation forms are skipped too: `GPU<Name>Operation` in a `gpu-<name>-operation.ts` file, its
// CPU twin in `cpu-<name>-operation.ts`, and `analysis-core/operations` (contributor contract,
// section 14). Each is another way to run a contributor that already has its own entry, as a
// luma.gl GPUGraph operation, not a further operation.

import {existsSync, readdirSync, readFileSync, statSync, writeFileSync, mkdirSync} from 'node:fs';
import {dirname, join, relative, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import ts from 'typescript';
import {build} from 'esbuild';
import type {CatalogAnalysisMetadata} from '../src/catalog/analysis-metadata';
import type {GPUAnalysisOperator} from '../../modules/analysis/src/gpu-spatial-analysis/ontology/index';

// The extracted operations use domain verbs rather than create/make. Their word arithmetic,
// selection-expression builders and upload helpers remain implementation building blocks.
const WIDE_NUMERIC_HELPER_FILE_PATTERN =
  /\/column-profile\/(?:gpu-wide-numeric-scan|wide-numeric-gpu-|numeric-narrowing-diagnostics)/;

const websiteDirectory = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const repositoryRoot = resolve(websiteDirectory, '..');
const analysisSource = join(repositoryRoot, 'modules/analysis/src');
const packageName = '@luma-contributors/analysis';

export type InventoryProp = {name: string; type: string; optional: boolean; jsdoc: string};
export type InventoryContract = {
  propsType: string | null;
  props: InventoryProp[];
  outputs: InventoryProp[];
  mode?: {values: string[]; optional: boolean};
};
type InventoryOntology = {
  capability: string;
  domain: string;
  stage: string;
  inputs: string[];
  outputs: string[];
};
type InventorySymbol = CatalogAnalysisMetadata & {
  symbol: string;
  kind: 'class' | 'function' | 'recipe';
  importPath: string;
  sourceFile: string;
  directory: string;
  propsType: string | null;
  props: InventoryProp[];
  /** Resolved output fields: the class \`output\`/\`outputs\` prop type, or the function return type. */
  outputs: InventoryProp[];
  overloads?: InventoryContract[];
  summary: string;
  docsSlug: string | null;
  lensScenarios: string[];
  ontology: InventoryOntology | null;
};

const FUNCTION_NAME_PATTERN = /^(add\w+Recipe|add\w+ToGraph|create[A-Z]\w*|make[A-Z]\w*)$/;
const LOW_LEVEL_PATTERN = /(Wgsl|WGSL|Shader|Kernel|Descriptor|Annotation|Ref$|Validator|Hash)/;
const OPERATION_FORM_FILE_PATTERN =
  /\/analysis-core\/operations\/|\/(gpu|cpu)-[a-z0-9-]+-operation\.ts$/;
const MAX_TYPE_LENGTH = 60;
const MAX_JSDOC_LENGTH = 70;
const MAX_LENS_SCENARIOS = 3;
const MAX_CONTENT_LENS_MATCHES = 3;
const RENAMED_OPERATOR_SCENARIOS = new Map([
  ['GPUDBSCAN', {recordedName: 'GPUSpatialClustering', id: 'gpu-spatial-analysis.spatial-clustering'}],
  [
    'GPUHDBSCAN',
    {
      recordedName: 'GPUHierarchicalDensityClustering',
      id: 'gpu-spatial-analysis.hierarchical-density-clustering'
    }
  ],
  ['GPUKeyLookup', {recordedName: 'GPUKeyJoin', id: 'gpu-dataframe.key-join'}],
  [
    'GPUPycnophylacticInterpolation',
    {recordedName: 'GPUPycnophylactic', id: 'gpu-spatial-analysis.pycnophylactic'}
  ]
]);

function listEntryPoints(): {importPath: string; file: string}[] {
  const packageJson = JSON.parse(
    readFileSync(join(repositoryRoot, 'modules/analysis/package.json'), 'utf8')
  );
  const entryPoints: {importPath: string; file: string}[] = [];
  for (const exportKey of Object.keys(packageJson.exports)) {
    const importPath = exportKey === '.' ? packageName : `${packageName}/${exportKey.slice(2)}`;
    const directory = exportKey === '.' ? '' : exportKey.slice(2);
    const file = join(analysisSource, directory, 'index.ts');
    if (existsSync(file)) entryPoints.push({importPath, file});
  }
  return entryPoints;
}

function listFilesRecursive(
  directory: string,
  extension: string,
  skipDirectories: string[] = []
): string[] {
  const files: string[] = [];
  for (const name of readdirSync(directory)) {
    const path = join(directory, name);
    if (statSync(path).isDirectory()) {
      if (!skipDirectories.includes(name))
        files.push(...listFilesRecursive(path, extension, skipDirectories));
    } else if (name.endsWith(extension)) {
      files.push(path);
    }
  }
  return files;
}

function summarizeText(text: string, maxLength: number): string {
  const firstSentence =
    text
      .replace(/\s+/g, ' ')
      .trim()
      .split(/(?<=\.)\s/)[0] ?? '';
  return firstSentence.length > maxLength
    ? `${firstSentence.slice(0, maxLength - 1)}…`
    : firstSentence;
}

async function readAnalysisLibrary(): Promise<
  typeof import('../src/catalog/analysis-metadata') & {
    getGPUAnalysisCapabilityCatalog: typeof import('../../modules/analysis/src/gpu-spatial-analysis/ontology/index')['getGPUAnalysisCapabilityCatalog'];
    getGPUAnalysisOperatorCatalog: typeof import('../../modules/analysis/src/gpu-spatial-analysis/ontology/index')['getGPUAnalysisOperatorCatalog'];
  }
> {
  const bundled = await build({
    stdin: {
      contents: `export * from './src/catalog/analysis-metadata.ts';
        export {getGPUAnalysisOperatorCatalog, getGPUAnalysisCapabilityCatalog} from '../modules/analysis/src/gpu-spatial-analysis/ontology/index.ts';`,
      resolveDir: websiteDirectory
    },
    bundle: true,
    platform: 'node',
    format: 'esm',
    write: false
  });
  return import(
    `data:text/javascript;base64,${Buffer.from(bundled.outputFiles[0].text).toString('base64')}`
  );
}

function readDocumentationPages(): {slugs: string[]; pages: {slug: string; text: string}[]} {
  const docsRoot = join(repositoryRoot, 'docs');
  const pages = listFilesRecursive(join(docsRoot, 'api'), '.md').map(file => {
    let slug = relative(docsRoot, file).replace(/\.md$/, '');
    if (slug.endsWith('/README')) slug = slug.slice(0, -'/README'.length);
    return {slug, text: readFileSync(file, 'utf8')};
  });
  return {slugs: pages.map(page => page.slug).sort(), pages};
}

function readLensScenarios(): {ids: string[]; records: {id: string; text: string}[]} {
  const directory = join(repositoryRoot, 'docs/lens-records');
  const records = readdirSync(directory)
    .filter(name => name.endsWith('.json') && statSync(join(directory, name)).isFile())
    .map(name => ({
      id: name.replace(/\.json$/, ''),
      text: readFileSync(join(directory, name), 'utf8')
    }));
  return {ids: records.map(record => record.id).sort(), records};
}

/** Resolves the object contract behind a generic constructor parameter. */
export function resolveInventoryParameterType(checker: ts.TypeChecker, type: ts.Type): ts.Type {
  const visited = new Set<ts.Type>();
  while ((type.flags & ts.TypeFlags.TypeParameter) !== 0 && !visited.has(type)) {
    visited.add(type);
    const constraint = checker.getBaseConstraintOfType(type);
    if (!constraint) break;
    type = constraint;
  }
  return checker.getNonNullableType(type);
}

export async function extractInventory(operatorsOverride?: readonly GPUAnalysisOperator[]) {
  const entryPoints = listEntryPoints();
  const configFile = ts.readConfigFile(join(repositoryRoot, 'tsconfig.json'), ts.sys.readFile);
  const parsed = ts.parseJsonConfigFileContent(configFile.config, ts.sys, repositoryRoot);
  const program = ts.createProgram({
    rootNames: entryPoints.map(entryPoint => entryPoint.file),
    options: {...parsed.options, noEmit: true, plugins: undefined}
  });
  const checker = program.getTypeChecker();
  const library = await readAnalysisLibrary();
  const operators = operatorsOverride ?? library.getGPUAnalysisOperatorCatalog();
  const operatorByIdentity = new Map(
    operators.map(operator => [`${operator.package}:${operator.exportName}`, operator])
  );
  const ambiguousNames = new Set(
    operators
      .filter(operator =>
        operators.some(
          other => other.exportName === operator.exportName && other.package !== operator.package
        )
      )
      .map(operator => operator.exportName)
  );
  const capabilities = new Map(
    library.getGPUAnalysisCapabilityCatalog().map(capability => [capability.id, capability])
  );
  const ontologyByOperator = new Map(
    operators.map(operator => [
      `${operator.package}:${operator.exportName}`,
      {
        capability: operator.primaryCapability,
        domain: capabilities.get(operator.primaryCapability)?.domain ?? operator.package,
        stage:
          capabilities.get(operator.primaryCapability)?.stage ?? operator.role ?? 'unspecified',
        inputs: [
          ...library.getCatalogAnalysisMetadata(operator.exportName, operator.package).dataKinds
            .input
        ],
        outputs: [
          ...library.getCatalogAnalysisMetadata(operator.exportName, operator.package).dataKinds
            .output
        ]
      }
    ])
  );
  const documentation = readDocumentationPages();
  const lensScenarios = readLensScenarios();

  const formatType = (type: ts.Type): string => {
    const text = checker
      .typeToString(
        type,
        undefined,
        ts.TypeFormatFlags.NoTruncation | ts.TypeFormatFlags.UseAliasDefinedOutsideCurrentScope
      )
      .replace(/\s+/g, ' ');
    return text.length > MAX_TYPE_LENGTH ? `${text.slice(0, MAX_TYPE_LENGTH - 1)}…` : text;
  };

  const resolveProps = (
    parameter: ts.Symbol | undefined,
    location: ts.Node
  ): Pick<InventoryContract, 'propsType' | 'props' | 'mode'> => {
    if (!parameter) return {propsType: null, props: []};
    const declaredType = checker.getTypeOfSymbolAtLocation(parameter, location);
    const parameterType = resolveInventoryParameterType(checker, declaredType);
    const declaration = parameter.valueDeclaration;
    const typeNode = declaration && ts.isParameter(declaration) ? declaration.type : undefined;
    const propsType = typeNode && (declaredType.flags & ts.TypeFlags.TypeParameter) === 0
      ? typeNode.getText().replace(/\s+/g, ' ').slice(0, MAX_TYPE_LENGTH)
      : formatType(parameterType);
    const props: InventoryProp[] = [];
    let mode: InventoryContract['mode'];
    // Unions (for example discriminated option sets) contribute every member's props; a prop that
    // is missing from some member is reported as optional.
    const members = parameterType.isUnion() ? parameterType.types : [parameterType];
    const merged = new Map<string, {property: ts.Symbol; count: number; optional: boolean}>();
    for (const member of members) {
      if (!(member.flags & ts.TypeFlags.Object || member.isIntersection())) continue;
      for (const property of checker.getPropertiesOfType(member)) {
        if (property.name.startsWith('__@')) continue;
        const propertyFile = property.declarations?.[0]?.getSourceFile().fileName ?? '';
        if (propertyFile.includes('node_modules')) continue;
        const entry = merged.get(property.name);
        if (entry) {
          entry.count += 1;
          entry.optional ||= (property.flags & ts.SymbolFlags.Optional) !== 0;
        } else {
          merged.set(property.name, {
            property,
            count: 1,
            optional: (property.flags & ts.SymbolFlags.Optional) !== 0
          });
        }
      }
    }
    for (const {property, count, optional} of merged.values()) {
      // A union's first arm may explicitly forbid this port with `never`; read the combined
      // symbol so the catalog retains the usable type and documentation from the other arm.
      const combinedProperty = checker.getPropertyOfType(parameterType, property.name) ?? property;
      const propertyType = checker.getTypeOfSymbolAtLocation(combinedProperty, location);
      const isOptional = optional || count < members.length;
      if (property.name === 'mode') {
        const nonNullableType = checker.getNonNullableType(propertyType);
        const modeTypes = nonNullableType.isUnion() ? nonNullableType.types : [nonNullableType];
        if (modeTypes.every(modeType => modeType.isStringLiteral()))
          mode = {values: modeTypes.map(modeType => modeType.value), optional: isOptional};
      }
      props.push({
        name: property.name,
        type: formatType(propertyType),
        optional: isOptional,
        jsdoc: summarizeText(
          ts.displayPartsToString(combinedProperty.getDocumentationComment(checker)),
          MAX_JSDOC_LENGTH
        )
      });
    }
    return {propsType, props, ...(mode ? {mode} : {})};
  };

  const listObjectFields = (type: ts.Type, location: ts.Node): InventoryProp[] => {
    const fields: InventoryProp[] = [];
    const members = type.isUnion() ? type.types : [type];
    const seen = new Set<string>();
    for (const member of members) {
      if (!(member.flags & ts.TypeFlags.Object || member.isIntersection())) continue;
      if (checker.getSignaturesOfType(member, ts.SignatureKind.Call).length > 0) continue;
      for (const property of checker.getPropertiesOfType(member)) {
        if (property.name.startsWith('__@') || seen.has(property.name)) continue;
        const propertyFile = property.declarations?.[0]?.getSourceFile().fileName ?? '';
        if (propertyFile.includes('node_modules') || propertyFile.includes('lib.')) continue;
        seen.add(property.name);
        fields.push({
          name: property.name,
          type: formatType(checker.getTypeOfSymbolAtLocation(property, location)),
          optional: (property.flags & ts.SymbolFlags.Optional) !== 0,
          jsdoc: summarizeText(
            ts.displayPartsToString(property.getDocumentationComment(checker)),
            MAX_JSDOC_LENGTH
          )
        });
      }
    }
    return fields;
  };

  // Functions whose options are not a single props object: report every parameter by name and
  // flatten object-typed parameters as dotted names such as \`options.scale\`.
  const resolveParameterProps = (signature: ts.Signature, location: ts.Node): InventoryProp[] => {
    const props: InventoryProp[] = [];
    for (const parameter of signature.getParameters()) {
      if (/^(device|graph|commandGraph)$/.test(parameter.name)) continue;
      const declaration = parameter.valueDeclaration;
      const optional =
        !!declaration &&
        ts.isParameter(declaration) &&
        (!!declaration.questionToken || !!declaration.initializer);
      const parameterType = checker.getNonNullableType(
        checker.getTypeOfSymbolAtLocation(parameter, location)
      );
      props.push({name: parameter.name, type: formatType(parameterType), optional, jsdoc: ''});
      const isPlainObject =
        (parameterType.flags & ts.TypeFlags.Object) !== 0 &&
        checker.getSignaturesOfType(parameterType, ts.SignatureKind.Call).length === 0 &&
        parameterType.symbol?.name !== 'Float32Array' &&
        !checker.isArrayLikeType(parameterType);
      if (!isPlainObject) continue;
      for (const field of resolveProps(parameter, location).props)
        props.push({...field, name: `${parameter.name}.${field.name}`});
    }
    return props;
  };

  const resolveOutputs = (
    isClass: boolean,
    props: InventoryProp[],
    parameter: ts.Symbol | undefined,
    signature: ts.Signature | undefined,
    location: ts.Node
  ): InventoryProp[] => {
    if (isClass) {
      const outputProp =
        (parameter &&
          checker.getPropertyOfType(
            checker.getNonNullableType(checker.getTypeOfSymbolAtLocation(parameter, location)),
            'outputs'
          )) ||
        (parameter &&
          checker.getPropertyOfType(
            checker.getNonNullableType(checker.getTypeOfSymbolAtLocation(parameter, location)),
            'output'
          ));
      return outputProp
        ? listObjectFields(
            checker.getNonNullableType(checker.getTypeOfSymbolAtLocation(outputProp, location)),
            location
          )
        : [];
    }
    if (!signature) return [];
    let returnType = checker.getReturnTypeOfSignature(signature);
    if (returnType.symbol?.name === 'Promise')
      returnType = checker.getTypeArguments(returnType as ts.TypeReference)[0] ?? returnType;
    return listObjectFields(returnType, location);
  };

  const symbols = new Map<string, Omit<InventorySymbol, keyof CatalogAnalysisMetadata>>();
  for (const entryPoint of entryPoints) {
    const sourceFile = program.getSourceFile(entryPoint.file);
    const moduleSymbol = sourceFile && checker.getSymbolAtLocation(sourceFile);
    if (!moduleSymbol) continue;
    for (const exported of checker.getExportsOfModule(moduleSymbol)) {
      const name = exported.name;
      const target =
        exported.flags & ts.SymbolFlags.Alias ? checker.getAliasedSymbol(exported) : exported;
      const declaration = target.declarations?.find(
        candidate => ts.isClassDeclaration(candidate) || ts.isFunctionDeclaration(candidate)
      );
      if (!declaration) continue;
      const declarationFile = declaration.getSourceFile().fileName;
      if (!declarationFile.startsWith(analysisSource) || declarationFile.includes('/testing/'))
        continue;
      const isClass = ts.isClassDeclaration(declaration);
      const identity = `${entryPoint.importPath}:${name}`;
      const isOperator = ontologyByOperator.has(identity);
      const isAbstract =
        isClass && (ts.getCombinedModifierFlags(declaration) & ts.ModifierFlags.Abstract) !== 0;
      const included = isClass
        ? !isAbstract && (/^GPU[A-Z]/.test(name) || isOperator)
        : (FUNCTION_NAME_PATTERN.test(name) && !LOW_LEVEL_PATTERN.test(name)) || isOperator;
      const isCoreInternal =
        /\/analysis-core\/(kernels|graph-support|contracts|ring-assembly)\//.test(declarationFile);
      // Exports tagged @internal in their JSDoc (for example the pointcloud grid-index builder that
      // GPUPointCloudICP, GPUPointCloudSurfaces and GPUPointCloudNeighborhoods share) are building
      // blocks of other operations, not operations a reader would call, so they are not catalog symbols.
      const isMarkedInternal = ts
        .getJSDocTags(declaration)
        .some(tag => tag.tagName.text === 'internal');
      // GPUGraph operation forms and CPU twins of contributors that have their own entry.
      const isOperationForm = OPERATION_FORM_FILE_PATTERN.test(declarationFile);
      if (WIDE_NUMERIC_HELPER_FILE_PATTERN.test(declarationFile) && !isOperator) continue;
      if (!included || (isCoreInternal && !isOperator) || isMarkedInternal || isOperationForm)
        continue;

      // Use the resolved export's public signatures. TypeScript excludes the implementation of an
      // overloaded function, while retaining any declared overload that accepts a union.
      const targetType = checker.getTypeOfSymbolAtLocation(target, declaration);
      const signatures = isClass
        ? targetType.getConstructSignatures()
        : targetType.getCallSignatures();
      const contracts: InventoryContract[] = signatures.map(signature => {
        // Skip leading infrastructure parameters (device, graph) to reach the options object.
        const parameter = signature.getParameters().find(candidate => {
          const typeText = checker.typeToString(
            checker.getTypeOfSymbolAtLocation(candidate, declaration)
          );
          return (
            !/^(device|graph|commandGraph)$/.test(candidate.name) &&
            !/^(Device|GPUCommandGraph)\b/.test(typeText)
          );
        });
        const {propsType, props: resolvedProps, mode} = resolveProps(parameter, declaration);
        const props =
          resolvedProps.length > 0 || isClass
            ? resolvedProps
            : resolveParameterProps(signature, declaration);
        const outputs = resolveOutputs(isClass, props, parameter, signature, declaration);
        return {
          propsType,
          props,
          outputs,
          ...(mode ? {mode} : {})
        };
      });
      const contract = contracts.find(contract => contract.mode?.optional) ??
        contracts[0] ?? {propsType: null, props: [], outputs: []};

      const importPath = entryPoint.importPath;
      // Re-exports of one declaration share a record; distinct same-name declarations do not.
      const declarationIdentity = `${declarationFile}:${declaration.pos}:${name}`;
      const existing = symbols.get(declarationIdentity);
      const existingIsOperator =
        existing && ontologyByOperator.has(`${existing.importPath}:${name}`);
      if (
        existing &&
        ((existingIsOperator && !isOperator) ||
          (Boolean(existingIsOperator) === isOperator &&
            existing.importPath.length >= importPath.length))
      )
        continue;

      const sourcePath = relative(repositoryRoot, declarationFile);
      const kind: InventorySymbol['kind'] = isClass
        ? 'class'
        : /Recipe$|ToGraph$/.test(name)
          ? 'recipe'
          : 'function';
      const pageScores = documentation.pages
        .map(page => ({
          page,
          count: (page.text.match(new RegExp(`\\b${name}\\b`, 'g')) ?? []).length
        }))
        .filter(({count}) => count > 0)
        .map(({page, count}) => ({
          slug: page.slug,
          score: count - (/catalog|cross-reference|^api$/.test(page.slug) ? 1000 : 0)
        }))
        .sort((left, right) => right.score - left.score || left.slug.localeCompare(right.slug));
      const kebabName = name
        .replace(/^(GPU|add|create|make)/, '')
        .replace(/(Recipe|ToGraph)$/, '')
        .replace(/([a-z0-9])([A-Z])/g, '$1-$2')
        .replace(/([A-Z]+)([A-Z][a-z])/g, '$1-$2')
        .toLowerCase();
      const scenarioName = (id: string): string => id.slice(id.indexOf('.') + 1);
      const qualifiedLensRecords = ambiguousNames.has(name)
        ? lensScenarios.records.filter(record =>
            record.id.startsWith(`${importPath.split('/').at(-1)}.`)
          )
        : lensScenarios.records;
      const archivedScenario = RENAMED_OPERATOR_SCENARIOS.get(name);
      const nameMatches = qualifiedLensRecords
        .filter(
          record =>
            record.id === archivedScenario?.id ||
            [kebabName, `${kebabName}-recipe`].includes(scenarioName(record.id))
        )
        .map(record => record.id);
      const nearMatches = qualifiedLensRecords
        .filter(
          record => scenarioName(record.id).includes(kebabName) && !nameMatches.includes(record.id)
        )
        .map(record => record.id);
      // Content references are only trusted when specific: a symbol named in many records is noise.
      const wordPattern = new RegExp(
        archivedScenario ? `\\b(?:${name}|${archivedScenario.recordedName})\\b` : `\\b${name}\\b`
      );
      const contentMatches = qualifiedLensRecords
        .filter(
          record =>
            wordPattern.test(record.text) &&
            !nameMatches.includes(record.id) &&
            !nearMatches.includes(record.id)
        )
        .map(record => record.id);
      const lensMatches = [
        ...nameMatches,
        ...nearMatches,
        ...(contentMatches.length <= MAX_CONTENT_LENS_MATCHES ? contentMatches : [])
      ];
      symbols.set(declarationIdentity, {
        symbol: name,
        kind,
        importPath,
        sourceFile: sourcePath,
        directory: relative(analysisSource, dirname(declarationFile))
          .split('/')
          .slice(0, 2)
          .join('/'),
        propsType: contract.propsType,
        props: contract.props,
        outputs: contract.outputs,
        ...(contracts.length > 1 ? {overloads: contracts} : {}),
        summary: summarizeText(
          ts.displayPartsToString(target.getDocumentationComment(checker)),
          MAX_JSDOC_LENGTH * 2
        ),
        docsSlug: (() => {
          const declared = operatorByIdentity
            .get(identity)
            ?.evidence.documentation.replace(/^docs\//, '')
            .replace(/\.md$/, '')
            .replace(/\/README$/, '');
          return declared && documentation.slugs.includes(declared)
            ? declared
            : (pageScores[0]?.slug ?? null);
        })(),
        lensScenarios: lensMatches.slice(0, MAX_LENS_SCENARIOS),
        ontology: ontologyByOperator.get(identity) ?? null
      });
    }
  }

  const missing: string[] = [];
  const sorted: InventorySymbol[] = [...symbols.values()]
    .map(symbol => {
      const identity = `${symbol.importPath}:${symbol.symbol}`;
      try {
        // Metadata is checked after declaration-level re-export deduplication, never by bare name.
        const original = library
          .getGPUAnalysisOperatorCatalog()
          .find(
            operator =>
              operator.package === symbol.importPath && operator.exportName === symbol.symbol
          );
        if (original && !operatorByIdentity.has(identity))
          throw new Error('missing qualified operator descriptor');
        return {...library.getCatalogAnalysisMetadata(symbol.symbol, symbol.importPath), ...symbol};
      } catch (error) {
        missing.push(`${identity}: ${error instanceof Error ? error.message : String(error)}`);
        return {questions: [], dataKinds: {input: [], output: []}, ...symbol};
      }
    })
    .sort(
      (left, right) =>
        left.symbol.localeCompare(right.symbol) || left.importPath.localeCompare(right.importPath)
    );
  if (missing.length) throw new Error(`Inventory metadata missing:\n${missing.join('\n')}`);
  const output = {
    entryPoints: entryPoints.map(entryPoint => entryPoint.importPath),
    docSlugs: documentation.slugs,
    lensScenarioIds: lensScenarios.ids,
    symbols: sorted
  };
  return output;
}

async function main(): Promise<void> {
  const output = await extractInventory();
  const outputFile = join(websiteDirectory, 'src/catalog/generated/inventory.json');
  mkdirSync(dirname(outputFile), {recursive: true});
  // Match the committed JSON layout so regenerated contracts produce focused diffs.
  writeFileSync(outputFile, `${JSON.stringify(output, null, 2)}\n`);
  console.log(
    `inventory: ${output.symbols.length} symbols -> ${relative(repositoryRoot, outputFile)}`
  );
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
