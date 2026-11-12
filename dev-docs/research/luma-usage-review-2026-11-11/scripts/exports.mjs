// Lists public exports (values and types) per entry point of a luma.gl package root, using the TS checker.
import ts from 'typescript';
import fs from 'node:fs';
import path from 'node:path';
const [root, out] = process.argv.slice(2);
const result = {};
for (const pkg of ['core', 'shaders', 'graph', 'engine']) {
  const dir = path.join(root, pkg);
  const json = JSON.parse(fs.readFileSync(path.join(dir, 'package.json'), 'utf8'));
  const entries = [];
  for (const [key, value] of Object.entries(json.exports || {})) {
    if (key.includes('*') || key.includes('_internal')) continue;
    let types = typeof value === 'string' ? null : value.types || (value.import && value.import.types) || null;
    if (!types && typeof value === 'object') { const imp = value.import || value.default; if (typeof imp === 'string') types = imp.replace(/\.js$/, '.d.ts'); }
    if (!types) continue;
    const file = path.join(dir, types);
    if (!fs.existsSync(file)) continue;
    entries.push([`@luma.gl/${pkg}${key === '.' ? '' : key.slice(1)}`, file]);
  }
  const program = ts.createProgram(entries.map(e => e[1]), {skipLibCheck: true, noResolve: false, moduleResolution: ts.ModuleResolutionKind.Bundler, types: []});
  const checker = program.getTypeChecker();
  for (const [name, file] of entries) {
    const source = program.getSourceFile(file);
    const symbol = source && checker.getSymbolAtLocation(source);
    if (!symbol) { result[name] = null; continue; }
    const map = {};
    for (const exported of checker.getExportsOfModule(symbol)) {
      let target = exported;
      if (target.flags & ts.SymbolFlags.Alias) { try { target = checker.getAliasedSymbol(exported); } catch {} }
      const f = target.flags;
      const kind = f & ts.SymbolFlags.Class ? 'class' : f & ts.SymbolFlags.Function ? 'function' : f & ts.SymbolFlags.Variable ? 'const' : f & ts.SymbolFlags.Enum ? 'enum' : 'type';
      const decl = target.declarations && target.declarations[0];
      map[exported.name] = {kind, file: decl ? path.relative(dir, decl.getSourceFile().fileName) : ''};
    }
    result[name] = map;
  }
}
fs.writeFileSync(out, JSON.stringify(result));
for (const [k, v] of Object.entries(result)) console.log(k, v ? Object.keys(v).length : 'unresolved');
