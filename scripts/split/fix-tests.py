# Post-port test fixes: luma-specific package names, globs and paths that the specifier rewrite cannot see.
import os
import subprocess
import sys

os.chdir(sys.argv[1])


def edit(path, pairs):
    text = open(path).read()
    for old, new in pairs:
        assert old in text, (path, old)
        text = text.replace(old, new)
    open(path, 'w').write(text)


edit('modules/graph/test/analysis-core/analysis-layering.node.spec.ts',
     [('^@luma\\.gl\\/graph\\/gpu-core$', '^@luma-contributors\\/graph\\/gpu-core$')])
for path in ['modules/graph/test/gpu-project/projection-planning.node.spec.ts',
             'modules/graph/test/gpu-project/projection-crs.node.spec.ts']:
    edit(path, [("external: ['@luma.gl/*']", "external: ['@luma.gl/*', '@luma-contributors/*']")])

files = subprocess.run(['grep', '-rl', 'modules/graph-lenses', 'modules'], capture_output=True, text=True).stdout.split()
for path in files:
    if path.endswith(('.ts', '.tsx', '.mjs', '.md', '.json')) and '/src/' not in path or path.endswith('README.md'):
        text = open(path).read()
        open(path, 'w').write(text.replace('modules/graph-lenses', 'modules/lenses'))
        print('path fix', path)

# The WGSL scan lists modules/*/src; this repo has no engine module, so check lenses instead.
edit('modules/graph/test/analysis-core/wgsl-reserved-identifiers.node.spec.ts',
     [("root.endsWith('/modules/engine/src')", "root.endsWith('/modules/lenses/src')")])

# GPUTableModel lives in @luma.gl/engine and binds tables through npm @luma.gl/graph. Those tests
# belong to luma.gl, and forked GPUTable instances are not interchangeable with npm graph tables.
for path in ['modules/graph/test/gpu-tables/gpu-table-model.node.spec.ts',
             'modules/graph/test/gpu-tables/gpu-table-model.spec.ts']:
    if os.path.exists(path):
        os.remove(path)
        print('removed engine-owned test', path)
