# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Execute pinned upstream depthmapX mathematical helpers; upstream source stays temporary.
Requires a C++ compiler. This checks normalization functions, not the entire GUI/CLI pipeline.
"""
import hashlib,json,pathlib,subprocess,tempfile,urllib.request
revision='6ea9755'
url=f'https://raw.githubusercontent.com/SpaceGroupUCL/depthmapX/{revision}/genlib/pafmath.h'
header=urllib.request.urlopen(url).read()
with tempfile.TemporaryDirectory(prefix='luma-depthmap-') as directory:
 path=pathlib.Path(directory);(path/'pafmath.h').write_bytes(header)
 (path/'reference.cpp').write_text('#include "pafmath.h"\n#include <iostream>\n#include <iomanip>\nint main(){double count,total,maximum;std::cout<<std::setprecision(17);while(std::cin>>count>>total>>maximum){double asymmetry=2*(total/(count-1)-1)/(count-2);std::cout<<dvalue(count)/asymmetry<<" "<<pvalue(count)/asymmetry<<" "<<teklinteg(count,total)<<" "<<(palmtree(count,maximum)-total)/(palmtree(count,maximum)-(count-1))<<"\\n";}}')
 subprocess.run(['c++','-std=c++11',str(path/'reference.cpp'),'-o',str(path/'reference')],check=True)
 # Five collinear viewpoints at unit spacing with range 1.1 form an independently known path.
 depths=[[abs(first-second) for second in range(5)] for first in range(5)]
 inputs=''.join(f'5 {sum(values)} {max(values)}\n' for values in depths)
 results=subprocess.check_output([str(path/'reference')],input=inputs.encode()).decode().splitlines()
 result={'upstream':'depthmapX v0.9.1','revision':revision,'source':url,'sha256':hashlib.sha256(header).hexdigest(),'path':{'positions':[component for vertex in range(5) for component in [vertex,0]],'rows':[list(map(float,row.split())) for row in results],'maximumDepth':[max(values) for values in depths]}}
 pathlib.Path(__file__).with_name('fixtures').joinpath('urban-depthmap.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
