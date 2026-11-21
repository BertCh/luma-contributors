# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/space-time-tests/mantel-output-oracles.oracle.py
# Independent integer Random123, Hoang-Morris-Rogaway swap-or-not and NumPy Pearson definition.
# No production/test helper is imported. Round material: counter (0, draw, round, MANT).
import json
import numpy as np

mask = (1 << 32) - 1
def philox(counter, key):
    for iteration in range(10):
        first, second = 0xD2511F53 * counter[0], 0xCD9E8D57 * counter[2]
        counter = [(second >> 32) ^ counter[1] ^ key[0], second & mask,
                   (first >> 32) ^ counter[3] ^ key[1], first & mask]
        key = [(key[0] + 0x9E3779B9) & mask, (key[1] + 0xBB67AE85) & mask]
    return counter

def mix(value):
    value = ((value ^ (value >> 16)) * 0x7feb352d) & mask
    value = ((value ^ (value >> 15)) * 0x846ca68b) & mask
    return value ^ (value >> 16)

def permutation(draw, seed=17):
    positions = list(range(3))
    for round_index in range(18):  # 6 ceil(log2(3)) + 6
        words = philox([0, draw, round_index, 0x4d414e54], [seed, 0])
        # Bound 3 rejects only word=0; none of these fixed fixture words is zero.
        assert (words[0] * 3) & mask >= 1
        pivot = (words[0] * 3) >> 32
        salt = words[1]
        positions = [(pivot - position) % 3 if mix(max(position, (pivot - position) % 3) ^ salt) >> 31
                     else position for position in positions]
    return positions

times = np.array([0, 1, 3])
statistics = []
orders = [list(range(3))] + [permutation(draw) for draw in range(1, 6)]
for order in orders:
    shuffled = times[order]
    differences = np.array([abs(shuffled[0]-shuffled[1]), abs(shuffled[0]-shuffled[2]), abs(shuffled[1]-shuffled[2])])
    statistics.append(float(np.corrcoef([1, 2, 4], differences)[0, 1]))
print(json.dumps({'orders': orders, 'statistics': statistics + [0, 0]}))

orders = [list(range(3))] + [permutation(draw, 0) for draw in range(1, 6)]
statistics = []
for order in orders:
    shuffled = times[order]
    differences = [abs(shuffled[0]-shuffled[1]), abs(shuffled[0]-shuffled[2])]
    statistics.append(int(np.sign(differences[1] - differences[0])))
print(json.dumps({'two_pair_orders': orders, 'statistics': statistics,
                  'greater_count': sum(value >= statistics[0] for value in statistics[1:]),
                  'stopping_h1': next(index for index in range(1, 6) if statistics[index] >= statistics[0])}))

orders = [permutation(draw, 96) for draw in range(1, 4)]
assert orders == [[1, 0, 2]] * 3
print(json.dumps({'identical_orders': orders, 'sample_variance': 0, 'z_score': 'NaN'}))

for seed, draws in [(0,5),(1,2)]:
    orders = [list(range(3))] + [permutation(draw,seed) for draw in range(1,draws+1)]
    differences = [[abs(np.array([0,3,4])[order][a]-np.array([0,3,4])[order][b])
                    for a,b in [(0,1),(0,2),(1,2)]] for order in orders]
    dots = [int(sum(x*y for x,y in zip([1,2,1], column))) for column in differences]
    print(json.dumps({'seed':seed,'three_pair_orders':orders,'dot_products':dots,
                      'correlations':[(3*dot-32)/np.sqrt(28) for dot in dots]}))
observed = np.float32(np.corrcoef([1,3,8],[1,3,2])[0,1])
null = np.float32(np.corrcoef([1,3,8],[1,2,3])[0,1])
deviation = np.float32(null-observed)
mean_deviation = np.float32(np.float32(np.float32(deviation+deviation)+deviation)/np.float32(3))
old_variance = np.float32(np.float32(3)*np.float32(deviation-mean_deviation)**2/np.float32(2))
print(json.dumps({'seed96_distances':[1,3,8],'null':float(null),'old_f32_summary_variance':float(old_variance),
                  'expected_variance':0,'expected_z':'NaN'}))

# Review-2 R10: the old variance guard loses distinct gaps on the second draw.
f=np.float32
shift=f(f(f(1/8192)+f(1))/f(2))
first=f(f(1-1/8192)-shift)
second=f(f(1)-shift)
sum_yy=f(f(first*first)+f(second*second))
sum_y=f(first+second)
correction=f(f(sum_y*sum_y)/f(2))
assert f(sum_yy-correction)==0
rescaled=np.array([0,1/8192,1])
signs=[]
for order in [list(range(3))]+[permutation(draw,0) for draw in range(1,6)]:
    shuffled=rescaled[order]
    signs.append(int(np.sign(abs(shuffled[0]-shuffled[2])-abs(shuffled[0]-shuffled[1]))))
assert signs==[1,-1,1,-1,1,1]
print(json.dumps({'R10_old_guard_variance':float(f(sum_yy-correction)),
                  'independent_signs':signs,'upper_count':3,'p':4/6}))

# Review-2 R1/R2 shader probe: data-valued anchors avoid fractional mean rounding.
# These are f32 arithmetic probes, not claims about unexecuted GPU readback.
def model_correlation(temporal, mean_anchor):
    x=np.array([1,2,1],dtype=np.float32)-f(4/3 if mean_anchor else 1)
    y=np.array(temporal,dtype=np.float32)-f(8/3 if mean_anchor else 3)
    def total(values):
        return f(f(values[0]+values[1])+values[2])
    sx,sy=total(x),total(y)
    covariance=f(total(x*y)-f(f(sx*sy)/f(3)))
    vx=f(total(x*x)-f(f(sx*sx)/f(3)))
    vy=f(total(y*y)-f(f(sy*sy)/f(3)))
    return float(f(covariance/np.sqrt(f(vx*vy))))
old=[model_correlation(y,True) for y in [[3,4,1],[1,4,3]]]
anchored=[model_correlation(y,False) for y in [[3,4,1],[1,4,3]]]
assert old[0]!=old[1] and anchored[0]==anchored[1]
print(json.dumps({'R1_R2_old_f32_mean_anchor':old,'data_valued_anchor':anchored}))

# Hierarchical-reduction regression: first listed pair belongs to row270 of 300.
positions=[270,271,272]
for round_index in range(60): # 6 ceil(log2(300))+6
    words=philox([0,1,round_index,0x4d414e54],[0,0])
    assert (words[0]*300)&mask >= (1<<32)%300 # no bounded-draw rejection on this seed
    pivot=(words[0]*300)>>32
    positions=[(pivot-position)%300 if mix(max(position,(pivot-position)%300)^words[1])>>31
               else position for position in positions]
gaps=[abs(positions[first]-positions[second]) for first,second in [(0,1),(0,2),(1,2)]]
assert positions==[270,53,13] and gaps==[217,257,40]
print(json.dumps({'late_anchor_rows':positions,'gaps':gaps,
                  'correlation':float(np.corrcoef([1,2,1],gaps)[0,1])}))
