import json
import pandas as pd
import numpy as np
from string import Template


dataset = 'ht09_contact'  # infectious SFHH tij_InVS15
# trim because vis.js can't scale
slice_start = 2000
slice_length = 400


data = {}
df = pd.read_csv(f'./{dataset}.txt', sep='\t', header=None, names=['source', 'destination', 'timestamp'])
df = df.iloc[slice_start:slice_start + slice_length]

with open('../vis/main.html.template', 'r') as f:
    template = Template(f.read())
html = template.safe_substitute(min_slider=df['timestamp'].min(), max_slider=df['timestamp'].max())
with open('../vis/main.html', 'w') as f:
    f.write(html)

src = df['source']
dst = df['destination']
ts = df['timestamp']
nodes = np.union1d(src.unique(), dst.unique())

data['nodes'] = [{ 'id': int(node) } for node in nodes]
data['edges'] = [{ 'from': int(s), 'to': int(d), 'ts': int(t) } for s, d, t in zip(src, dst, ts)]

out = 'data = ' + json.dumps(data)
with open('../vis/data.js', 'w') as f:
    f.write(out)
