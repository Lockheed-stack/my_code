#%%
import os
# %%
path = './SCHOLAT Link Prediction/attribute'
attr_files = os.listdir(path)
node_attr_dict={}
filter_word=['研究兴趣','experience']
for file in attr_files:
    f = open(path+'/'+file)
    attrs = []
    for attr in f.readlines():
        attr=attr.strip('\n')
        if attr not in filter_word:
            attrs.append(attr)
    f.close()
    node_attr_dict[int(file[:-4])]=attrs
# %%
len(node_attr_dict)
# %%
node_attr_dict
# %%
