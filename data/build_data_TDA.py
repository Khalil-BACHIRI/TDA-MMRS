import array
import gzip
import json
import os
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer

from gtda.homology import VietorisRipsPersistence
from gensim.models import KeyedVectors
import re
from skimage.color import rgb2gray
import gudhi as gd
from skimage import io
from gudhi.representations import Entropy, vector_methods, Landscape, Silhouette
from to_other_models import to_other_models
from gudhi.representations import Entropy, BettiCurve, Landscape, Silhouette
from gtda.homology import VietorisRipsPersistence

from scipy import ndimage
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

np.random.seed(123)

name = 'Musical_Instruments'
folder = './'+name+'/'
print("Initializing SentenceTransformer...")
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

core = 5

if not os.path.exists(folder + '%d-core'%core):
    os.makedirs(folder + '%d-core'%core)

def compute_TDA_text(sentence):
    sentence = sentence.lower()
    sentence = re.sub("[^a-z']", " ", sentence)
    words = list(set([w for w in sentence.split() if w in google_vectors_2.key_to_index]))
    n = len(words)

    # print(sentence)

    dissimilarity = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            s = google_vectors_2.similarity(words[i], words[j])
            dissimilarity[i][j] = 1 - s
            dissimilarity[j][i] = 1 - s

    VR = VietorisRipsPersistence(metric="precomputed")

    persistence = VR.fit_transform([dissimilarity])[0]

    persistence_0 = persistence[persistence[:, -1] == 0][:, :2]
    persistence_1 = persistence[persistence[:, -1] == 1][:, :2]
    persistence_0_no_inf = np.array([bars for bars in persistence_0 if bars[1] != np.inf])
    persistence_1_no_inf = np.array([bars for bars in persistence_1 if bars[1] != np.inf])

    pt_0 = np.sum(
        np.fromiter((interval[1] - interval[0] for interval in persistence_0_no_inf), dtype=np.dtype(np.float64)))
    pt_1 = np.sum(
        np.fromiter((interval[1] - interval[0] for interval in persistence_1_no_inf), dtype=np.dtype(np.float64)))

    al_0 = 0
    al_1 = 0

    sd_0 = 0
    sd_1 = 0

    PE = gd.representations.Entropy()
    pe_0 = 0
    pe_1 = 0

    # Betti numbers
    bc = gd.representations.vector_methods.BettiCurve()
    bc_0 = np.zeros(100)
    bc_1 = np.zeros(100)

    # Landscapes
    num_landscapes = 10
    points_per_landscape = 100
    lc = gd.representations.Landscape(num_landscapes=num_landscapes, resolution=points_per_landscape)
    area_under_lc_0 = np.zeros(num_landscapes)
    area_under_lc_1 = np.zeros(num_landscapes)

    # Silhouettes
    p = 2
    resolution = 100
    s = gd.representations.Silhouette()
    s2 = gd.representations.Silhouette(weight=lambda x: np.power(x[1] - x[0], p), resolution=resolution)
    area_under_s_0 = 0
    area_under_s_1 = 0
    area_under_s2_0 = 0
    area_under_s2_1 = 0

    if (persistence_0_no_inf.size > 0):
        al_0 = pt_0 / len(persistence_0_no_inf)
        sd_0 = np.std([(start + end) / 2 for start, end in persistence_0_no_inf])
        pe_0 = PE.fit_transform([persistence_0_no_inf])[0][0]
        bc_0 = bc(persistence_0_no_inf)
        reshaped_landscapes_0 = lc(persistence_0_no_inf).reshape(num_landscapes,points_per_landscape)
        for i in range(num_landscapes):
            area_under_lc_0[i] = np.trapz(reshaped_landscapes_0[i], dx=1)
        s_0 = s(persistence_0_no_inf)
        s2_0 = s2(persistence_0_no_inf)
        area_under_s_0 = np.trapz(s_0, dx=1)
        area_under_s2_0 = np.trapz(s2_0, dx=1)

    if (persistence_1_no_inf.size > 0):
        al_1 = pt_1 / len(persistence_1_no_inf)
        sd_1 = np.std([(start + end) / 2 for start, end in persistence_1_no_inf])
        pe_1 = PE.fit_transform([persistence_1_no_inf])[0][0]
        bc_1 = bc(persistence_1_no_inf)
        reshaped_landscapes_1 = lc(persistence_1_no_inf).reshape(num_landscapes, points_per_landscape)
        for i in range(num_landscapes):
            area_under_lc_1[i] = np.trapz(reshaped_landscapes_1[i], dx=1)
        s_1 = s(persistence_1_no_inf)
        s2_1 = s2(persistence_1_no_inf)
        area_under_s_1 = np.trapz(s_1, dx=1)
        area_under_s2_1 = np.trapz(s2_1, dx=1)

    return np.nan_to_num(np.concatenate(
        (np.array([pt_0, pt_1, al_0, al_1, sd_0, sd_1, pe_0, pe_1, area_under_s_0, area_under_s_1, area_under_s2_0,
                   area_under_s2_1]), area_under_lc_0, area_under_lc_1, np.array(bc_0), np.array(bc_1))), nan=0)

def compute_TDA_image(grayscale_image):
    flat_grayscale_image = grayscale_image.flatten()
    # CubicalComplex
    cc = gd.CubicalComplex(dimensions=grayscale_image.shape, top_dimensional_cells=flat_grayscale_image)

    # Persistence
    persistence = cc.persistence()

    # Descriptors
    persistence_0 = cc.persistence_intervals_in_dimension(0)  # intervals de persistencia de dimensio 0
    persistence_1 = cc.persistence_intervals_in_dimension(1)  # intervals de persistencia de dimensio 1

    persistence_0_no_inf = np.array([bars for bars in persistence_0 if bars[1] != np.inf])
    persistence_1_no_inf = np.array([bars for bars in persistence_1 if bars[1] != np.inf])

    # Persistence total
    pt_0 = np.sum(
        np.fromiter((interval[1] - interval[0] for interval in persistence_0_no_inf), dtype=np.dtype(np.float64)))
    pt_1 = np.sum(
        np.fromiter((interval[1] - interval[0] for interval in persistence_1_no_inf), dtype=np.dtype(np.float64)))

    al_0 = 0
    al_1 = 0

    # Standard Deviation
    sd_0 = 0
    sd_1 = 0

    # Entropy
    PE = gd.representations.Entropy()
    pe_0 = 0
    pe_1 = 0

    # Betti curves
    bc = gd.representations.vector_methods.BettiCurve()
    bc_0 = np.zeros(100)
    bc_1 = np.zeros(100)

    # Landscapes
    num_landscapes = 10
    points_per_landscape = 100
    lc = gd.representations.Landscape(num_landscapes=num_landscapes, resolution=points_per_landscape)
    area_under_lc_0 = np.zeros(num_landscapes)
    area_under_lc_1 = np.zeros(num_landscapes)

    # Silhouettes
    p = 2
    resolution = 100
    s = gd.representations.Silhouette()
    s2 = gd.representations.Silhouette(weight=lambda x: np.power(x[1] - x[0], p), resolution=resolution)
    area_under_s_0 = 0
    area_under_s_1 = 0
    area_under_s2_0 = 0
    area_under_s2_1 = 0

    if (persistence_0_no_inf.size > 0):
        al_0 = pt_0 / len(persistence_0_no_inf)
        sd_0 = np.std([(start + end) / 2 for start, end in persistence_0_no_inf])
        pe_0 = PE.fit_transform([persistence_0_no_inf])[0][0]
        bc_0 = bc(persistence_0_no_inf)
        reshaped_landscapes_0 = lc(persistence_0_no_inf).reshape(num_landscapes,points_per_landscape)
        for i in range(num_landscapes):
            area_under_lc_0[i] = np.trapz(reshaped_landscapes_0[i], dx=1)
        s_0 = s(persistence_0_no_inf)
        s2_0 = s2(persistence_0_no_inf)
        area_under_s_0 = np.trapz(s_0, dx=1)
        area_under_s2_0 = np.trapz(s2_0, dx=1)

    if (persistence_1_no_inf.size > 0):
        al_1 = pt_1 / len(persistence_1_no_inf)
        sd_1 = np.std([(start + end) / 2 for start, end in persistence_1_no_inf])
        pe_1 = PE.fit_transform([persistence_1_no_inf])[0][0]
        bc_1 = bc(persistence_1_no_inf)
        reshaped_landscapes_1 = lc(persistence_1_no_inf).reshape(num_landscapes, points_per_landscape)
        for i in range(num_landscapes):
            area_under_lc_1[i] = np.trapz(reshaped_landscapes_1[i], dx=1)
        s_1 = s(persistence_1_no_inf)
        s2_1 = s2(persistence_1_no_inf)
        area_under_s_1 = np.trapz(s_1, dx=1)
        area_under_s2_1 = np.trapz(s2_1, dx=1)

    return np.concatenate(
        (np.array([pt_0, pt_1, al_0, al_1, sd_0, sd_1, pe_0, pe_1, area_under_s_0, area_under_s_1, area_under_s2_0,
                   area_under_s2_1]), area_under_lc_0, area_under_lc_1, np.array(bc_0), np.array(bc_1)))

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.dumps(eval(l))

print("----------parse metadata----------")
if not os.path.exists(folder + "meta-data/meta.json"):
    with open(folder + "meta-data/meta.json", 'w') as f:
        for l in parse(folder + 'meta-data/' + "meta_%s.json.gz"%(name)):
            f.write(l+'\n')

print("----------parse data----------")
if not os.path.exists(folder + "meta-data/%d-core.json" % core):
    with open(folder + "meta-data/%d-core.json" % core, 'w') as f:
        for l in parse(folder + 'meta-data/' + "reviews_%s_%d.json.gz"%(name, core)):
            f.write(l+'\n')

print("----------load data----------")
jsons = []
for line in open(folder + "meta-data/%d-core.json" % core).readlines():
    jsons.append(json.loads(line))


print("----------Build dict----------")
items = set()
users = set()
for j in jsons:
    items.add(j['asin'])
    users.add(j['reviewerID'])
print("n_items:", len(items), "n_users:", len(users))


item2id = {}
with open(folder + '%d-core/item_list.txt'%core, 'w') as f:
    for i, item in enumerate(items):
        item2id[item] = i
        f.writelines(item+'\t'+str(i)+'\n')

user2id =  {}
with open(folder + '%d-core/user_list.txt'%core, 'w') as f:
    for i, user in enumerate(users):
        user2id[user] = i
        f.writelines(user+'\t'+str(i)+'\n')


ui = defaultdict(list)
for j in jsons:
    u_id = user2id[j['reviewerID']]
    i_id = item2id[j['asin']]
    ui[u_id].append(i_id)
with open(folder + '%d-core/user-item-dict.json'%core, 'w') as f:
    f.write(json.dumps(ui))


print("----------Split Data----------")
train_json = {}
val_json = {}
test_json = {}
for u, items in ui.items():
    if len(items) < 10:
        testval = np.random.choice(len(items), 2, replace=False)
    else:
        testval = np.random.choice(len(items), int(len(items) * 0.2), replace=False)

    test = testval[:len(testval)//2]
    val = testval[len(testval)//2:]
    train = [i for i in list(range(len(items))) if i not in testval]
    train_json[u] = [items[idx] for idx in train]
    val_json[u] = [items[idx] for idx in val.tolist()]
    test_json[u] = [items[idx] for idx in test.tolist()]

with open(folder + '%d-core/train.json'%core, 'w') as f:
    json.dump(train_json, f)
with open(folder + '%d-core/val.json'%core, 'w') as f:
    json.dump(val_json, f)
with open(folder + '%d-core/test.json'%core, 'w') as f:
    json.dump(test_json, f)


jsons = []
with open(folder + "meta-data/meta.json", 'r') as f:
    for line in f.readlines():
        jsons.append(json.loads(line))

print("----------Text Features----------")
raw_text = {}
image_links = {}
for json in jsons:
    if json['asin'] in item2id:
        string = ' '
        if 'categories' in json:
            for cates in json['categories']:
                for cate in cates:
                    string += cate + ' '
        if 'title' in json:
            string += json['title']
        if 'brand' in json:
            # hauria de ser brand??
            string += json['brand']
        if 'description' in json:
            string += json['description']
        raw_text[item2id[json['asin']]] = string.replace('\n', ' ')
        # Add url
        if 'imUrl' in json:
            image_links[json['asin']] = json['imUrl']
        else:
            image_links[json['asin']] = ''
texts = []

google_vectors_2 = KeyedVectors.load('google.d2v')



# bert

with open(folder + '%d-core/raw_text.txt'%core, 'w') as f:
    for i in range(len(item2id)):
        f.write(raw_text[i] + '\n')
        texts.append(raw_text[i] + '\n')
sentence_embeddings = bert_model.encode(texts)
assert sentence_embeddings.shape[0] == len(item2id)

np.save(folder+'text_feat.npy', sentence_embeddings)

complete_embeddings = []

#TDA i concat
for i in range(len(texts)):
    original_values = sentence_embeddings[i]
    tda = compute_TDA_text(texts[i])
    complete_embeddings.append(np.concatenate((original_values, tda), axis=0))

complete_embeddings = np.array(complete_embeddings)

assert complete_embeddings.shape[0] == len(item2id)


np.save(folder+'text_feat_TDA.npy', np.array(complete_embeddings))



# Prepare containers
attributes_feat = {}
brand_list = []
category_list = []
temp_feats = {}

# First pass to collect brands and categories
for json in jsons:
    if json['asin'] in item2id:
        brand = json.get('brand', 'unknown')
        brand_list.append(brand)

        categories = []
        if 'categories' in json:
            for cates in json['categories']:
                categories.extend(cates)
        categories = list(set(categories)) if categories else ['unknown']
        category_list.extend(categories)

# Build encoder for brands and categories
brand_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
category_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

brand_encoder.fit(np.array(brand_list).reshape(-1, 1))
category_encoder.fit(np.array(category_list).reshape(-1, 1))

# Second pass to build features
for json in jsons:
    if json['asin'] in item2id:
        features = []
        item_id = item2id[json['asin']]

        # 1. Price (normalize missing values)
        try:
            price = float(str(json.get('price', '0')).replace('$', '').replace(',', ''))
        except:
            price = 0.0
        features.append(price)

        # 2. Brand (one-hot)
        brand = json.get('brand', 'unknown')
        brand_vec = brand_encoder.transform([[brand]])[0]
        features.extend(brand_vec)

        # 3. Categories (multi-hot)
        categories = []
        if 'categories' in json:
            for cates in json['categories']:
                categories.extend(cates)
        categories = list(set(categories)) if categories else ['unknown']

        cat_vec = category_encoder.transform([[c]]).sum(axis=0) if categories else np.zeros(len(category_encoder.categories_[0]))
        features.extend(cat_vec)

        # Save raw feature vector
        temp_feats[item_id] = features

# Normalize price and align to all items
all_feats_array = np.array(list(temp_feats.values()))
scaler = StandardScaler()
all_feats_array = scaler.fit_transform(all_feats_array)

# Replace features with normalized
for i, key in enumerate(temp_feats):
    temp_feats[key] = all_feats_array[i].tolist()

# Compute average vector for missing items
avg_attr = np.mean(all_feats_array, axis=0).tolist()

# Align features to item2id index
final_attributes = []
for i in range(len(item2id)):
    if i in temp_feats:
        final_attributes.append(temp_feats[i])
    else:
        final_attributes.append(avg_attr)

final_attributes = np.array(final_attributes)
np.save(folder + 'attributes_feat.npy', final_attributes)

print("----------User Behavior TDA----------")
# Load user-item interaction dict
with open(folder + f'{core}-core/user-item-dict.json', 'r') as f:
    user_item_dict = json.load(f)

num_items = len(item2id)
num_users = len(user_item_dict)

# Initialize user-item binary matrix
user_behavior_matrix = np.zeros((num_users, num_items))

for u_str, items in user_item_dict.items():
    u = int(u_str)
    user_behavior_matrix[u, items] = 1  # binary interaction vector

# Normalize interaction vectors
from sklearn.preprocessing import StandardScaler
user_behavior_matrix = StandardScaler().fit_transform(user_behavior_matrix)

# TDA setup
tda_user_behavior = []
VR = VietorisRipsPersistence(metric="euclidean")

PE = Entropy()
bc = BettiCurve()
lc = Landscape(num_landscapes=5, resolution=50)
sl = Silhouette()

for i in range(user_behavior_matrix.shape[0]):
    x = user_behavior_matrix[i]
    x_matrix = np.expand_dims(x, axis=0)

    # Compute persistence diagram
    persistence = VR.fit_transform([x_matrix])[0]

    if persistence.shape[0] == 0:
        descriptors = np.zeros(1 + 5 + 50 + 50)
    else:
        entropy = PE.fit_transform([persistence])[0]
        betti = bc(persistence)
        land = lc(persistence)
        silhou = sl(persistence)
        descriptors = np.concatenate([entropy, betti, land, silhou])
    tda_user_behavior.append(descriptors)

tda_user_behavior = np.array(tda_user_behavior)
np.save(folder + 'user_behavior_feat_TDA.npy', tda_user_behavior)


print("----------Image Features----------")
def readImageFeatures(path):
    f = open(path, 'rb')
    while True:
        asin = f.read(10).decode('UTF-8')
        if asin == '': break
        a = array.array('f')
        a.fromfile(f, 4096)
        yield asin, a.tolist()

data = readImageFeatures(folder + 'meta-data/' + "image_features_%s.b" % name)
feats = {}
feats_TDA={}
avg = []
avg_TDA = []



for d in data:
    if d[0] in item2id:
        feats[int(item2id[d[0]])] = d[1]
        avg.append(d[1])

        # TDA to image
        img_link = image_links[d[0]]

        if(img_link != '' and img_link[-3:] != 'gif'):
            # 1. image processing
            image = io.imread(img_link)
            grayscale_image = rgb2gray(image)

            # descriptor
            aux1 = compute_TDA_image(grayscale_image)

            # Convolution
            treshold = 0.5
            binarized_image = grayscale_image > treshold
            aux2 = compute_TDA_image(binarized_image)

            # Convolution
            convolve = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
            convolved_image = ndimage.convolve(grayscale_image, convolve, mode='constant', cval=0.0)
            convolved_image = convolved_image / 9
            aux3 = compute_TDA_image(convolved_image)

            aux = np.concatenate((aux1, aux2, aux3))

            feats_TDA[int(item2id[d[0]])] = aux
            avg_TDA.append(aux)

if avg != []:
    avg = np.array(avg).mean(0).tolist()
if(avg_TDA != []):
    avg_TDA = np.array(avg_TDA).mean(0).tolist()

ret_TDA = []
ret = []
for i in range(len(item2id)):
    p1 = []
    p2 = []
    if i in feats:
        p1 = feats[i]
        ret.append(feats[i])
    else:
        p1 = avg
        ret.append(avg)
    if i in feats_TDA:
        p2 = feats_TDA[i]
    else:
        p2 = avg_TDA

    ret_TDA.append(np.concatenate((p1, p2)))

assert len(ret) == len(item2id)
assert len(ret_TDA) == len(item2id)

np.save(folder+'image_feat.npy', np.array(ret))
np.save(folder+'image_feat_TDA.npy',np.array(ret_TDA))



to_other_models(name)