from __future__ import absolute_import, division, print_function
from easydict import EasyDict as edict

import os
import sys
import random
import pickle
import logging
import logging.handlers
import numpy as np
import csv
#import scipy.sparse as sp
import torch
from collections import defaultdict

# Dataset names.
#from sklearn.feature_extraction.text import TfidfTransformer

ML1M = 'ml1m'
LFM1M = 'lfm1m'
CELL = 'cellphones'

# Dataset directories.
DATASET_DIR = {
    ML1M: f'../../data/{ML1M}/preprocessed/pgpr',
    LFM1M: f'../../data/{LFM1M}/preprocessed/pgpr',
    CELL: f'../../data/{CELL}/preprocessed/pgpr'
}

# Model result directories.
TMP_DIR = {
    ML1M: f'{DATASET_DIR[ML1M]}/tmp',
    LFM1M: f'{DATASET_DIR[LFM1M]}/tmp',
    CELL: f'{DATASET_DIR[CELL]}/tmp',
}

# Label files.
LABELS = {
    ML1M: (TMP_DIR[ML1M] + '/train_label.pkl', TMP_DIR[ML1M] + '/test_label.pkl'),
    LFM1M: (TMP_DIR[ML1M] + '/train_label.pkl', TMP_DIR[ML1M] + '/test_label.pkl'),
    CELL: (TMP_DIR[ML1M] + '/train_label.pkl', TMP_DIR[ML1M] + '/test_label.pkl')
}

#ML1M ENTITIES
PRODUCT = 'product'
CINEMATOGRAPHER = 'cinematographer'
PRODUCTION_COMPANY = 'production_company'
COMPOSER = 'composer'
ACTOR = 'actor'
COUNTRY = 'country'
WIKIPAGE = 'wikipage'
EDITOR = 'editor'
WRITTER = 'writter'
DIRECTOR = 'director'

#LASTFM ENTITIES
PRODUCT = 'song'
ARTIST = 'artist'
FEATURED_ARTIST = 'featured_artist'
ENGINEER = 'engineer'
PRODUCER = 'producer'
RELATED_PRODUCT = 'related_product'

#CELL ENTITIES
BRAND = 'brand'

#SHARED ENTITIES
USER = 'user'
CATEGORY = 'category'
PRODUCT = 'product'

#ML1M RELATIONS
WATCHED = 'watched'
DIRECTED_BY = 'directed_by'
PRODUCED_BY_COMPANY = 'produced_by_company'
STARRING = 'starring'
EDITED_BY = 'edited_by'
WROTE_BY = 'wrote_by'
CINEMATOGRAPHY_BY = 'cinematography_by'
COMPOSED_BY = 'composed_by'
PRODUCED_IN = 'produced_in'

#LASTFM RELATIONS
LISTENED = 'listened'
MIXED_BY = 'mixed_by'
FEATURED_BY = 'featured_by'
SANG_BY = 'sang_by'

#CELL RELATIONS
PURCHASE = 'purchase'
ALSO_BOUGHT_RP = 'also_bought_related_product'
ALSO_VIEWED_RP = 'also_viewed_related_product'
ALSO_BOUGHT_P = 'also_bought_product'
ALSO_VIEWED_P = 'also_viewed_product'

#SHARED RELATIONS
RELATED_TO = 'related_to'
BELONG_TO = 'belong_to'
PRODUCED_BY_PRODUCER = 'produced_by_producer'
SELF_LOOP = 'self_loop'

RELATION_LIST = {
    ML1M: {
        0: "http://dbpedia.org/ontology/CINEMATOGRAPHY_BY",
        1: "http://dbpedia.org/property/productionCompanies",
        2: "http://dbpedia.org/property/composer",
        3: "http://purl.org/dc/terms/subject",
        4: "http://dbpedia.org/ontology/openingFilm",
        5: "http://www.w3.org/2000/01/rdf-schema",
        6: "http://dbpedia.org/property/story",
        7: "http://dbpedia.org/ontology/series",
        8: "http://www.w3.org/1999/02/22-rdf-syntax-ns",
        9: "http://dbpedia.org/ontology/basedOn",
        10: "http://dbpedia.org/ontology/starring",
        11: "http://dbpedia.org/ontology/country",
        12: "http://dbpedia.org/ontology/wikiPageWikiLink",
        13: "http://purl.org/linguistics/gold/hypernym",
        14: "http://dbpedia.org/ontology/editing",
        15: "http://dbpedia.org/property/producers",
        16: "http://dbpedia.org/property/allWriting",
        17: "http://dbpedia.org/property/notableWork",
        18: "http://dbpedia.org/ontology/director",
        19: "http://dbpedia.org/ontology/award",
    },
    LFM1M: {
        0: "http://rdf.freebase.com/ns/common.topic.notable_types",
        1: "http://rdf.freebase.com/ns/music.recording.releases",
        2: "http://rdf.freebase.com/ns/music.recording.artist",
        3: "http://rdf.freebase.com/ns/music.recording.engineer",
        4: "http://rdf.freebase.com/ns/music.recording.producer",
        5: "http://rdf.freebase.com/ns/music.recording.canonical_version",
        6: "http://rdf.freebase.com/ns/music.recording.song",
        7: "http://rdf.freebase.com/ns/music.single.versions",
        8: "http://rdf.freebase.com/ns/music.recording.featured_artists",
    },
}


KG_RELATION = {
    ML1M: {
        USER: {
            WATCHED: PRODUCT,
        },
        ACTOR: {
            STARRING: PRODUCT,
        },
        DIRECTOR: {
            DIRECTED_BY: PRODUCT,
        },
        PRODUCT: {
            WATCHED: USER,
            PRODUCED_BY_COMPANY: PRODUCTION_COMPANY,
            PRODUCED_BY_PRODUCER: PRODUCER,
            EDITED_BY: EDITOR,
            WROTE_BY: WRITTER,
            CINEMATOGRAPHY_BY: CINEMATOGRAPHER,
            BELONG_TO: CATEGORY,
            DIRECTED_BY: DIRECTOR,
            STARRING: ACTOR,
            COMPOSED_BY: COMPOSER,
            PRODUCED_IN: COUNTRY,
            RELATED_TO: WIKIPAGE,
        },
        PRODUCTION_COMPANY: {
            PRODUCED_BY_COMPANY: PRODUCT,
        },
        COMPOSER: {
            COMPOSED_BY: PRODUCT,
        },
        PRODUCER: {
            PRODUCED_BY_PRODUCER: PRODUCT,
        },
        WRITTER: {
            WROTE_BY: PRODUCT,
        },
        EDITOR: {
            EDITED_BY: PRODUCT,
        },
        CATEGORY: {
            BELONG_TO: PRODUCT,
        },
        CINEMATOGRAPHER: {
            CINEMATOGRAPHY_BY: PRODUCT,
        },
        COUNTRY: {
            PRODUCED_IN: PRODUCT,
        },
        WIKIPAGE: {
            RELATED_TO: PRODUCT,
        }
    },
    LFM1M: {
        USER: {
            LISTENED: PRODUCT,
        },
        ARTIST: {
            SANG_BY: PRODUCT,
        },
        ENGINEER: {
            MIXED_BY: PRODUCT,
        },
        PRODUCT: {
            LISTENED: USER,
            PRODUCED_BY_PRODUCER: PRODUCER,
            SANG_BY: ARTIST,
            FEATURED_BY: FEATURED_ARTIST,
            MIXED_BY: ENGINEER,
            BELONG_TO: CATEGORY,
            RELATED_TO: RELATED_PRODUCT,
        },
        PRODUCER: {
            PRODUCED_BY_PRODUCER: PRODUCT,
        },
        CATEGORY: {
            BELONG_TO: PRODUCT,
        },
        RELATED_PRODUCT: {
            RELATED_TO: PRODUCT,
        },
        FEATURED_ARTIST: {
            FEATURED_BY: PRODUCT,
        },
    },
    CELL: {
        USER: {
            PURCHASE: PRODUCT,
        },
        PRODUCT: {
            PURCHASE: USER,
            PRODUCED_BY_COMPANY: BRAND,
            BELONG_TO: CATEGORY,
            ALSO_BOUGHT_RP: RELATED_PRODUCT,
            ALSO_VIEWED_RP: RELATED_PRODUCT,
            ALSO_BOUGHT_P: PRODUCT,
            ALSO_VIEWED_P: PRODUCT,
        },
        BRAND: {
            PRODUCED_BY_COMPANY: PRODUCT,
        },
        CATEGORY: {
            BELONG_TO: PRODUCT,
        },
        RELATED_PRODUCT: {
            ALSO_BOUGHT_RP: PRODUCT,
            ALSO_VIEWED_RP: PRODUCT,
        }
    },
}


#0 is reserved to the main relation, 1 to mention
PATH_PATTERN = {
    ML1M: {
        0: ((None, USER), (WATCHED, PRODUCT), (WATCHED, USER), (WATCHED, PRODUCT)),
        2: ((None, USER), (WATCHED, PRODUCT), (CINEMATOGRAPHY_BY, CINEMATOGRAPHER), (CINEMATOGRAPHY_BY, PRODUCT)),
        3: ((None, USER), (WATCHED, PRODUCT), (PRODUCED_BY_COMPANY, PRODUCTION_COMPANY), (PRODUCED_BY_COMPANY, PRODUCT)),
        4: ((None, USER), (WATCHED, PRODUCT), (COMPOSED_BY, COMPOSER), (COMPOSED_BY, PRODUCT)),
        5: ((None, USER), (WATCHED, PRODUCT), (BELONG_TO, CATEGORY), (BELONG_TO, PRODUCT)),
        7: ((None, USER), (WATCHED, PRODUCT), (STARRING, ACTOR), (STARRING, PRODUCT)),
        8: ((None, USER), (WATCHED, PRODUCT), (EDITED_BY, EDITOR), (EDITED_BY, PRODUCT)),
        9: ((None, USER), (WATCHED, PRODUCT), (PRODUCED_BY_PRODUCER, PRODUCER), (PRODUCED_BY_PRODUCER, PRODUCT)),
        10: ((None, USER), (WATCHED, PRODUCT), (WROTE_BY, WRITTER), (WROTE_BY, PRODUCT)),
        11: ((None, USER), (WATCHED, PRODUCT), (DIRECTED_BY, DIRECTOR), (DIRECTED_BY, PRODUCT)),
        12: ((None, USER), (WATCHED, PRODUCT), (PRODUCED_IN, COUNTRY), (PRODUCED_IN, PRODUCT)),
        #13: ((None, USER), (WATCHED, PRODUCT), (RELATED_TO, WIKIPAGE), (RELATED_TO, PRODUCT)),
    },
    LFM1M: {
        0: ((None, USER), (LISTENED, PRODUCT), (LISTENED, USER), (LISTENED, PRODUCT)),
        2: ((None, USER), (LISTENED, PRODUCT), (BELONG_TO, CATEGORY), (BELONG_TO, PRODUCT)),
        3: ((None, USER), (LISTENED, PRODUCT), (RELATED_TO, RELATED_PRODUCT), (RELATED_TO, PRODUCT)),
        4: ((None, USER), (LISTENED, PRODUCT), (SANG_BY, ARTIST), (SANG_BY, PRODUCT)),
        5: ((None, USER), (LISTENED, PRODUCT), (MIXED_BY, ENGINEER), (MIXED_BY, PRODUCT)),
        6: ((None, USER), (LISTENED, PRODUCT), (PRODUCED_BY_PRODUCER, PRODUCER), (PRODUCED_BY_PRODUCER, PRODUCT)),
        #10: ((None, USER), (LISTENED, PRODUCT), (FEATURED_BY, FEATURED_ARTIST), (FEATURED_BY, PRODUCT)),
    },
    CELL: {
        0: ((None, USER), (PURCHASE, PRODUCT), (PURCHASE, USER), (PURCHASE, PRODUCT)),
        2: ((None, USER), (PURCHASE, PRODUCT), (BELONG_TO, CATEGORY), (BELONG_TO, PRODUCT)),
        3: ((None, USER), (PURCHASE, PRODUCT), (PRODUCED_BY_COMPANY, BRAND), (PRODUCED_BY_COMPANY, PRODUCT)),
        4: ((None, USER), (PURCHASE, PRODUCT), (ALSO_BOUGHT_P, PRODUCT)),
        5: ((None, USER), (PURCHASE, PRODUCT), (ALSO_VIEWED_P, PRODUCT)),
        6: ((None, USER), (PURCHASE, PRODUCT), (ALSO_BOUGHT_RP, RELATED_PRODUCT), (ALSO_BOUGHT_RP, PRODUCT)),
        10: ((None, USER), (PURCHASE, PRODUCT), (ALSO_VIEWED_RP, RELATED_PRODUCT), (ALSO_VIEWED_RP, PRODUCT)),
    }
}


MAIN_PRODUCT_INTERACTION = {
    ML1M: (PRODUCT, WATCHED),
    LFM1M: (PRODUCT, LISTENED),
    CELL: (PRODUCT, PURCHASE)
}



def get_entities(dataset_name):
    return list(KG_RELATION[dataset_name].keys())


def get_knowledge_derived_relations(dataset_name):
    main_entity, main_relation = MAIN_PRODUCT_INTERACTION[dataset_name]
    ans = list(KG_RELATION[dataset_name][main_entity].keys())
    ans.remove(main_relation)
    return ans


def get_dataset_relations(dataset_name, entity_head):
    return list(KG_RELATION[dataset_name][entity_head].keys())


def get_entity_tail(dataset_name, relation):
    entity_head, _ = MAIN_PRODUCT_INTERACTION[dataset_name]
    return KG_RELATION[dataset_name][entity_head][relation]


def get_logger(logname):
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]  %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.handlers.RotatingFileHandler(logname, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_dataset(dataset, dataset_obj):
    dataset_file = os.path.join(TMP_DIR[dataset], 'dataset.pkl')
    with open(dataset_file, 'wb') as f:
        pickle.dump(dataset_obj, f)


def load_dataset(dataset):
    dataset_file = os.path.join(TMP_DIR[dataset], 'dataset.pkl')
    dataset_obj = pickle.load(open(dataset_file, 'rb'))
    return dataset_obj


def save_labels(dataset, labels, mode='train'):
    if mode == 'train':
        label_file = LABELS[dataset][0]
    elif mode == 'test':
        label_file = LABELS[dataset][1]
    else:
        raise Exception('mode should be one of {train, test}.')
    with open(label_file, 'wb') as f:
        pickle.dump(labels, f)


def load_labels(dataset, mode='train'):
    if mode == 'train':
        label_file = LABELS[dataset][0]
        # CHANGED
    elif mode == 'test':
        label_file = LABELS[dataset][1]
    else:
        raise Exception('mode should be one of {train, test}.')
    user_products = pickle.load(open(label_file, 'rb'))
    return user_products


def save_embed(dataset, embed):
    embed_file = '{}/transe_embed.pkl'.format(TMP_DIR[dataset])
    pickle.dump(embed, open(embed_file, 'wb'))


def load_embed(dataset):
    embed_file = '{}/transe_embed.pkl'.format(TMP_DIR[dataset])
    print('Load embedding:', embed_file)
    embed = pickle.load(open(embed_file, 'rb'))
    return embed

#Receive paths in form (score, prob, [path]) return the last relationship
def get_path_pattern(path):
    return path[-1][-1][0]



def get_pid_to_kgid_mapping(dataset_name):
    if dataset_name == "ml1m":
        file = open(DATASET_DIR[dataset_name] + "/entities/mappings/movie.txt", "r")
    elif dataset_name == "lfm1m":
        file = open(DATASET_DIR[dataset_name] + "/entities/mappings/song.txt", "r")
    else:
        print("Dataset mapping not found!")
        exit(-1)
    reader = csv.reader(file, delimiter=' ')
    dataset_pid2kg_pid = {}
    next(reader, None)
    for row in reader:
        if dataset_name == "ml1m" or dataset_name == "lfm1m":
            dataset_pid2kg_pid[int(row[0])] = int(row[1])
    file.close()
    return dataset_pid2kg_pid


def get_entity_edict(dataset_name):
    if dataset_name == ML1M:
        entity_files = edict(
            user='users.txt.gz',
            product='products.txt.gz',
            actor='actor.txt.gz',
            composer='composer.txt.gz',
            director='director.txt.gz',
            producer='producer.txt.gz',
            production_company='production_company.txt.gz',
            category='category.txt.gz',
            country='country.txt.gz',
            editor='editor.txt.gz',
            writter='writter.txt.gz',
            cinematographer='cinematographer.txt.gz',
            wikipage='wikipage.txt.gz',
        )
    elif dataset_name == LFM1M:
        entity_files = edict(
            user='users.txt.gz',
            product='products.txt.gz',
            artist='artist.txt.gz',
            featured_artist='featured_artist.txt.gz',
            engineer='engineer.txt.gz',
            producer='producer.txt.gz',
            category='category.txt.gz',
            related_product='related_product.txt.gz',
        )
    elif dataset_name == CELL:
        entity_files = edict(
            user='users.txt.gz',
            product='products.txt.gz',
            related_product='related_product.txt.gz',
            brand='brand.txt.gz',
            category='category.txt.gz',
        )
    return entity_files


def get_validation_pids(dataset_name):
    if not os.path.isfile(os.path.join(DATASET_DIR[dataset_name], 'valid.txt')):
        return []
    validation_pids = defaultdict(set)
    with open(os.path.join(DATASET_DIR[dataset_name], 'valid.txt')) as valid_file:
        reader = csv.reader(valid_file, delimiter=" ")
        for row in reader:
            uid = int(row[0])
            pid = int(row[1])
            validation_pids[uid].add(pid)
    valid_file.close()
    return validation_pids

def get_uid_to_kgid_mapping(dataset_name):
    dataset_uid2kg_uid = {}
    with open(DATASET_DIR[dataset_name] + "/entities/mappings/user.txt", 'r') as file:
        reader = csv.reader(file, delimiter=" ")
        next(reader, None)
        for row in reader:
            if dataset_name == "ml1m" or dataset_name == "lfm1m":
                uid_review = int(row[0])
            uid_kg = int(row[1])
            dataset_uid2kg_uid[uid_review] = uid_kg
    return dataset_uid2kg_uid

def save_kg(dataset, kg):
    kg_file = TMP_DIR[dataset] + '/kg.pkl'
    pickle.dump(kg, open(kg_file, 'wb'))


def load_kg(dataset):
    kg_file = TMP_DIR[dataset] + '/kg.pkl'
    # CHANGED
    kg = pickle.load(open(kg_file, 'rb'))
    return kg

def shuffle(arr):
    for i in range(len(arr) - 1, 0, -1):
        # Pick a random index from 0 to i
        j = random.randint(0, i + 1)

        # Swap arr[i] with the element at random index
        arr[i], arr[j] = arr[j], arr[i]
    return arr