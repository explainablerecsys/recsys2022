import csv
import os
from collections import defaultdict
import argparse
from utils import ensure_dir
import gzip

ML1M = 'ml1m'
LFM1M = 'lfm1m'
CELL = 'cellphones'

"""
ENTITIES
"""
#ML1M ENTITIES
MOVIE = 'movie'
ACTOR = 'actor'
DIRECTOR = 'director'
PRODUCTION_COMPANY = 'production_company'
EDITOR = 'editor'
WRITTER = 'writter'
CINEMATOGRAPHER = 'cinematographer'
COMPOSER = 'composer'
COUNTRY = 'country'
AWARD = 'award'

#LASTFM ENTITIES
SONG = 'song'
ARTIST = 'artist'
ENGINEER = 'engineer'
PRODUCER = 'producer'

#COMMON ENTITIES
USER = 'user'
CATEGORY = 'category'
PRODUCT = 'product'

RELATION_LIST = {
    ML1M: {
        0: "http://dbpedia.org/ontology/cinematography",
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
    CELL: {
        0: "belong_to",
        1: "also_buy_related_product",
        2: "also_buy_product",
        3: "produced_by_company",
        4: "also_view_related_product",
        5: "also_view_product",
    }
}
relation_name2entity_name = {
    ML1M: {
            "cinematographer_p_ci": 'cinematographer',
            "production_company_p_pr" :'production_company',
            "composer_p_co":'composer',
            "category_p_ca":'category',
            "actor_p_ac":'actor',
            "country_p_co":'country',
            "wikipage_p_wi":'wikipage',
            "editor_p_ed":'editor',
            "producer_p_pr":'producer',
            "writter_p_wr": 'writter',
            "director_p_di":'director',
        },
    LFM1M: {
        "category_p_ca": "category",
        "related_product_p_re": "related_product",
        "artist_p_ar": "artist",
        "engineer_p_en": "engineer",
        "producer_p_pr": "producer",
        "featured_artist_p_fe": "featured_artist",
    },
    CELL: {
        "category_p_ca": "category",
        "also_buy_related_product_p_re": "related_product",
        "also_buy_product_p_pr": "product",
        "brand_p_br": "brand",
        "also_view_related_product_p_re": "related_product",
        "also_view_product_p_pr": "product",
    }

}
relation_to_entity = {
    ML1M: {
        "http://dbpedia.org/ontology/cinematography": 'cinematographer',
        "http://dbpedia.org/property/productionCompanies": 'production_company',
        "http://dbpedia.org/property/composer": 'composer',
        "http://purl.org/dc/terms/subject": 'category',
        "http://dbpedia.org/ontology/starring": 'actor',
        "http://dbpedia.org/ontology/country": 'country',
        "http://dbpedia.org/ontology/wikiPageWikiLink": 'wikipage',
        "http://dbpedia.org/ontology/editing": 'editor',
        "http://dbpedia.org/property/producers": 'producer',
        "http://dbpedia.org/property/allWriting": 'writter',
        "http://dbpedia.org/ontology/director": 'director',
    },
    LFM1M: {
        "http://rdf.freebase.com/ns/common.topic.notable_types": "category",
        "http://rdf.freebase.com/ns/music.recording.releases": "related_product",
        "http://rdf.freebase.com/ns/music.recording.artist": "artist",
        "http://rdf.freebase.com/ns/music.recording.engineer": "engineer",
        "http://rdf.freebase.com/ns/music.recording.producer": "producer",
        "http://rdf.freebase.com/ns/music.recording.featured_artists": "featured_artist",
    },
    CELL: {
        "category": "category",
        "also_buy_related_product": "related_product",
        "also_buy_product": "product",
        "brand": "brand",
        "also_view_product": "product",
        "also_view_related_product": "related_product",
    }
}

relation_id2plain_name = {
    ML1M: {
        "0" : "cinematography_by",
        "1" : "produced_by_company",
        "2" : "composed_by",
        "3" : "belong_to",
        "10": "starred_by",
        "11": "produced_in",
        "12": "related_to",
        "14": "edited_by",
        "15": "produced_by_producer",
        "16": "wrote_by",
        "18": "directed_by",
    },
    LFM1M: {
        "0": "category",
        "1": "related_product",
        "2": "artist",
        "3": "engineer",
        "4": "producer",
        "5": "featured_artist",
    },
    CELL: {
        "0": "category",
        "1": "also_buy_related_product",
        "2": "related_product",
        "3": "brand",
        "4": "also_view_related_product",
        "5": "related_product"
    }
}

def write_time_based_train_test_split(dataset_name, model_name, train_size, valid_size=0, ratings_pid2local_id = {}, ratings_uid2global_id = {}):
    input_folder = f'data/{dataset_name}/preprocessed/'
    input_folder_kg = f'data/{dataset_name}/preprocessed/'
    output_folder = f'data/{dataset_name}/preprocessed/{model_name}/'

    ensure_dir(output_folder)

    uid2pids_timestamp_tuple = defaultdict(list)
    with open(input_folder + 'ratings.txt', 'r') as ratings_file: #uid	pid	rating	timestamp
        reader = csv.reader(ratings_file, delimiter="\t")
        next(reader, None)
        for row in reader:
            k, pid, rating, timestamp = row
            uid2pids_timestamp_tuple[k].append([pid, int(timestamp)])
    ratings_file.close()

    for k in uid2pids_timestamp_tuple.keys():
        uid2pids_timestamp_tuple[k].sort(key=lambda x: x[1])

    train_file = gzip.open(output_folder + 'train.txt.gz', 'wt')
    writer_train = csv.writer(train_file, delimiter="\t")
    valid_file = gzip.open(output_folder + 'valid_labels.txt.gz', 'wt')
    writer_valid = csv.writer(valid_file, delimiter="\t")
    test_file = gzip.open(output_folder + 'test.txt.gz', 'wt')
    writer_test = csv.writer(test_file, delimiter="\t")
    if model_name == "cafe":
        for k in uid2pids_timestamp_tuple.keys():
            uid = ratings_uid2global_id[k]
            curr = uid2pids_timestamp_tuple[k]
            n = len(curr)
            last_idx_train = int(n * train_size)
            pids_train = [ratings_pid2local_id[pid] for pid, timestamp in curr[:last_idx_train]]
            writer_train.writerow([uid, *pids_train])
            if valid_size != 0:
                last_idx_valid = last_idx_train + int(n * valid_size)
                pids_valid = [ratings_pid2local_id[pid] for pid, timestamp in curr[:last_idx_valid]]
                writer_valid.writerow([uid, *pids_valid])
            else:
                last_idx_valid = last_idx_train
            pids_test = [ratings_pid2local_id[pid] for pid, timestamp in curr[last_idx_valid:]]
            writer_test.writerow([uid, *pids_test])
    elif model_name == "pgpr":
        for k in uid2pids_timestamp_tuple.keys():
            curr = uid2pids_timestamp_tuple[k]
            n = len(curr)
            last_idx_train = int(n * train_size)
            pids_train = curr[:last_idx_train]
            for pid, timestamp in pids_train:
                writer_train.writerow([k, pid, 1, timestamp])
            if valid_size != 0:
                last_idx_valid = last_idx_train + int(n * valid_size)
                pids_valid = curr[:last_idx_valid]
                for pid, timestamp in pids_valid:
                    writer_valid.writerow([k, pid, 1, timestamp])
            else:
                last_idx_valid = last_idx_train
            pids_test = curr[last_idx_valid:]
            for pid, timestamp in pids_test:
                writer_test.writerow([k, pid, 1, timestamp])
    train_file.close()
    valid_file.close()
    test_file.close()

def get_time_based_train_test_split(dataset_name, model_name, train_size, valid_size=0):
    input_folder = f'data/{dataset_name}/preprocessed/'

    uid2pids_timestamp_tuple = defaultdict(list)
    with open(input_folder + 'ratings.txt', 'r') as ratings_file: #uid	pid	rating	timestamp
        reader = csv.reader(ratings_file, delimiter="\t")
        next(reader, None)
        for row in reader:
            uid, pid, rating, timestamp = row
            uid2pids_timestamp_tuple[uid].append([pid, int(timestamp)])
    ratings_file.close()

    for uid in uid2pids_timestamp_tuple.keys():
        uid2pids_timestamp_tuple[uid].sort(key=lambda x: x[1])

    train = defaultdict(list)
    valid = defaultdict(list)
    test = defaultdict(list)
    if model_name == "cafe":
        for uid in uid2pids_timestamp_tuple.keys():
            curr = uid2pids_timestamp_tuple[uid]
            n = len(curr)
            last_idx_train = int(n * train_size)
            pids_train = [pid for pid, timestamp in curr[:last_idx_train]]
            train[uid] = pids_train
            if valid_size != 0:
                last_idx_valid = last_idx_train + int(n * valid_size)
                pids_valid = [pid for pid, timestamp in curr[:last_idx_valid]]
                valid[uid] = pids_valid
            else:
                last_idx_valid = last_idx_train
            pids_test = [pid for pid, timestamp in curr[last_idx_valid:]]
            test[uid] = pids_test
    elif model_name == "pgpr":
        for uid in uid2pids_timestamp_tuple.keys():
            curr = uid2pids_timestamp_tuple[uid]
            n = len(curr)
            last_idx_train = int(n * train_size)
            pids_train = curr[:last_idx_train]
            for pid, timestamp in pids_train:
                train.append([uid, pid, 1, timestamp])
            if valid_size != 0:
                last_idx_valid = last_idx_train + int(n * valid_size)
                pids_valid = curr[:last_idx_valid]
                for pid, timestamp in pids_valid:
                    valid.append([uid, pid, 1, timestamp])
            else:
                last_idx_valid = last_idx_train
            pids_test = curr[last_idx_valid:]
            for pid, timestamp in pids_test:
                test.append([uid, pid, 1, timestamp])
    return train, valid, test

#def random_train_test_split(dataset_name, model_name):
#    input_folder = f'data/{dataset_name}/preprocessed/'
#    input_folder_kg = f'data/{dataset_name}/preprocessed/kg/'
#    output_folder = f'data/{dataset_name}/preprocessed/kg/{model_name}/'
#
#    if model_name == "cafe":
#        with open(input_folder + 'ratings.txt', 'r') as ratings_file:
#
#        ratings_file.close()

def map_to_CAFE(dataset_name, train_size, valid_size=0):
    input_folder = f'data/{dataset_name}/preprocessed/'
    input_folder_kg = f'data/{dataset_name}/preprocessed/'
    output_folder = f'data/{dataset_name}/preprocessed/cafe/'

    ensure_dir(output_folder)

    relation_id2entity = {}
    relation_id2relation_name = {}
    with open(input_folder_kg + 'r_map.txt', 'r') as relation_file:
        reader = csv.reader(relation_file, delimiter="\t")
        next(reader, None)
        for row in reader:
            relation_id, relation_url = int(row[0]), row[1]
            relation_id2entity[relation_id] = relation_to_entity[dataset_name][relation_url]
            relation_id2relation_name[relation_id] = relation_id2entity[relation_id] + f'_p_{relation_id2entity[relation_id][:2]}' #mapper a mano
    relation_file.close()

    kg_entity2org_dataset_id = {}
    org_dataset_id2kg_entity = {}
    with open(input_folder_kg + 'i2kg_map.txt', 'r') as item_to_kg_file:
        reader = csv.reader(item_to_kg_file, delimiter="\t")
        next(reader, None)
        for row in reader:
            dataset_id, entity_id = row[0], row[-1]
            kg_entity2org_dataset_id[entity_id] = dataset_id
            org_dataset_id2kg_entity[dataset_id] = entity_id
    item_to_kg_file.close()

    #org_dataset_id2kg_entity = dict(zip(kg_entity2org_dataset_id.values(), kg_entity2org_dataset_id.keys()))

    org_dataset_id2ratings_id = {}
    with open(f"data/{dataset_name}/preprocessed/products.txt", 'r') as item_to_kg_file:
        reader = csv.reader(item_to_kg_file, delimiter="\t")
        next(reader, None)
        for row in reader:
            new_id, dataset_id = row[0], row[1]
            org_dataset_id2ratings_id[dataset_id] = new_id
    item_to_kg_file.close()

    entity2name = defaultdict(set)
    with open(input_folder_kg + 'e_map.txt', 'r') as kg_file:
        reader = csv.reader(kg_file, delimiter="\t")
        next(reader, None)
        for row in reader:
            eid, entity_url = row[0], row[1]
            entity2name[eid] = (entity_url.split("/")[-1])
    kg_file.close()

    entity2kg_entity_list = defaultdict(set)
    with open(input_folder_kg + 'kg_final.txt', 'r') as kg_file:
        reader = csv.reader(kg_file, delimiter="\t")
        next(reader, None)
        for row in reader:
            entity_head, entity_tail, relation = row[0], row[1], row[2]
            entity2kg_entity_list[relation_id2entity[int(relation)]].add(entity_tail)
    kg_file.close()

    entity_type_eid2global_id = defaultdict(dict)
    ratings_pid2global_id = {}
    ratings_uid2global_id = {}
    #Write kg_entities
    with gzip.open(output_folder + 'kg_entities.txt.gz', 'wt') as entity_file:
        writer = csv.writer(entity_file, delimiter="\t")
        writer.writerow(['entity_global_id', 'entity_local_id', 'entity_value'])
        global_id = 0

        #Users
        with open(input_folder + 'users.txt', 'r') as user_file:
            reader = csv.reader(user_file, delimiter="\t")
            next(reader, None)
            for local_id, row in enumerate(reader):
                ratings_id, old_id = row[0], row[1]
                ratings_uid2global_id[ratings_id] = global_id
                writer.writerow([global_id, f"user_{local_id}", old_id]) #OLD OR NEW? OLD OK FOR KG, NEW OK FOR RATINGS
                global_id+=1
        user_file.close()

        #Products
        ratings_pid2local_id = {}
        with open(input_folder + 'products.txt', 'r') as product_file:
            reader = csv.reader(product_file, delimiter="\t")
            next(reader, None)
            for local_id, row in enumerate(reader):
                ratings_id, old_id = row[0], row[1]
                ratings_pid2global_id[ratings_id] = global_id
                ratings_pid2local_id[ratings_id] = local_id
                if dataset_name != CELL:
                    entity_type_eid2global_id['product'][org_dataset_id2kg_entity[old_id]] = global_id
                else:
                    entity_type_eid2global_id['product'][ratings_id] = global_id
                writer.writerow([global_id, f"product_{local_id}", entity2name[old_id]])
                global_id+=1
        product_file.close()

        #Entities
        for entity_name, entity_list in entity2kg_entity_list.items():
            for local_id, entity in enumerate(entity_list):
                if entity_name == "product":
                    continue
                writer.writerow([global_id, f"{entity_name}_{local_id}", entity2name[entity]])
                entity_type_eid2global_id[entity_name][entity] = global_id
                global_id+=1
    entity_file.close()

    #Relations
    rid2reverse = {}
    old_rid2new_id = {}
    with gzip.open(output_folder + 'kg_relations.txt.gz', 'wt') as relations_fileo:
        writer = csv.writer(relations_fileo, delimiter="\t")
        if dataset_name == ML1M:
            writer.writerow([0, "watched"])
        elif dataset_name == LFM1M:
            writer.writerow([0, "listened"])
        else:
            writer.writerow([0, "purchase"])
        with open(input_folder_kg + 'r_map.txt', 'r') as relations_file:
            reader = csv.reader(relations_file, delimiter="\t")
            next(reader, None)
            relation_names = []
            new_id = 1
            for row in reader:
                rid, r_url = row[0], row[1]
                old_rid2new_id[rid] = new_id
                relation_names.append(relation_id2plain_name[dataset_name][rid])
                new_id += 1
        relations_file.close()
        new_rid = 1
        #Write relations
        for new_name in relation_names:
            writer.writerow([new_rid, new_name])
            new_rid += 1
        if dataset_name == ML1M:
            main_rel = "watched"
        elif dataset_name == LFM1M:
            main_rel = "listened"
        else:
            main_rel = "purchase"
        writer.writerow([new_rid, f"rev_{main_rel}"])
        rid2reverse[0] = new_rid
        new_rid += 1
        #Write reverse relations
        i = 1
        for  new_name in relation_names:
            rid2reverse[i] = new_rid
            writer.writerow([new_rid, f"rev_{new_name}"])
            new_rid += 1
            i+=1
    relations_fileo.close()

    train, _, _ = get_time_based_train_test_split(dataset_name, "cafe", train_size, valid_size)
    with gzip.open(output_folder + 'kg_triples.txt.gz', 'wt') as kg_final_file:
        writer = csv.writer(kg_final_file, delimiter="\t")
        for uid, pids in train.items():
            for pid in pids:
                writer.writerow([ratings_uid2global_id[uid], 0, ratings_pid2global_id[pid]])
                writer.writerow([ratings_pid2global_id[pid], rid2reverse[0], ratings_uid2global_id[uid]])
        with open(input_folder_kg + "kg_final.txt") as triplets_file:
            reader = csv.reader(triplets_file, delimiter="\t")
            next(reader, None)
            for row in reader:
                entity_head, entity_tail, rid = row

                new_rid = old_rid2new_id[rid]
                new_entity_head = entity_type_eid2global_id['product'][entity_head]
                new_entity_tail = entity_type_eid2global_id[relation_id2entity[int(rid)]][entity_tail]
                writer.writerow([new_entity_head, new_rid, new_entity_tail])
                writer.writerow([new_entity_tail, rid2reverse[new_rid], new_entity_head])
        triplets_file.close()
    kg_final_file.close()
    write_time_based_train_test_split(dataset_name, "cafe", 0.8, 0, ratings_pid2local_id,
                                      ratings_uid2global_id)

    with gzip.open(output_folder + 'kg_rules.txt.gz', 'wt') as kg_rules_file:
        #rules must be defined by hand
        writer = csv.writer(kg_rules_file, delimiter="\t")
        main_relation = 0
        if dataset_name != "cellphones":
            for rid in rid2reverse.keys():
                if rid == 0:
                    writer.writerow([main_relation, rid2reverse[rid], rid])
                else:
                    writer.writerow([main_relation, rid, rid2reverse[rid]])
        else:
            for rid in rid2reverse.keys():
                if rid == 0:
                    writer.writerow([main_relation, rid2reverse[rid], rid])
                elif rid in [2, 5]:
                    writer.writerow([main_relation, rid2reverse[rid]])
                else:
                    writer.writerow([main_relation, rid, rid2reverse[rid]])
    kg_rules_file.close()

def map_to_PGPR(dataset_name):
    if dataset_name == CELL:
        map_to_PGPR_amazon(dataset_name)
        return
    input_folder = f'data/{dataset_name}/preprocessed/'
    input_folder_kg = f'data/{dataset_name}/preprocessed/'
    output_folder = f'data/{dataset_name}/preprocessed/pgpr/'

    ensure_dir(output_folder)

    relation_id2entity = {}
    relation_id2relation_name = {}
    with open(input_folder_kg + 'r_map.txt', 'r') as relation_file:
        reader = csv.reader(relation_file, delimiter="\t")
        next(reader, None)
        for row in reader:
            relation_id, relation_url = int(row[0]), row[1]
            relation_id2entity[relation_id] = relation_to_entity[dataset_name][relation_url]
            relation_id2relation_name[relation_id] = relation_id2entity[relation_id] + f'_p_{relation_id2entity[relation_id][:2]}'
    relation_file.close()

    entity_type_id2plain_name = defaultdict(dict)
    org_datasetid2movie_title = {}
    with open('data/ml1m/movies.dat', 'r', encoding="latin-1") as org_movies_file:
        reader = csv.reader(org_movies_file)
        next(reader, None)
        for row in reader:
            row = row[0].split("::")
            org_datasetid2movie_title[row[0]] = row[1]
    org_movies_file.close()

    entity2dataset_id = {}
    with open(input_folder_kg + 'i2kg_map.txt', 'r') as item_to_kg_file:
        reader = csv.reader(item_to_kg_file, delimiter="\t")
        next(reader, None)
        for row in reader:
            dataset_id, entity_id = row[0], row[-1]
            entity2dataset_id[entity_id] = dataset_id

    item_to_kg_file.close()

    dataset_id2new_id = {}
    with open(input_folder + "products.txt", 'r') as item_to_kg_file:
        reader = csv.reader(item_to_kg_file, delimiter="\t")
        next(reader, None)
        for row in reader:
            new_id, dataset_id = row[0], row[1]
            entity_type_id2plain_name["product"][new_id] = org_datasetid2movie_title[dataset_id]
            dataset_id2new_id[dataset_id] = new_id
    item_to_kg_file.close()

    triplets_groupby_entity = defaultdict(set)
    relation_pid_to_entity = {relation_file_name: defaultdict(list) for relation_file_name in relation_id2relation_name.values()}
    with open(input_folder_kg + 'kg_final.txt', 'r') as kg_file:
        reader = csv.reader(kg_file, delimiter="\t")
        next(reader, None)
        for row in reader:
            entity_head, entity_tail, relation = row[0], row[1], row[2]
            triplets_groupby_entity[relation_id2entity[int(relation)]].add(entity_tail)
            dataset_new_id = dataset_id2new_id[entity2dataset_id[entity_head]]
            relation_pid_to_entity[relation_id2relation_name[int(relation)]][dataset_new_id].append(entity_tail)
    kg_file.close()

    entity_to_entity_url = {}
    with open(input_folder_kg + 'e_map.txt', 'r') as entities_file:
        reader = csv.reader(entities_file, delimiter="\t")
        next(reader, None)
        for row in reader:
            entity_id, entity_url = row[0], row[1]
            entity_to_entity_url[entity_id] = entity_url
    entities_file.close()

    entity_id2new_id = defaultdict(dict)
    for entity_name, entity_list in triplets_groupby_entity.items():
        with gzip.open(output_folder + f'{entity_name}.txt.gz', 'wt') as entity_file:
            writer = csv.writer(entity_file, delimiter="\t")
            writer.writerow(['new_id', 'name'])
            for new_id, entity in enumerate(entity_list):
                writer.writerow([new_id, entity])
                if entity in entity_to_entity_url:
                    entity_type_id2plain_name[entity_name][new_id] = entity_to_entity_url[entity]
                else:
                    entity_type_id2plain_name[entity_name][new_id] = new_id
                entity_id2new_id[entity_name][entity] = new_id
        entity_file.close()

    with gzip.open(output_folder + f'mappings.txt.gz', 'wt') as mapping_file:
        writer = csv.writer(mapping_file, delimiter="\t")
        for entity_type, entities in entity_type_id2plain_name.items():
            for entity in entities:
                name = entity_type_id2plain_name[entity_type][entity]
                name = name.split("/")[-1] if type(name) == str else str(name)
                writer.writerow([f"{entity_type}_{entity}", name])
    mapping_file.close()

    for relation_name, items_list in relation_pid_to_entity.items():
        entity_name = relation_name2entity_name[dataset_name][relation_name]
        with gzip.open(output_folder + f'{relation_name}.txt.gz', 'wt') as relation_file:
            writer = csv.writer(relation_file, delimiter="\t")
            #writer.writerow(['new_id', 'name'])
            for i in range(len(dataset_id2new_id.keys())+1):
                entity_list = items_list[str(i)]
                entity_list_mapped = [entity_id2new_id[entity_name][entity_id] for entity_id in entity_list]
                writer.writerow(entity_list_mapped)
        relation_file.close()

    with gzip.open(output_folder + 'products.txt.gz', 'wt') as product_fileo:
        writer = csv.writer(product_fileo, delimiter="\t")
        with open(input_folder + 'products.txt', 'r') as product_file:
            reader = csv.reader(product_file, delimiter="\t")
            for row in reader:
                writer.writerow(row)
        product_file.close()
    product_fileo.close()

    with gzip.open(output_folder + 'users.txt.gz', 'wt') as users_fileo:
        writer = csv.writer(users_fileo, delimiter="\t")
        with open(input_folder + 'users.txt', 'r') as users_file:
            reader = csv.reader(users_file, delimiter="\t")
            for row in reader:
                writer.writerow(row)
        users_file.close()
    users_fileo.close()


def map_to_PGPR_amazon(dataset_name):
    input_folder = f'data/{dataset_name}/preprocessed/'
    input_folder_kg = f'data/{dataset_name}/preprocessed/'
    output_folder = f'data/{dataset_name}/preprocessed/pgpr/'

    ensure_dir(output_folder)

    relation_id2entity = {}
    relation_id2relation_name = {}
    with open(input_folder_kg + 'r_map.txt', 'r') as relation_file:
        reader = csv.reader(relation_file, delimiter="\t")
        next(reader, None)
        for row in reader:
            relation_id, relation_url = int(row[0]), row[1]
            relation_id2entity[relation_id] = relation_to_entity[dataset_name][relation_url]
            if relation_id in [1, 2, 4, 5]:
                relation_id2relation_name[relation_id] = relation_url + f'_p_{relation_id2entity[relation_id][:2]}'
            else:
                relation_id2relation_name[relation_id] = relation_id2entity[relation_id] + f'_p_{relation_id2entity[relation_id][:2]}'
    relation_file.close()

    entity2dataset_id = {}
    with open(input_folder_kg + 'i2kg_map.txt', 'r') as item_to_kg_file:
        reader = csv.reader(item_to_kg_file, delimiter="\t")
        next(reader, None)
        for row in reader:
            dataset_id, entity_id = row[0], row[-1]
            entity2dataset_id[entity_id] = dataset_id
    item_to_kg_file.close()

    dataset_id2new_id = {}
    with open(input_folder + "products.txt", 'r') as item_to_kg_file:
        reader = csv.reader(item_to_kg_file, delimiter="\t")
        next(reader, None)
        for row in reader:
            new_id, dataset_id = row[0], row[1]
            dataset_id2new_id[dataset_id] = new_id
    item_to_kg_file.close()

    triplets_groupby_entity = defaultdict(set)
    relation_pid_to_entity = {relation_file_name: defaultdict(list) for relation_file_name in relation_id2relation_name.values()}
    with open(input_folder_kg + 'kg_final.txt', 'r') as kg_file:
        reader = csv.reader(kg_file, delimiter="\t")
        next(reader, None)
        for row in reader:
            entity_head, entity_tail, relation = row[0], row[1], row[2]
            if relation == "1" or relation == "4":
                triplets_groupby_entity['related_product'].add(entity_tail)
                relation_pid_to_entity[relation_id2relation_name[int(relation)]][entity_head].append(entity_tail)
            elif relation == "1" or relation == "4":
                triplets_groupby_entity['product'].add(entity_tail)
                relation_pid_to_entity[relation_id2relation_name[int(relation)]][entity_head].append(entity_tail)
            else:
                triplets_groupby_entity[relation_id2entity[int(relation)]].add(entity_tail)
                relation_pid_to_entity[relation_id2relation_name[int(relation)]][entity_head].append(entity_tail)
    kg_file.close()

    entity_to_entity_url = {}
    with open(input_folder_kg + 'e_map.txt', 'r') as entities_file:
        reader = csv.reader(entities_file, delimiter="\t")
        next(reader, None)
        for row in reader:
            entity_id, entity_url = row[0], row[1]
            entity_to_entity_url[entity_id] = entity_url
    entities_file.close()

    entity_id2new_id = defaultdict(dict)
    for entity_name, entity_list in triplets_groupby_entity.items():
        if entity_name == "product":
            for new_id, entity in enumerate(set(entity_list)):
                entity_id2new_id[entity_name][entity] = new_id
            continue
        with gzip.open(output_folder + f'{entity_name}.txt.gz', 'wt') as entity_file:
            writer = csv.writer(entity_file, delimiter="\t")
            writer.writerow(['new_id', 'name'])
            for new_id, entity in enumerate(set(entity_list)):
                writer.writerow([new_id, entity])
                entity_id2new_id[entity_name][entity] = new_id
        entity_file.close()

    for relation_name, items_list in relation_pid_to_entity.items():
        entity_name = relation_name2entity_name[dataset_name][relation_name]
        #print(relation_name, entity_name)
        with gzip.open(output_folder + f'{relation_name}.txt.gz', 'wt') as relation_file:
            writer = csv.writer(relation_file, delimiter="\t")
            #writer.writerow(['new_id', 'name'])
            for i in range(len(dataset_id2new_id.keys())+1):
                entity_list = items_list[str(i)]
                entity_list_mapped = [entity_id2new_id[entity_name][entity_id] for entity_id in entity_list]
                writer.writerow(entity_list_mapped)
        relation_file.close()

    with gzip.open(output_folder + 'products.txt.gz', 'wt') as product_fileo:
        writer = csv.writer(product_fileo, delimiter="\t")
        with open(input_folder + 'products.txt', 'r') as product_file:
            reader = csv.reader(product_file, delimiter="\t")
            for row in reader:
                writer.writerow(row)
        product_file.close()
    product_fileo.close()

    with gzip.open(output_folder + 'users.txt.gz', 'wt') as users_fileo:
        writer = csv.writer(users_fileo, delimiter="\t")
        with open(input_folder + 'users.txt', 'r') as users_file:
            reader = csv.reader(users_file, delimiter="\t")
            for row in reader:
                writer.writerow(row)
        users_file.close()
    users_fileo.close()

map_to_PGPR(ML1M)
#map_to_CAFE(ML1M, 0.8)
#write_time_based_train_test_split(LFM1M, "pgpr", 0.8, 0)