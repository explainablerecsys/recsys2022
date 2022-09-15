from collections import defaultdict, Counter

from utils import getDF
import csv

def propagate_item_removal_to_kg(ml1m_movies_df, movies_to_kg_df, entities_df, kg_df):
    movies_to_kg_df_after = movies_to_kg_df[movies_to_kg_df.dataset_id.isin(ml1m_movies_df.movie_id)]
    removed_movies = movies_to_kg_df[~movies_to_kg_df.dataset_id.isin(movies_to_kg_df_after.dataset_id)]
    print(f"Removed {removed_movies.shape[0]} entries from i2kg map.")
    removed_entities = entities_df[entities_df.entity_url.isin(removed_movies.entity_url)]
    print(f"Removed {removed_entities.shape[0]} entries from e_map")
    entities_df = entities_df[~entities_df.entity_url.isin(removed_movies.entity_url)]
    n_triplets = kg_df.shape[0]
    kg_df = kg_df[~kg_df.entity_head.isin(removed_entities.entity_id)]
    print(f"Removed {n_triplets - kg_df.shape[0]} triplets from kg_df")
    return movies_to_kg_df_after, entities_df, kg_df

def discard_entity_with_lt_th(entities_list, th):
    return [k for k, v in Counter(entities_list).items() if v >= th]

def discard_k_letter_categories(entities_list, k):
    return [x for x in entities_list if len(x) > k]

def create_kg_from_metadata(dataset):
    input_data = f'data/{dataset}/preprocessed'
    input_kg = f'data/{dataset}/kg'
    metaproduct_df = getDF(input_kg + '/meta_Cell_Phones_and_Accessories.json.gz')
    metaproduct_df = metaproduct_df.drop(['tech1', 'description', 'fit', 'title', 'tech2', 'feature', 'rank', 'details',
                                          'similar_item', 'date', 'price', 'imageURL', 'imageURLHighRes'], axis=1)

    valid_products = set()
    with open(input_data + '/products.txt', 'r') as products_file:
        reader = csv.reader(products_file, delimiter="\t")
        for row in reader:
            _, dataset_asin = row
            valid_products.add(dataset_asin)
    products_file.close()

    metaproduct_df = metaproduct_df[metaproduct_df.asin.isin(valid_products)]
    #Create i2kg.txt
    products_id = metaproduct_df['asin'].unique()
    product_id2new_id = {}
    entities = {}
    with open(input_data + "/i2kg_map.txt", 'w+') as fo:
        writer = csv.writer(fo, delimiter="\t")
        writer.writerow(["entity_id", "entity_url"])
        for new_id, pid in enumerate(products_id):
            product_id2new_id[pid] = new_id
            entities[pid] = new_id
            writer.writerow([new_id, pid])
    fo.close()

    columns = list(metaproduct_df.columns)
    columns.remove('asin')
    columns.remove('main_cat')
    relation_name2id = {}
    with open(input_data + "/r_map.txt", "w+") as fo:
        writer = csv.writer(fo, delimiter="\t")
        writer.writerow(["relation_id", "relation_url"])
        new_rid = 0
        for relation in columns:
            if relation == "also_buy" or relation == "also_view":
                relation_related_product = relation + "_related_product"
                writer.writerow([new_rid, relation_related_product])
                relation_name2id[relation_related_product] = new_rid
                new_rid += 1
                relation_product = relation + "_product"
                writer.writerow([new_rid, relation_product])
                relation_name2id[relation_product] = new_rid
                new_rid += 1
            else:
                writer.writerow([new_rid, relation])
                relation_name2id[relation] = new_rid
                new_rid += 1
    fo.close()

    #Create kg_final.txt and e_map.txt
    entity_names = set()
    for col in columns:
        if col == 'also_view':
            entity_name = 'related_product'
            entity_names.add(entity_name) #spaghetti
            entity_name = 'also_view_product'
            entity_names.add(entity_name)
        elif col == 'also_buy':
            entity_name = 'also_buy_product'
            entity_names.add(entity_name)
        else:
            entity_name = col
            entity_names.add(entity_name)

    last_id = len(entities)
    triplets = []
    for entity_name in entity_names:
        for _, row in metaproduct_df.iterrows():
            pid = row['asin']
            if entity_name == 'also_buy_product' or entity_name == 'also_view_product':
                relation = '_'.join(entity_name.split("_")[:2])
                related_products_in_catalog = [related_product for related_product in
                                               row[relation] if related_product in product_id2new_id]
                for product in related_products_in_catalog:
                    triplets.append([entities[pid], entities[product], relation_name2id[entity_name]])
            elif entity_name == 'related_product':
                for relation in ['also_buy', 'also_view']:
                    related_products_not_in_catalog = [related_product for related_product in row[relation] if
                                                       related_product not in product_id2new_id]
                    for related_product in related_products_not_in_catalog:
                        entities[related_product] = last_id
                        triplets.append([entities[pid], entities[related_product], relation_name2id[relation + f"_{entity_name}"]])
                        last_id += 1
            else:
                curr_attributes = row[entity_name]
                if curr_attributes == "": continue
                if type(curr_attributes) == list:
                    valid_entities = [value for value in curr_attributes if value not in entities]
                    for entity in valid_entities:
                        entities[entity] = last_id
                        triplets.append([entities[pid], entities[entity], relation_name2id[entity_name]])
                        last_id += 1
                else:
                    if curr_attributes not in entities:
                        entities[curr_attributes] = last_id
                        triplets.append([entities[pid], entities[curr_attributes], relation_name2id[entity_name]])
                        last_id += 1

    #Create e_map.txt
    with open(input_data + "/e_map.txt", 'w+') as fo:
        writer = csv.writer(fo, delimiter="\t")
        writer.writerow(["entity_id", "entity_url"])
        for entity_id, new_id in entities.items():
            writer.writerow([new_id, entity_id])
    fo.close()

    #Create kg_final.txt
    with open(input_data + "/kg_final.txt", 'w+') as fo:
        writer = csv.writer(fo, delimiter="\t")
        writer.writerow(["entity_head","entity_tail","relation"])
        for triple in triplets:
            e_h, e_t, r = triple
            triple = [e_h, e_t, r]
            writer.writerow(triple)
    fo.close()

create_kg_from_metadata("cellphones")