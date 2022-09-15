from __future__ import absolute_import, division, print_function

import random
import numpy as np
import gzip

from cafe_utils import DATA_DIR

ML1M = "ml1m"
LASTFM = "lfm1m"
CELL = "cellphones"

# ENTITY TYPES ML1M
PRODUCT = 'product'
ACTOR = 'actor'
DIRECTOR = 'director'
PRODUCER = 'producer'
PRODUCTION_COMPANY = 'production_company'
EDITOR = 'editor'
WRITTER = 'writter'
CINEMATOGRAPHER = 'cinematographer'
COMPOSER = 'composer'
COUNTRY = 'country'
USER = 'user'
CATEGORY = 'category'
WIKIPAGE = 'wikipage'
FEATURED_ARTIST = 'featured_artist'
BRAND = 'brand'
# ENTITY TYPES LASTFM
ARTIST = 'artist'
ENGINEER = 'engineer'
CATEGORY = 'category'
PRODUCER = 'producer'
RELATED_PRODUCT = 'related_product'
USER = 'user'

ENTITY_LIST = {
    ML1M: [PRODUCT, ACTOR, DIRECTOR, PRODUCTION_COMPANY, EDITOR, WRITTER, CINEMATOGRAPHER, COMPOSER, COUNTRY, USER, CATEGORY, PRODUCER, WIKIPAGE],
    LASTFM: [PRODUCT, FEATURED_ARTIST, ARTIST, ENGINEER, CATEGORY, PRODUCER, RELATED_PRODUCT, USER],
    CELL: [PRODUCT, BRAND, CATEGORY, RELATED_PRODUCT, USER],
}

# RELATIONS ML1M
BELONG_TO = 'belong_to'
PRODUCED_BY_PRODUCER = 'produced_by_producer'
WATCHED = 'watched'
DIRECTED_BY = 'directed_by'
PRODUCED_BY_COMPANY = 'produced_by_company'
STARRING = 'starring'
EDITED_BY = 'edited_by'
WROTE_BY = 'wrote_by'
CINEMATOGRAPHY_BY = 'cinematography_by'
COMPOSED_BY = 'composed_by'
PRODUCED_IN = 'produced_in'

# RELATIONS LASTFM
LISTENED = 'listened'
BELONG_TO = 'belong_to'
FEATURED_BY = 'featured_by'
MIXED_BY = 'mixed_by'
PRODUCED_BY = 'produced_by'
SANG_BY = 'sang_by'
RELATED_TO = 'related_to'
PURCHASE = 'purchase'
ALSO_BOUGHT_RP = 'also_bought_related_product'
ALSO_VIEWED_RP = 'also_viewed_related_product'
ALSO_BOUGHT_P = 'also_bought_product'
ALSO_VIEWED_P = 'also_viewed_product'

#RELATIONS CELL
PURCHASE = 'purchase'
RELATION_LIST = {
    ML1M: [BELONG_TO, PRODUCED_BY_PRODUCER, WATCHED, DIRECTED_BY, PRODUCED_BY_COMPANY, STARRING,
             EDITED_BY, WROTE_BY, CINEMATOGRAPHY_BY, COMPOSED_BY, PRODUCED_IN, RELATED_TO],
    LASTFM: [BELONG_TO, FEATURED_BY, MIXED_BY, PRODUCED_BY_PRODUCER, SANG_BY, RELATED_TO, LISTENED],
    CELL: [PURCHASE, BELONG_TO, PRODUCED_BY_COMPANY, ALSO_VIEWED_P, ALSO_BOUGHT_P, ALSO_BOUGHT_RP, ALSO_VIEWED_RP]
}
# REVERSED RELATIONS
REV_PREFIX = 'rev_'
REV_BELONG_TO = REV_PREFIX + 'belong_to'
REV_PRODUCED_BY_PRODUCER = REV_PREFIX + 'produced_by'
REV_WATCHED = REV_PREFIX + 'watched'
REV_DIRECTED_BY = REV_PREFIX + 'directed_by'
REV_PRODUCED_BY_COMPANY = REV_PREFIX + 'produced_by_company'
REV_STARRING = REV_PREFIX + 'starring'
REV_EDITED_BY = REV_PREFIX + 'edited_by'
REV_WROTE_BY = REV_PREFIX + 'wrote_by'
REV_CINEMATOGRAPHY = REV_PREFIX + 'cinematography'
REV_COMPOSED_BY = REV_PREFIX + 'composed_by'
REV_PRODUCED_IN = REV_PREFIX + 'produced_in'

# REV RELATIONS LASTFM
REV_BELONG_TO = REV_PREFIX + 'belong_to'
REV_FEATURED_BY = REV_PREFIX + 'featured_by'
REV_MIXED_BY = REV_PREFIX + 'mixed_by'
REV_PRODUCED_BY = REV_PREFIX + 'produced_by'
REV_SANG_BY = REV_PREFIX + 'sang_by'
REV_RELATED_TO = REV_PREFIX + 'related_to'


REV_PURCHASE = REV_PREFIX + 'purchase'
REV_ALSO_BOUGHT_RP = REV_PREFIX + 'also_bought_related_product'
REV_ALSO_VIEWED_RP = REV_PREFIX + 'also_viewed_related_product'
REV_ALSO_BOUGHT_P = REV_PREFIX + 'also_bought_product'
REV_ALSO_VIEWED_P = REV_PREFIX + 'also_viewed_product'

SELF_LOOP = 'self_loop'



class MyKnowledgeGraph:
    def __init__(self, dataset):
        self.G = dict()
        # Assume each relation correspnonds to a unique pair of (head_entity_type, tail_entity_type)!
        # E.g., purchase -> (user, item)
        # Otherwise, the KG cannot be built in this case.
        self.relation_info = dict()
        self.metapaths = list()
        self._init(dataset)

    def _init(self, dataset):
        # Load entities
        entity_file = DATA_DIR[dataset] + "/kg_entities.txt.gz"
        id2entity = {}
        num_entities = 0
        with gzip.open(entity_file, "rb") as f:
            for i, line in enumerate(f):
                # Format: [entity_global_id]\t[entity_type]_[entity_local_id]\t[entity_value]
                cells = line.decode().strip().split("\t")
                if i == 0:
                    continue
                global_id = int(cells[0])
                entity_eid = cells[1].rsplit("_", maxsplit=1)
                entity, eid = entity_eid[0], int(entity_eid[1])
                if entity not in self.G:
                    self.G[entity] = {}
                self.G[entity][eid] = {}
                id2entity[global_id] = (entity, eid)
                num_entities += 1
        print(f'>>> {num_entities} entities are loaded.')

        # Load relations
        relation_file = DATA_DIR[dataset] + "/kg_relations.txt.gz"
        id2rel = {}
        with gzip.open(relation_file, "rb") as f:
            for line in f:
                # Format: [relation_global_id]\t[relation_name]
                cells = line.decode().strip().split("\t")
                rid = int(cells[0])
                rel = cells[1]
                id2rel[rid] = rel
        print(f'>>> {len(id2rel)} relations are loaded.')

        # Load triples
        triple_file = DATA_DIR[dataset] + "/kg_triples.txt.gz"
        num_triples = 0
        invalid = 0
        with gzip.open(triple_file, "rb") as f:
            for line in f:
                # Format: [head_entity_global_id]\t[relation_global_id]\t[tail_entity_global_id]
                cells = line.decode().strip().split("\t")
                head_ent, hid = id2entity[int(cells[0])]
                rel = id2rel[int(cells[1])]
                tail_ent, tid = id2entity[int(cells[2])]

                # Validate if the assumption is correct!
                if rel not in self.relation_info:
                    self.relation_info[rel] = (head_ent, tail_ent)
                else:
                    assert self.relation_info[rel] == (head_ent, tail_ent)

                # Add edge.                
                if rel not in self.G[head_ent][hid]:
                    self.G[head_ent][hid][rel] = []
                self.G[head_ent][hid][rel].append(tid)
                num_triples += 1
        print(f'>>> Discarted {invalid} triplets\n {num_triples} triples are loaded (including reverse edges).')
        # Load rules
        rule_file = DATA_DIR[dataset] + "/kg_rules.txt.gz"
        with gzip.open(rule_file) as f:
            for line in f:
                cells = line.decode().strip().split("\t")
                mp = []
                for rid in cells:
                    rel = id2rel[int(rid)]
                    head_ent, tail_ent = self.relation_info[rel]
                    if not mp:
                        mp.append((None, head_ent))
                    else:
                        pass
                        #assert mp[-1][1] == head_ent
                    mp.append((rel, tail_ent))
                self.metapaths.append(mp)
        print(f'>>> {len(self.metapaths)} rules are loaded.')
        # print(self.metapaths)

    def __call__(self, eh_type, eh_id=None, relation=None):
        return self.get(eh_type, eh_id, relation)

    def get(self, eh_type, eh_id=None, relation=None):
        data = self.G
        if eh_type is not None:
            assert eh_type in data
            data = data[eh_type]
        if eh_id is not None:
            assert eh_id in data
            data = data[eh_id]
        if relation is not None:
            data = data[relation] if relation in data else []
        return data


    def sample_noise_path(self, metapath, user_id):
        path = [(None, USER, user_id)]
        for i in range(1, len(metapath)):
            next_relation, next_entity = metapath[i]
            vocab_size = len(self.G[next_entity])
            eid = random.choice(range(vocab_size))
            path.append((next_relation, next_entity, eid))
        return path

    def sample_paths(self, metapath, user_id, sample_sizes):
        """BFS with random sampling."""
        paths = [[(None, USER, user_id)]]
        for i in range(1, len(metapath)):
            next_relation, next_entity = metapath[i]
            tmp_paths = []
            for path in paths:
                _, last_entity, last_id = path[-1]
                next_ids = self.get(last_entity, last_id, next_relation)
                if len(next_ids) > sample_sizes[i - 1]:
                    next_ids = np.random.choice(next_ids, size=sample_sizes[i - 1], replace=False)
                for next_id in next_ids:
                    tmp_path = path + [(next_relation, next_entity, next_id)]
                    tmp_paths.append(tmp_path)
            paths = tmp_paths
        paths = [tuple(p) for p in paths]
        return paths

    def sample_paths_with_target(self, metapath, user_id, target_id, num_paths):
        """BFS from both sides."""
        path_len = len(metapath) - 1
        mid_level = int((path_len + 0) / 2)

        # Forward BFS.
        # t1 = time.time()
        forward_paths = [[(None, USER, user_id)]]
        for i in range(1, mid_level + 1):
            next_relation, next_entity = metapath[i]
            tmp_paths = []
            for fp in forward_paths:
                _, last_entity, last_id = fp[-1]
                next_ids = self.get(last_entity, last_id, next_relation)
                for next_id in next_ids:
                    tmp_path = fp + [(next_relation, next_entity, next_id)]
                    tmp_paths.append(tmp_path)
            forward_paths = tmp_paths

        # t2 = time.time()
        # print('---- FP ----')
        # print((t2-t1), len(forward_paths))
        # for i in range(len(forward_paths)):
        #    print(forward_paths[i])

        def _rev_rel(rel):
            if rel.startswith(REV_PREFIX):
                return rel[len(REV_PREFIX) :]
            return REV_PREFIX + rel

        # Backward BFS.
        # t1 = time.time()
        backward_paths = [[(metapath[-1][0], metapath[-1][1], target_id)]]
        for i in reversed(range(mid_level, path_len)):
            curr_relation, curr_entity = metapath[i]
            tmp_paths = []
            for bp in backward_paths:
                next_relation, next_entity, next_id = bp[0]
                curr_ids = self.get(next_entity, next_id, _rev_rel(next_relation))
                for curr_id in curr_ids:
                    tmp_path = [(curr_relation, curr_entity, curr_id)] + bp
                    tmp_paths.append(tmp_path)
            backward_paths = tmp_paths
        # t2 = time.time()
        # print('---- BP ----')
        # print((t2-t1), len(backward_paths))
        # for i in range(len(backward_paths)):
        #    print(backward_paths[i])

        # Find intersection paths.
        final_paths = []
        backward_map = {}
        for bp in backward_paths:
            backward_map[bp[0]] = bp[1:]
        for fp in forward_paths:
            if fp[-1] in backward_map:
                final_paths.append(tuple(fp + backward_map[fp[-1]]))
        if len(final_paths) > num_paths:
            # final_paths = np.random.choice(final_paths, size=num_paths)
            final_paths = random.sample(final_paths, num_paths)
        return final_paths

    def fast_sample_path_with_target(self, mpath_id, user_id, target_id, num_paths, sample_size=100):
        """Sample one path given source and target, using BFS from both sides.
        Returns:
            list of entity ids.
        """
        metapath = self.metapaths[mpath_id]
        path_len = len(metapath) - 1
        mid_level = int((path_len + 0) / 2)

        # Forward BFS (e.g. u--e1--e2--e3).
        forward_paths = [[user_id]]
        for i in range(1, mid_level + 1):
            _, last_entity = metapath[i - 1]
            next_relation, _ = metapath[i]
            tmp_paths = []
            for fp in forward_paths:
                next_ids = self.get(last_entity, fp[-1], next_relation)
                # Random sample ids
                if len(next_ids) > sample_size:
                    # next_ids = np.random.permutation(next_ids)[:sample_size]
                    next_ids = np.random.choice(next_ids, size=sample_size, replace=False)
                for next_id in next_ids:
                    tmp_paths.append(fp + [next_id])
            forward_paths = tmp_paths

        def _rev_rel(rel):
            if rel.startswith(REV_PREFIX):
                return rel[len(REV_PREFIX) :]
            return REV_PREFIX + rel

        # Backward BFS (e.g. e4--e5--e6).
        backward_paths = [[target_id]]
        for i in reversed(range(mid_level + 2, path_len + 1)):  # i=l, l-2,..., mid+2
            next_relation, next_entity = metapath[i]
            tmp_paths = []
            for bp in backward_paths:
                # print(next_entity, bp[0], next_relation)
                curr_ids = self.get(next_entity, bp[0], _rev_rel(next_relation))
                # Random sample ids
                if len(curr_ids) > sample_size:
                    # curr_ids = np.random.permutation(curr_ids)[:sample_size]
                    curr_ids = np.random.choice(curr_ids, size=sample_size, replace=False)
                for curr_id in curr_ids:
                    tmp_paths.append([curr_id] + bp)
            backward_paths = tmp_paths

        # Build hash map for indexing backward paths.
        # e.g. a dict with key=e3 and value=(e4--e5--e6).
        backward_map = {}
        next_relation, next_entity = metapath[mid_level + 1]
        for bp in backward_paths:
            curr_ids = self.get(next_entity, bp[0], _rev_rel(next_relation))
            if len(curr_ids) > sample_size:
                curr_ids = np.random.choice(curr_ids, size=sample_size, replace=False)
            for curr_id in curr_ids:
                if curr_id not in backward_map:
                    backward_map[curr_id] = []
                backward_map[curr_id].append(bp)

        # Find intersection of forward paths and backward paths.
        final_paths = []
        for fp_idx in np.random.permutation(len(forward_paths)):
            fp = forward_paths[fp_idx]
            mid_id = fp[-1]
            if mid_id not in backward_map:
                continue
            np.random.shuffle(backward_map[mid_id])
            for bp in backward_map[mid_id]:
                final_paths.append(fp + bp)
                if len(final_paths) >= num_paths:
                    break
            if len(final_paths) >= num_paths:
                break
        return final_paths

    def count_paths_with_target(self, mpath_id, user_id, target_id, sample_size=50):
        """This is an approx count, not exact."""
        metapath = self.metapaths[mpath_id]
        path_len = len(metapath) - 1
        mid_level = int((path_len + 0) / 2)

        # Forward BFS (e.g. u--e1--e2--e3).
        forward_ids = [user_id]
        for i in range(1, mid_level + 1):  # i=1, 2,..., mid
            _, last_entity = metapath[i - 1]
            next_relation, _ = metapath[i]
            tmp_ids = []
            for eid in forward_ids:
                next_ids = self.get(last_entity, eid, next_relation)
                if len(next_ids) > sample_size:
                    next_ids = np.random.choice(next_ids, size=sample_size, replace=False).tolist()
                tmp_ids.extend(next_ids)
            forward_ids = tmp_ids
        # cnt = len([_ for i in forward_ids if i == target_id])
        # return cnt
        # forward_ids, forward_counts = np.unique(forward_ids, return_counts=True)
        # print(forward_ids, forward_counts)

        def _rev_rel(rel):
            if rel.startswith(REV_PREFIX):
                return rel[len(REV_PREFIX) :]
            return REV_PREFIX + rel

        # Backward BFS (e.g. e4--e5--e6).
        backward_ids = [target_id]
        for i in reversed(range(mid_level + 1, path_len + 1)):  # i=l, l-1,..., mid+1
            next_relation, next_entity = metapath[i]
            tmp_ids = []
            for eid in backward_ids:
                curr_ids = self.get(next_entity, eid, _rev_rel(next_relation))
                tmp_ids.extend(curr_ids)
            backward_ids = tmp_ids
        # backward_ids, backward_counts = np.unique(backward_ids, return_counts=True)
        # print(backward_ids, backward_counts)

        count = len(set(forward_ids).intersection(backward_ids))
        return count

    def is_valid_path(self, mpid, path):
        metapath = self.metapaths[mpid]
        if len(metapath) != len(path):
            return False
        graph = self.get(metapath[0][1])
        if path[0] not in graph:
            return False
        last_node = (metapath[0][1], path[0])
        for i in range(1, len(path)):
            relation, entity = metapath[i]
            node_id = path[i]
            candidates = self.get(last_node[0], last_node[1], relation)
            if node_id not in candidates:
                return False
            last_node = (entity, node_id)
        return True
