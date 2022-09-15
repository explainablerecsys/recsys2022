from __future__ import absolute_import, division, print_function

from pgpr_utils import *


class KnowledgeGraph(object):

    def __init__(self, dataset):
        self.G = dict()
        self._load_entities(dataset)
        self.dataset_name = dataset.dataset_name
        self._load_reviews(dataset)
        self._load_knowledge(dataset)
        self._clean()
        self.top_matches = None


    def _load_entities(self, dataset):
        print('Load entities...')
        num_nodes = 0
        for entity in get_entities(dataset.dataset_name):
            self.G[entity] = {}
            vocab_size = getattr(dataset, entity).vocab_size
            for eid in range(vocab_size):
                relations = get_dataset_relations(dataset.dataset_name, entity)
                self.G[entity][eid] = {r: [] for r in relations}
            num_nodes += vocab_size
        print('Total {:d} nodes.'.format(num_nodes))

    def _load_reviews(self, dataset):
        print('Load reviews...')

        num_edges = 0
        for rid, data in enumerate(dataset.review.data):
            uid, pid, _, _ = data

            # (2) Add edges.
            main_product, main_interaction = MAIN_PRODUCT_INTERACTION[dataset.dataset_name]
            self._add_edge(USER, uid, main_interaction, main_product, pid)
            num_edges += 2

        print('Total {:d} review edges.'.format(num_edges))

    def _load_knowledge(self, dataset):
        relations = get_knowledge_derived_relations(dataset.dataset_name)
        main_entity, _ = MAIN_PRODUCT_INTERACTION[dataset.dataset_name]
        for relation in relations:
            print('Load knowledge {}...'.format(relation))
            data = getattr(dataset, relation).data
            num_edges = 0
            for pid, eids in enumerate(data):
                if len(eids) <= 0:
                    continue
                for eid in set(eids):
                    et_type = get_entity_tail(dataset.dataset_name, relation)
                    self._add_edge(main_entity, pid, relation, et_type, eid)
                    num_edges += 2
            print('Total {:d} {:s} edges.'.format(num_edges, relation))

    def _add_edge(self, etype1, eid1, relation, etype2, eid2):
        self.G[etype1][eid1][relation].append(eid2)
        self.G[etype2][eid2][relation].append(eid1)

    def _clean(self):
        print('Remove duplicates...')
        for etype in self.G:
            for eid in self.G[etype]:
                for r in self.G[etype][eid]:
                    data = self.G[etype][eid][r]
                    data = tuple(sorted(set(data)))
                    self.G[etype][eid][r] = data

    def compute_degrees(self):
        print('Compute node degrees...')
        self.degrees = {}
        self.max_degree = {}
        for etype in self.G:
            self.degrees[etype] = {}
            for eid in self.G[etype]:
                count = 0
                for r in self.G[etype][eid]:
                    count += len(self.G[etype][eid][r])
                self.degrees[etype][eid] = count

    def get(self, eh_type, eh_id=None, relation=None):
        data = self.G
        if eh_type is not None:
            data = data[eh_type]
        if eh_id is not None:
            data = data[eh_id]
        if relation is not None:
            data = data[relation]
        return data

    def __call__(self, eh_type, eh_id=None, relation=None):
        return self.get(eh_type, eh_id, relation)

    def get_tails(self, entity_type, entity_id, relation):
        return self.G[entity_type][entity_id][relation]

    '''
    def get_tails_given_user(self, entity_type, entity_id, relation, user_id):
        """ Very important!
        :param entity_type:
        :param entity_id:
        :param relation:
        :param user_id:
        :return:
        """
        tail_type = KG_RELATION[entity_type][relation]
        tail_ids = self.G[entity_type][entity_id][relation]
        if tail_type not in self.top_matches:
            return tail_ids
        top_match_set = set(self.top_matches[tail_type][user_id])
        top_k = len(top_match_set)
        if len(tail_ids) > top_k:
            tail_ids = top_match_set.intersection(tail_ids)
        return list(tail_ids)

    def trim_edges(self):
        degrees = {}
        for entity in self.G:
            degrees[entity] = {}
            for eid in self.G[entity]:
                for r in self.G[entity][eid]:
                    if r not in degrees[entity]:
                        degrees[entity][r] = []
                    degrees[entity][r].append(len(self.G[entity][eid][r]))

        for entity in degrees:
            for r in degrees[entity]:
                tmp = sorted(degrees[entity][r], reverse=True)
                print(entity, r, tmp[:10])

    def get_user_item_path_distribution(self, path_patter_name):
        path_pattern_degree = 0
        for (k, v) in self.degrees[path_patter_name]:
            path_pattern_degree += v
        return path_pattern_degree

    def get_total_path_pattern_number(self):
        path_pattern_degree = 0
        for (path_pattern, ) in self.degrees:
            for (k, v) in self.degrees[k]:
                path_pattern_degree += v
        return path_pattern_degree

    def set_top_matches(self, u_u_match, u_p_match, u_w_match):
        self.top_matches = {
            USER: u_u_match,
            MOVIE: u_p_match,
            #WORD: u_w_match,
        }


    def heuristic_search(self, uid, pid, pattern_id, trim_edges=False):
        if trim_edges and self.top_matches is None:
            raise Exception('To enable edge-trimming, must set top_matches of users first!')
        if trim_edges:
            _get = lambda e, i, r: self.get_tails_given_user(e, i, r, uid)
        else:
            _get = lambda e, i, r: self.get_tails(e, i, r)

        pattern = PATH_PATTERN[pattern_id]
        paths = []
        if pattern_id == 1:  # OK
            wids_u = set(_get(USER, uid, MENTION))  # USER->MENTION->WORD
            wids_p = set(_get(PRODUCT, pid, DESCRIBED_AS))  # PRODUCT->DESCRIBE->WORD
            intersect_nodes = wids_u.intersection(wids_p)
            paths = [(uid, x, pid) for x in intersect_nodes]
        elif pattern_id in [11, 12, 13, 14, 15, 16, 17]:
            pids_u = set(_get(USER, uid, PURCHASE))  # USER->PURCHASE->PRODUCT
            pids_u = pids_u.difference([pid])  # exclude target product
            nodes_p = set(_get(PRODUCT, pid, pattern[3][0]))  # PRODUCT->relation->node2
            if pattern[2][1] == USER:
                nodes_p.difference([uid])
            for pid_u in pids_u:
                relation, entity_tail = pattern[2][0], pattern[2][1]
                et_ids = set(_get(PRODUCT, pid_u, relation))  # USER->PURCHASE->PRODUCT->relation->node2
                intersect_nodes = et_ids.intersection(nodes_p)
                tmp_paths = [(uid, pid_u, x, pid) for x in intersect_nodes]
                paths.extend(tmp_paths)
        elif pattern_id == 18:
            wids_u = set(_get(USER, uid, MENTION))  # USER->MENTION->WORD
            uids_p = set(_get(PRODUCT, pid, PURCHASE))  # PRODUCT->PURCHASE->USER
            uids_p = uids_p.difference([uid])  # exclude source user
            for uid_p in uids_p:
                wids_u_p = set(_get(USER, uid_p, MENTION))  # PRODUCT->PURCHASE->USER->MENTION->WORD
                intersect_nodes = wids_u.intersection(wids_u_p)
                tmp_paths = [(uid, x, uid_p, pid) for x in intersect_nodes]
                paths.extend(tmp_paths)

        return paths

'''

def check_test_path(dataset_str, kg):
    # Check if there exists at least one path for any user-product in test set.
    test_user_products = load_labels(dataset_str, 'test')
    for uid in test_user_products:
        for pid in test_user_products[uid]:
            count = 0
            for pattern_id in [1, 11, 12, 13, 14, 15, 16, 17, 18]:
                tmp_path = kg.heuristic_search(uid, pid, pattern_id)
                count += len(tmp_path)
            if count == 0:
                print(uid, pid)

