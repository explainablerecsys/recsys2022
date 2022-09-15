from __future__ import absolute_import, division, print_function

import numpy as np
import gzip
from easydict import EasyDict as edict
import random
from pgpr_utils import get_knowledge_derived_relations, DATASET_DIR, \
    get_pid_to_kgid_mapping, get_uid_to_kgid_mapping, get_entity_edict, ML1M, LFM1M, CELL


class Dataset(object):
    """This class is used to load data files and save in the instance."""

    def __init__(self, args, set_name='train', word_sampling_rate=1e-4):
        self.dataset_name = args.dataset
        self.data_dir = DATASET_DIR[self.dataset_name]
        if not self.data_dir.endswith('/'):
            self.data_dir += '/'
        self.review_file = set_name + '.txt.gz'
        self.load_entities()
        self.load_product_relations()
        self.load_reviews()

    def _load_file(self, filename):
        with gzip.open(self.data_dir + filename, 'r') as f:
            next(f, None)
            return [line.decode('utf-8').strip() for line in f]

    def load_entities(self):
        """Load 10 global entities from data files:
        'user','movie','actor','director','producer','production_company','category','editor','writter','cinematographer'
        Create a member variable for each entity associated with attributes:
        - `vocab`: a list of string indicating entity values.
        - `vocab_size`: vocabulary size.
        """
        entity_files = get_entity_edict(self.dataset_name)
        for name in entity_files:
            vocab = [x.split("\t")[0] for x in self._load_file(entity_files[name])]
            setattr(self, name, edict(vocab=vocab, vocab_size=len(vocab)+1))
            print('Load', name, 'of size', len(vocab))

    def load_reviews(self):
        """Load user-product reviews from train/test data files.
        Create member variable `review` associated with following attributes:
        - `data`: list of tuples (user_idx, product_idx, [word_idx...]).
        - `size`: number of reviews.
        - `product_distrib`: product vocab frequency among all eviews.
        - `product_uniform_distrib`: product vocab frequency (all 1's)
        - `word_distrib`: word vocab frequency among all reviews.
        - `review_count`: number of words (including duplicates).
        - `review_distrib`: always 1.
        """
        review_data = []  # (user_idx, product_idx, rating out of 5, timestamp)
        product_distrib = np.zeros(self.product.vocab_size)
        positive_reviews = 0
        negative_reviews = 0
        threshold = 3
        invalid_users = 0
        invalid_pid = 0
        for line in self._load_file(self.review_file):
            arr = line.split('\t')
            user_idx = int(arr[0])
            product_idx = int(arr[1])
            rating = int(arr[2])
            timestamp = int(arr[3])
            if rating >= threshold:
                positive_reviews+=1
            else:
                negative_reviews+=1
            review_data.append((user_idx, product_idx, rating, timestamp))
            product_distrib[product_idx] += 1
        print(invalid_users, invalid_pid)
        self.review = edict(
                data=review_data,
                size=len(review_data),
                product_distrib=product_distrib,
                product_uniform_distrib=np.ones(self.product.vocab_size),
                review_count=len(review_data),
                review_distrib=np.ones(len(review_data)) #set to 1 now
        )

        print('Load review of size', self.review.size, 'with positive reviews=',
              positive_reviews, ' and negative reviews=',
              negative_reviews)#, ' considered as positive the ratings >= of ', threshold)

    def load_product_relations(self):
        """Load 8 product -> ? relations:
        - 'directed_by': movie -> director
        - 'produced_by_company': movie->production_company,
        - 'produced_by_producer': movie->producer,
        - 'starring': movie->actor,
        - 'belong_to': movie->category,
        - 'edited_by': movie->editor,
        - 'written_by': movie->writter,
        - 'cinematography': movie->cinematographer,

        Create member variable for each relation associated with following attributes:
        - `data`: list of list of entity_tail indices (can be empty).
        - `et_vocab`: vocabulary of entity_tail (copy from entity vocab).
        - `et_distrib`: frequency of entity_tail vocab.
        """
        if self.dataset_name == ML1M:
            product_relations = edict(
                    directed_by=('director_p_di.txt.gz', self.director),
                    composed_by=('composer_p_co.txt.gz', self.composer),
                    produced_by_company=('production_company_p_pr.txt.gz', self.production_company),
                    produced_by_producer=('producer_p_pr.txt.gz', self.producer),
                    produced_in=('country_p_co.txt.gz', self.country),
                    starring=('actor_p_ac.txt.gz', self.actor),
                    belong_to=('category_p_ca.txt.gz', self.category),
                    edited_by=('editor_p_ed.txt.gz', self.editor),
                    wrote_by=('writter_p_wr.txt.gz', self.writter),
                    cinematography_by=('cinematographer_p_ci.txt.gz', self.cinematographer),
                    related_to=('wikipage_p_wi.txt.gz', self.wikipage)
            )
        elif self.dataset_name == LFM1M:
            product_relations = edict(
                mixed_by=("engineer_p_en.txt.gz", self.engineer),
                featured_by=("featured_artist_p_fe.txt.gz",self.featured_artist),
                sang_by=('artist_p_ar.txt.gz', self.artist),
                related_to=('related_product_p_re.txt.gz', self.related_product),
                belong_to=('category_p_ca.txt.gz', self.category),
                produced_by_producer=('producer_p_pr.txt.gz', self.producer),
            )
        elif self.dataset_name == CELL:
            product_relations = edict(
                also_bought_product=("also_buy_product_p_pr.txt.gz", self.product),
                also_bought_related_product=("also_buy_related_product_p_re.txt.gz", self.related_product),
                also_viewed_product=("also_view_product_p_pr.txt.gz", self.product),
                also_viewed_related_product=("also_view_related_product_p_re.txt.gz", self.related_product),
                belong_to=('category_p_ca.txt.gz', self.category),
                produced_by_company=('brand_p_br.txt.gz', self.brand),
            )

        for name in product_relations:
            # We save information of entity_tail (et) in each relation.
            # Note that `data` variable saves list of entity_tail indices.
            # The i-th record of `data` variable is the entity_tail idx (i.e. product_idx=i).
            # So for each product-relation, there are always |products| records.
            relation = edict(
                    data=[],
                    et_vocab=product_relations[name][1].vocab, #copy of brand, catgory ... 's vocab 
                    et_distrib= np.zeros(product_relations[name][1].vocab_size) #[1] means self.brand ..
            )
            size = 0
            for line in self._load_file(product_relations[name][0]): #[0] means brand_p_b.txt.gz ..
                knowledge = []
                line = line.split('\t')
                for x in line:  # some lines may be empty
                    if len(x) > 0:
                        x = int(x)
                        knowledge.append(x)
                        relation.et_distrib[x] += 1
                        size += 1
                relation.data.append(knowledge)
            setattr(self, name, relation)
            print('Load', name, 'of size', size)


class DataLoader(object):
    """This class acts as the dataloader for training knowledge graph embeddings."""

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.review_size = self.dataset.review.size
        self.product_relations = get_knowledge_derived_relations(dataset.dataset_name)
        self.finished_review_num = 0
        self.reset()

    def reset(self):
        # Shuffle reviews order
        self.review_seq = np.random.permutation(self.review_size)
        self.cur_review_i = 0
        self.cur_word_i = 0
        self._has_next = True

    def get_batch(self):
        """Return a matrix of [batch_size x n_relations], where each row contains
        (u_id, p_id, w_id, b_id, c_id, rp_id, rp_id, rp_id).
        """
        batch = []
        review_idx = self.review_seq[self.cur_review_i]
        user_idx, product_idx, rating, _ = self.dataset.review.data[review_idx]
        product_knowledge = {pr: getattr(self.dataset, pr).data[product_idx] for pr in self.product_relations} #DEFINES THE ORDER OF BATCH_IDX

        while len(batch) < self.batch_size:
            data = [user_idx, product_idx]
            for pr in self.product_relations:
                if len(product_knowledge[pr]) <= 0:
                    data.append(-1)
                else:
                    data.append(random.choice(product_knowledge[pr]))
            batch.append(data)

            self.cur_review_i += 1
            self.finished_review_num += 1
            if self.cur_review_i >= self.review_size:
                self._has_next = False
                break
            review_idx = self.review_seq[self.cur_review_i]
            user_idx, product_idx, rating, _ = self.dataset.review.data[review_idx]
            product_knowledge = {pr: getattr(self.dataset, pr).data[product_idx] for pr in self.product_relations}
        return np.array(batch)

    def has_next(self):
        """Has next batch."""
        return self._has_next

