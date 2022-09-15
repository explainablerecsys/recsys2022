from __future__ import absolute_import, division, print_function

import numpy as np
from tqdm import tqdm

from my_knowledge_graph import *
from cafe_utils import *


def load_kg_embedding(dataset: str):
    """Note that entity embedding is of size [vocab_size+1, d]."""
    print('>>> Load KG embeddings ...')
    state_dict = load_embed_sd(dataset)
    # print(state_dict.keys())
    embeds = dict()
    # Load entity embeddings
    for entity in ENTITY_LIST[dataset]:
        embeds[entity] = state_dict[entity + '.weight'].cpu().data.numpy()[:-1]   # remove last dummy embed with 0 values.
        print(f'>>> {entity}: {embeds[entity].shape}')
    for rel in RELATION_LIST[dataset]:
        embeds[rel] = (
            state_dict[rel].cpu().data.numpy()[0],
            state_dict[rel + '_bias.weight'].cpu().data.numpy()
        )
    return embeds


def compute_top100_items(dataset):
    embeds = load_embed(dataset)
    user_embed = embeds[USER]
    product_embed = embeds[PRODUCT]
    if dataset == ML1M:
        purchase_embed, purchase_bias = embeds[WATCHED]
    elif dataset == LFM1M:
        purchase_embed, purchase_bias = embeds[LISTENED]
    else:
        purchase_embed, purchase_bias = embeds[PURCHASE]
    scores = np.dot(user_embed + purchase_embed, product_embed.T)
    user_products = np.argsort(scores, axis=1)  # From worst to best
    best100 = user_products[:, -100:][:, ::-1]
    print(best100.shape)
    return best100


def estimate_path_count(args):
    kg = load_kg(args.dataset)
    num_mp = len(kg.metapaths)
    train_labels = load_labels(args.dataset, 'train')
    counts = {}
    pbar = tqdm(total=len(train_labels))
    for uid in train_labels:
        counts[uid] = np.zeros(num_mp)
        for pid in train_labels[uid]:
            for mpid in range(num_mp):
                cnt = kg.count_paths_with_target(mpid, uid, pid, 50)
                counts[uid][mpid] += cnt
        counts[uid] = counts[uid] / len(train_labels[uid])
        pbar.update(1)
    save_path_count(args.dataset, counts)


def main(args):
    # Run following code to extract embeddings from state dict.
    # ========== BEGIN ========== #
    embeds = load_kg_embedding(args.dataset)
    save_embed(args.dataset, embeds)
    # =========== END =========== #
    # Run following codes to generate MyKnowledgeGraph object.
    # ========== BEGIN ========== #
    kg = MyKnowledgeGraph(args.dataset)
    save_kg(args.dataset, kg)
    # =========== END =========== #

    # Run following codes to generate top100 items for each user.
    # ========== BEGIN ========== #
    best100 = compute_top100_items(args.dataset)
    save_user_products(args.dataset, best100, 'pos')
    # =========== END =========== #

    # Run following codes to estimate paths count.
    # ========== BEGIN ========== #
    estimate_path_count(args)
    # =========== END =========== #


if __name__ == '__main__':
    args = parse_args()
    main(args)
