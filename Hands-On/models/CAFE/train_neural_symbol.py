from __future__ import absolute_import, division, print_function

import os
import sys
import numpy as np
import logging
import logging.handlers
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import time

from my_knowledge_graph import *
from data_utils import OnlinePathLoader, OnlinePathLoaderWithMPSplit, KGMask
from symbolic_model import EntityEmbeddingModel, SymbolicNetwork, create_symbolic_model
from cafe_utils import *

logger = None


def set_logger(logname):
    global logger
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]  %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.handlers.RotatingFileHandler(logname, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def train(args):
    dataloader = OnlinePathLoader(args.dataset, args.batch_size, topk=args.topk_candidates)
    metapaths = dataloader.kg.metapaths

    kg_embeds = load_embed(args.dataset) if train else None
    model = create_symbolic_model(args, dataloader.kg, train=True, pretrain_embeds=kg_embeds)
    params = [name for name, param in model.named_parameters() if param.requires_grad]
    logger.info(f'Trainable parameters: {params}')
    logger.info('==================================')

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    total_steps = args.epochs * dataloader.total_steps
    steps = 0
    smooth_loss = []
    smooth_reg_loss = []
    smooth_rank_loss = []
    train_writer = SummaryWriter(args.log_dir)

    torch.save(model.state_dict(), '{}/symbolic_model_epoch{}.ckpt'.format(args.log_dir, 0))
    start_time = time.time()
    model.train()
    for epoch in range(1, args.epochs + 1):
        dataloader.reset()
        while dataloader.has_next():
            # Update learning rate
            lr = args.lr * max(1e-4, 1.0 - steps / total_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            # pos_paths: [bs, path_len], neg_paths: [bs, n, path_len]
            mpid, pos_paths, neg_pids = dataloader.get_batch()
            pos_paths = torch.from_numpy(pos_paths).to(args.device)
            neg_pids = torch.from_numpy(neg_pids).to(args.device)

            optimizer.zero_grad()
            reg_loss, rank_loss = model(metapaths[mpid], pos_paths, neg_pids)
            train_loss = reg_loss + args.rank_weight * rank_loss
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            smooth_loss.append(train_loss.item())
            smooth_reg_loss.append(reg_loss.item())
            smooth_rank_loss.append(rank_loss.item())

            if steps % args.steps_per_checkpoint == 0:
                smooth_loss = np.mean(smooth_loss)
                smooth_reg_loss = np.mean(smooth_reg_loss)
                smooth_rank_loss = np.mean(smooth_rank_loss)
                train_writer.add_scalar('train/smooth_loss', smooth_loss, steps)
                train_writer.add_scalar('train/smooth_reg_loss', smooth_reg_loss, steps)
                train_writer.add_scalar('train/smooth_rank_loss', smooth_rank_loss, steps)
                logger.info('Epoch/Step: {:02d}/{:08d} | '.format(epoch, steps) +
                            'LR: {:.5f} | '.format(lr) +
                            'Smooth Loss: {:.5f} | '.format(smooth_loss) +
                            'Reg Loss: {:.5f} | '.format(smooth_reg_loss) +
                            'Rank Loss: {:.5f} | '.format(smooth_rank_loss) +
                            'Executing for: {:.2f} | '.format((time.time() - start_time)))
                smooth_loss = []
                smooth_reg_loss = []
                smooth_rank_loss = []
            steps += 1
        if epoch % 10:
            torch.save(model.state_dict(), '{}/symbolic_model_epoch{}.ckpt'.format(args.log_dir, epoch))


def main():
    args = parse_args()
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)
    set_logger(args.log_dir + '/train_log.txt')
    logger.info(args)
    train(args)


if __name__ == '__main__':
    main()
