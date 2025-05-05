import datetime
import math
import os
import random
import sys
from time import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.sparse as sparse

from utility.parser import parse_args
from Models import LATTICE, MF, LightGCN, NGCF, LATTICE_TDA_first_graph, LATTICE_TDA_each_graph, LATTICE_TDA_drop_nodes
from utility.batch_test import *

args = parse_args()

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

class Trainer(object):
    def __init__(self, data_config):
        # argument settings
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.model_name = args.model_name
        self.mess_dropout = eval(args.mess_dropout)
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.weight_size)
        self.n_layers = len(self.weight_size)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]

        self.imageTDA = args.imageTDA
        self.textTDA = args.textTDA
        self.behaviorTDA = args.behaviorTDA
        self.attrTDA = args.attrTDA

        print(self.imageTDA, self.textTDA, self.behaviorTDA, self.attrTDA, self.model_name,  args.percentNodesDropped)

        if(self.imageTDA):
            image_feats = np.load('../data/{}/image_feat_TDA.npy'.format(args.dataset))
        else:
            image_feats = np.load('../data/{}/image_feat.npy'.format(args.dataset))
        if(self.textTDA):
            text_feats = np.load('../data/{}/text_feat_TDA.npy'.format(args.dataset))
        else:
            text_feats = np.load('../data/{}/text_feat.npy'.format(args.dataset))

        if(self.behaviorTDA):
            behavior_feats = np.load('../data/{}/behavior_feat_TDA.npy'.format(args.dataset))
    b   else:
            behavior_feats = np.load('../data/{}/behavior_feat.npy'.format(args.dataset))
        if(self.attrTDA):
            attr_feats = np.load('../data/{}/attributes_feat_TDA.npy'.format(args.dataset))
        else:
            attr_feats = np.load('../data/{}/attributes_feat.npy'.format(args.dataset))

        if (self.model_name == 'lattice_tda_drop_nodes'):
            self.norm_adj = data_config['norm_adj']
            self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(device)

            self.model = LATTICE_TDA_drop_nodes(self.n_users, self.n_items, self.emb_dim, self.weight_size,
                                             self.mess_dropout, image_feats, text_feats, self.norm_adj, data_generator,  args.percentNodesDropped)
            self.norm_adj = self.model.norm_adj
            self.n_items = self.n_items - image_feats.shape[0] * args.percentNodesDropped // 100
        else:

            self.norm_adj = data_config['norm_adj']
            self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(device)

            if(self.model_name == 'lattice'):
                self.model = LATTICE(self.n_users, self.n_items, self.emb_dim, self.weight_size, self.mess_dropout, image_feats, text_feats)
            elif(self.model_name == 'lattice_tda_first_graph'):
                self.model = LATTICE_TDA_first_graph(self.n_users, self.n_items, self.emb_dim, self.weight_size, self.mess_dropout, image_feats, text_feats)
            elif(self.model_name == 'lattice_tda_each_graph'):
                self.model = LATTICE_TDA_each_graph(self.n_users, self.n_items, self.emb_dim, self.weight_size, self.mess_dropout, image_feats, text_feats)
            elif(self.model_name == 'mf'):
                self.model = MF(self.n_users, self.n_items, self.emb_dim, self.weight_size, self.mess_dropout,
                                     image_feats, text_feats)
            elif (self.model_name == 'lightgcn'):
                self.model = LightGCN(self.n_users, self.n_items, self.emb_dim, self.weight_size, self.mess_dropout,
                                image_feats, text_feats)
            elif (self.model_name == 'ngcf'):
                self.model = NGCF(self.n_users, self.n_items, self.emb_dim, self.weight_size, self.mess_dropout,
                                image_feats, text_feats)
            else:
                raise Exception("Invalid parameter; choose between {lattice, lattice_tda_first_graph, "
                                "lattice_tda_each_graph, lattice_tda_drop_nodes, mf, ngcf, lightgcn}")



        self.model = self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = self.set_lr_scheduler()

    def set_lr_scheduler(self):
        fac = lambda epoch: 0.96 ** (epoch / 50)
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        return scheduler

    def test(self, users_to_test, is_val):
        self.model.eval()
        with torch.no_grad():
            ua_embeddings, ia_embeddings = self.model(self.norm_adj, build_item_graph=True)
        result = test_torch(ua_embeddings, ia_embeddings, users_to_test, is_val, n_items=self.n_items)
        return result

    def train(self):
        training_time_list = []
        loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
        stopping_step = 0
        should_stop = False
        cur_best_pre_0 = 0.

        n_batch = data_generator.n_train // args.batch_size + 1
        best_recall = 0

        k = args.epoch
        for epoch in (range(k)):
            t1 = time()
            loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
            n_batch = data_generator.n_train // args.batch_size + 1
            f_time, b_time, loss_time, opt_time, clip_time, emb_time = 0., 0., 0., 0., 0., 0.
            sample_time = 0.
            build_item_graph = True

            for idx in (range(n_batch)):
                self.model.train()
                self.optimizer.zero_grad()
                sample_t1 = time()
                #sample
                users, pos_items, neg_items = data_generator.sample()
                sample_time += time() - sample_t1
                # la primera iteracio calcula tot, despres nomes la h
                ua_embeddings, ia_embeddings = self.model(self.norm_adj, build_item_graph=build_item_graph)
                build_item_graph = False
                build_item_graph = (idx == 0) or (args.recompute_graph_every > 0 and epoch % args.recompute_graph_every == 0)

                u_g_embeddings = ua_embeddings[users]

                pos_i_g_embeddings = ia_embeddings[pos_items]
                neg_i_g_embeddings = ia_embeddings[neg_items]

                # bpr
                batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                              neg_i_g_embeddings)

                batch_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss

                batch_loss.backward(retain_graph=True)
                self.optimizer.step()

                loss += float(batch_loss)
                mf_loss += float(batch_mf_loss)
                emb_loss += float(batch_emb_loss)
                reg_loss += float(batch_reg_loss)


            self.lr_scheduler.step()

            del ua_embeddings, ia_embeddings, u_g_embeddings, neg_i_g_embeddings, pos_i_g_embeddings

            if math.isnan(loss) == True:
                print('ERROR: loss is nan.')
                sys.exit()

            perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                epoch, time() - t1, loss, mf_loss, emb_loss)
            training_time_list.append(time() - t1)
            print(perf_str)

            if epoch % args.verbose != 0:
                continue


            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            users_to_val = list(data_generator.val_set.keys())
            ret = self.test(users_to_val, is_val=True)
            training_time_list.append(t2 - t1)

            t3 = time()

            loss_loger.append(loss)
            rec_loger.append(ret['recall'])
            pre_loger.append(ret['precision'])
            ndcg_loger.append(ret['ndcg'])
            hit_loger.append(ret['hit_ratio'])
            if args.verbose > 0:
                perf_str = 'Epoch %d [%.1fs + %.1fs]:  val==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f], ' \
                           'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                           (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss, ret['recall'][0],
                            ret['recall'][-1],
                            ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                            ret['ndcg'][0], ret['ndcg'][-1])
                print(perf_str)

            if ret['recall'][1] > best_recall:
                best_recall = ret['recall'][1]
                test_ret = self.test(users_to_test, is_val=False)
                perf_str = 'Epoch %d [%.1fs + %.1fs]: test==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f], ' \
                           'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                           (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss, test_ret['recall'][0],
                            test_ret['recall'][-1],
                            test_ret['precision'][0], test_ret['precision'][-1], test_ret['hit_ratio'][0], test_ret['hit_ratio'][-1],
                            test_ret['ndcg'][0], test_ret['ndcg'][-1])
                print(perf_str)                
                stopping_step = 0
            elif stopping_step < args.early_stopping_patience:
                stopping_step += 1
                print('#####Early stopping steps: %d #####' % stopping_step)
            else:
                print('#####Early stop! #####')
                break

        print(test_ret)

        return "./logs/" + str(self.model_name) + str(args.percentNodesDropped)+ "-" + str(args.dataset) + "-" + str(self.textTDA) + "-" + str(self.imageTDA), "Epoch: " + str(epoch) + "\nResult: " + str(test_ret)

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1./2*(users**2).sum() + 1./2*(pos_items**2).sum() + 1./2*(neg_items**2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.decay * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

def set_seed(seed, reproducibility=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

if __name__ == '__main__':
    start_time = time()

    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()
    config['norm_adj'] = norm_adj

    trainer = Trainer(data_config=config)
    model, aux = trainer.train()

    stop_time = time()

    model_id = f"{args.model_name}_drop{args.percentNodesDropped}_textTDA{args.textTDA}_imageTDA{args.imageTDA}_behaviorTDA{args.behaviorTDA}_attrTDA{args.attrTDA}_{args.dataset}"
    file_path = f"./logs/{model_id}_result_{int(time())}.txt"


    with open(file_path, 'w') as file:
        file.write(str(aux))
        file.write("\nElapsed time: " + str(stop_time - start_time))

'''

python main.py --dataset Musical_Instruments --model lattice_tda_drop_nodes --percentNodesDropped 1
python main.py --dataset Musical_Instruments --model lattice_tda_drop_nodes --percentNodesDropped 2

python main.py --dataset Musical_Instruments --model lattice_tda_each_graph

python main.py --dataset Baby --model lattice_tda_each_graph
python main.py --dataset Digital_Music --model lattice_tda_each_graph
python main.py --dataset Digital_Music --model lattice_tda_each_graph
python main.py --dataset Digital_Music --model lattice_tda_each_graph

'''