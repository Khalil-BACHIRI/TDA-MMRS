import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--data_path', nargs='?', default='../data/',
                        help='Input data path.')
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed')
    parser.add_argument('--dataset', nargs='?', default='Baby',
                        help='Choose a dataset from {Baby, Musical_Instruments, Digital_Music}')
    parser.add_argument('--verbose', type=int, default=5,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=1000,
                        help='Number of epoch.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]',
                        help='Regularizations.')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate.')

    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--feat_embed_dim', type=int, default=64,
                        help='')                        
    parser.add_argument('--weight_size', nargs='?', default='[64,64]',
                        help='Output sizes of every layer')
    parser.add_argument('--core', type=int, default=5,
                        help='5-core for warm-start; 0-core for cold start')
    parser.add_argument('--topk', type=int, default=10,
                        help='K value of k-NN sparsification')  
    parser.add_argument('--lambda_coeff', type=float, default=0.9,
                        help='Lambda value of skip connection')
    parser.add_argument('--cf_model', nargs='?', default='lightgcn',
                        help='Downstream Collaborative Filtering model {mf, ngcf, lightgcn}')   
    parser.add_argument('--n_layers', type=int, default=1,
                        help='Number of item graph conv layers')  
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='') 
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU id')
    parser.add_argument('--Ks', nargs='?', default='[10, 20]',
                        help='K value of ndcg/recall @ k')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--imageTDA', type=bool, default=False,
                        help='Image TDA?')
    parser.add_argument('--textTDA', type=bool, default=False,
                        help='Text TDA?')
    parser.add_argument('--behaviorTDA', type=bool, default=False,
                        help='Text TDA?')
    parser.add_argument('--attrTDA', type=bool, default=False,
                        help='Text TDA?')
    parser.add_argument('--percentNodesDropped', type=int, default=25,
                        help='TDA drop nodes % dropped')
    parser.add_argument('--model_name', nargs='?', default='lattice',
                        help='Specify the model name: {lattice, lattice_tda_first_graph, lattice_tda_each_graph, lattice_tda_drop_nodes, mf, ngcf, lightgcn}')

    return parser.parse_args()
