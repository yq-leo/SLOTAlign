import json
import time

import torch

from GWLTorch import *
from utils import *
from distance import *
from dataset import *
import csv

random.seed(123)
torch.random.manual_seed(123)
np.random.seed(123)

torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None)
parser.add_argument('--dataset', type=str, default='dblp', help='douban/dblp/cora/citeseer/facebook/ppi')
parser.add_argument('--feat_noise', type=float, default=0.)
parser.add_argument('--noise_type', type=int, default=0, help='1: permutation, 2: truncation, 3: compression')
parser.add_argument('--edge_noise', type=float, default=0.)
parser.add_argument('--output', type=str, default='result.txt')
parser.add_argument('--step_size', type=float, default=1)
parser.add_argument('--bases', type=int, default=4)
parser.add_argument('--joint_epoch', type=int, default=100)
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--gw_beta', type=float, default=0.01)
parser.add_argument('--truncate', type=bool, default=False)
parser.add_argument('--plain', action='store_true', default=False)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--record', action='store_true', default=False)


args = parser.parse_args()
# if args.config:
#     f = open(args.config, 'r')
#     arg_dict = json.load(f)
#     for t in arg_dict:
#         args.__dict__[t] = arg_dict[t]
#     print(args)

nn_times_run_list = list()
ot_times_run_list = list()

for run in range(args.runs):
    print(f"Run {run + 1}/{args.runs}")
    # Aadj, Badj, Afeat, Bfeat, test_pairs, anchor_links, edge_index1, edge_index2 = myload_new(args.dataset, args.plain, args.edge_noise)
    Aadj, Badj, Afeat, Bfeat, test_pairs, anchor_links, edge_index1, edge_index2 = load_data_mat(f"data/scale_anchor/{args.dataset}")
    anchor1, anchor2 = anchor_links[:, 0], anchor_links[:, 1]
    G1, G2 = build_nx_graph(edge_index1, anchor1, Afeat), build_nx_graph(edge_index2, anchor2, Bfeat)
    rwr1, rwr2 = get_rwr_matrix(G1, G2, anchor_links, args.dataset, 0.2)

    if not args.plain:
        Afeat = np.concatenate([Afeat, rwr1], axis=1)
        Bfeat = np.concatenate([Bfeat, rwr2], axis=1)
    else:
        Afeat = rwr1
        Bfeat = rwr2
    Adim, Bdim = Afeat.shape[0], Bfeat.shape[0]
    Ag = dgl.graph(np.nonzero(Aadj), num_nodes=Adim)
    Bg = dgl.graph(np.nonzero(Badj), num_nodes=Bdim)
    Afeat -= Afeat.mean(0)
    Bfeat -= Bfeat.mean(0)

    if args.truncate:
        Afeat = Afeat[:,:100]
        Bfeat = Bfeat[:,:100]

    if args.noise_type == 1:
        Bfeat = feature_permutation(Bfeat, ratio=args.feat_noise)
    elif args.noise_type == 2:
        Bfeat = feature_truncation(Bfeat, ratio=args.feat_noise)
    elif args.noise_type == 3:
        Bfeat = feature_compression(Bfeat, ratio=args.feat_noise)

    print('feature size:', Afeat.shape, Bfeat.shape)

    Afeat = torch.tensor(Afeat).float()
    Bfeat = torch.tensor(Bfeat).float()

    time_st = time.time()
    layers = args.bases-2
    conv = GraphConv(0, 0, norm='both', weight=False, bias=False)
    Afeats = [torch.clone(Afeat)]
    Bfeats = [torch.clone(Bfeat)]
    Ag = Ag.to('cpu')
    Bg = Bg.to('cpu')
    for i in range(layers):
        Afeats.append(conv(dgl.add_self_loop(Ag), torch.clone(Afeats[-1])).detach().clone())
        Bfeats.append(conv(dgl.add_self_loop(Bg), torch.clone(Bfeats[-1])).detach().clone())

    Asims, Bsims = [Ag.adj().to_dense()], [Bg.adj().to_dense()]
    for i in range(len(Afeats)):
        Afeat = Afeats[i]
        Bfeat = Bfeats[i]
        Afeat = Afeat / (Afeat.norm(dim=1)[:, None]+1e-16)
        Bfeat = Bfeat / (Bfeat.norm(dim=1)[:, None]+1e-16)
        Asim = Afeat.mm(Afeat.T)
        Bsim = Bfeat.mm(Bfeat.T)
        Asims.append(Asim)
        Bsims.append(Bsim)


    Adim, Bdim = Afeat.shape[0], Bfeat.shape[0]
    a = torch.ones([Adim,1])/Adim
    b = torch.ones([Bdim,1])/Bdim
    X = a@b.T
    As = torch.stack(Asims, dim=2)
    Bs = torch.stack(Bsims, dim=2)

    alpha0 = np.ones(layers+2).astype(np.float32)/(layers+2)
    beta0 = np.ones(layers+2).astype(np.float32)/(layers+2)

    nn_time = list()
    for ii in range(args.joint_epoch):
        nn_start_time = time.time()
        alpha = torch.autograd.Variable(torch.tensor(alpha0))
        alpha.requires_grad = True
        beta = torch.autograd.Variable(torch.tensor(beta0))
        beta.requires_grad = True
        A = (As * alpha).sum(2)
        B = (Bs * beta).sum(2)
        objective = (A ** 2).mean() + (B ** 2).mean() - torch.trace(A @ X @ B @ X.T)
        alpha_grad = torch.autograd.grad(outputs=objective, inputs=alpha, retain_graph=True)[0]
        alpha = alpha - args.step_size * alpha_grad
        alpha0 = alpha.detach().cpu().numpy()
        alpha0 = euclidean_proj_simplex(alpha0)
        beta_grad = torch.autograd.grad(outputs=objective, inputs=beta)[0]
        beta = beta - args.step_size * beta_grad
        beta0 = beta.detach().cpu().numpy()
        beta0 = euclidean_proj_simplex(beta0)
        X = gw_torch(A.clone().detach(), B.clone().detach(), a, b, X.clone().detach(), beta=args.gw_beta,
                     outer_iter=1, inner_iter=50).clone().detach()
        print('Epoch:', ii, 'Objective:', objective.item())
        nn_end_time = time.time()
        nn_time.append(nn_end_time - nn_start_time)
        if ii == args.joint_epoch-1:
            ot_start_time = time.time()
            print(alpha0, beta0)
            X = gw_torch(A.clone().detach(), B.clone().detach(), a, b, X,
                         beta=args.gw_beta,
                         outer_iter=args.epoch-args.joint_epoch,
                         inner_iter=20,
                         gt=test_pairs)
            ot_end_time = time.time()

    time_ed = time.time()
    res=X.T.cpu().numpy()
    a1,a5,a10,a30 = compute_metrics(res, test_pairs)
    time_cost = time_ed - time_st

    nn_time = np.array(nn_time).mean()
    ot_time = ot_end_time - ot_start_time
    print('nn_time:', nn_time)
    print('ot_time:', ot_time)
    print('Total Time:', nn_time + ot_time)

    nn_times_run_list.append(nn_time)
    ot_times_run_list.append(ot_time)

    print('{} Edge Noise:{} Feat Noise:{} Type:{} Bases:{} GW beta:{} ss:{}, ep:{}'.format(
            args.dataset, args.edge_noise, args.feat_noise, args.noise_type, args.bases, args.gw_beta, args.step_size,  args.epoch))
    print('H@1 %.2f%% H@5 %.2f%% H@10 %.2f%% H@30 %.2f%% Time: %.2fs' % (a1 * 100, a5 * 100, a10 * 100, a30 * 100, time_cost))

final_nn_time = np.array(nn_times_run_list).mean()
final_ot_time = np.array(ot_times_run_list).mean()
final_total_time = final_nn_time + final_ot_time
if args.record:
    exp_name = "times"
    if not os.path.exists("results"):
        os.makedirs("results")
    out_file = f"results/{exp_name}.csv"
    if not os.path.exists(out_file):
        with open(out_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Dataset", "NN Time", "OT Time", "Total Time"])

    with open(out_file, "a", newline='') as f:
        writer = csv.writer(f)
        header = f"{args.dataset}"
        writer.writerow([header, f"{final_nn_time:.3f}", f"{final_ot_time:.3f}", f"{final_total_time:.3f}"])

