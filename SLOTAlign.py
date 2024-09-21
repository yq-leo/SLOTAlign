import json
import time
from collections import defaultdict

from GWLTorch import *
from utils import *
from distance import *
from dataset import *
import csv

random.seed(123)
torch.random.manual_seed(123)
np.random.seed(123)

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
# Experiment settings
parser.add_argument('--runs', dest='runs', type=int, default=1, help='number of runs')
parser.add_argument('--edge_noise_rate', type=float, default=0.0, help='edge noise rate')
parser.add_argument('--attr_noise_rate', type=float, default=0.0, help='attribute noise rate')
parser.add_argument('--exp_name', type=str, default='exp', help='experiment name')
parser.add_argument('--record', action='store_true', default=False, help='record results')
parser.add_argument('--robust', action='store_true', default=False, help='remove metric outliers')
parser.add_argument('--strong_noise', action='store_true', default=False, help='use strong attribute noise')

args = parser.parse_args()
assert args.edge_noise == 0 and args.feat_noise == 0, 'Default noise should not be set'
# if args.config:
#     f = open(args.config, 'r')
#     arg_dict = json.load(f)
#     for t in arg_dict:
#         args.__dict__[t] = arg_dict[t]
#     print(args)

final_hits_list = defaultdict(list)
final_mrr_list = []
for run in range(args.runs):
    print(f"Run {run + 1}/{args.runs}")

    Aadj, Badj, Afeat, Bfeat, test_pairs, anchor_links, edge_index1, edge_index2 = myload_new(args.dataset, args.plain,
                                                                                              args.edge_noise)
    anchor1, anchor2 = anchor_links[:, 0], anchor_links[:, 1]
    G1, G2 = build_nx_graph(edge_index1, anchor1, Afeat), build_nx_graph(edge_index2, anchor2, Bfeat)
    rwr1, rwr2 = get_rwr_matrix(G1, G2, anchor_links, args.dataset, 0.2)

    if not args.plain:
        Afeat = np.concatenate([Afeat, rwr1], axis=1)
        Bfeat = perturb_attr(Bfeat, args.attr_noise_rate, strong_noise=args.strong_noise)
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
        Afeat = Afeat[:, :100]
        Bfeat = Bfeat[:, :100]

    if args.noise_type == 1:
        Bfeat = feature_permutation(Bfeat, ratio=args.feat_noise)
    elif args.noise_type == 2:
        Bfeat = feature_truncation(Bfeat, ratio=args.feat_noise)
    elif args.noise_type == 3:
        Bfeat = feature_compression(Bfeat, ratio=args.feat_noise)

    print('feature size:', Afeat.shape, Bfeat.shape)

    Afeat = torch.tensor(Afeat).float().cuda()
    Bfeat = torch.tensor(Bfeat).float().cuda()

    time_st = time.time()
    layers = args.bases - 2
    conv = GraphConv(0, 0, norm='both', weight=False, bias=False)
    Afeats = [torch.clone(Afeat)]
    Bfeats = [torch.clone(Bfeat)]
    Ag = Ag.to('cuda:0')
    Bg = Bg.to('cuda:0')
    for i in range(layers):
        Afeats.append(conv(dgl.add_self_loop(Ag), torch.clone(Afeats[-1])).detach().clone())
        Bfeats.append(conv(dgl.add_self_loop(Bg), torch.clone(Bfeats[-1])).detach().clone())

    Asims, Bsims = [Ag.adj().to_dense().cuda()], [Bg.adj().to_dense().cuda()]
    for i in range(len(Afeats)):
        Afeat = Afeats[i]
        Bfeat = Bfeats[i]
        Afeat = Afeat / (Afeat.norm(dim=1)[:, None] + 1e-16)
        Bfeat = Bfeat / (Bfeat.norm(dim=1)[:, None] + 1e-16)
        Asim = Afeat.mm(Afeat.T)
        Bsim = Bfeat.mm(Bfeat.T)
        Asims.append(Asim)
        Bsims.append(Bsim)

    Adim, Bdim = Afeat.shape[0], Bfeat.shape[0]
    a = torch.ones([Adim, 1]).cuda() / Adim
    b = torch.ones([Bdim, 1]).cuda() / Bdim
    X = a @ b.T
    As = torch.stack(Asims, dim=2)
    Bs = torch.stack(Bsims, dim=2)

    alpha0 = np.ones(layers + 2).astype(np.float32) / (layers + 2)
    beta0 = np.ones(layers + 2).astype(np.float32) / (layers + 2)
    hits_k_max, mrr_max = defaultdict(int), 0
    for ii in range(args.joint_epoch):
        alpha = torch.autograd.Variable(torch.tensor(alpha0)).cuda()
        alpha.requires_grad = True
        beta = torch.autograd.Variable(torch.tensor(beta0)).cuda()
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
        X, _, _ = gw_torch(A.clone().detach(), B.clone().detach(), a, b, X.clone().detach(), beta=args.gw_beta,
                           outer_iter=1, inner_iter=50)
        X = X.clone().detach()
        print('Epoch:', ii, 'Objective:', objective.item())
        if ii == args.joint_epoch - 1:
            print(alpha0, beta0)
            X, hits_k_max, mrr_max = gw_torch(A.clone().detach(), B.clone().detach(), a, b, X,
                                              beta=args.gw_beta,
                                              outer_iter=args.epoch - args.joint_epoch,
                                              inner_iter=20,
                                              gt=test_pairs)

    time_ed = time.time()
    res = X.T.cpu().numpy()
    a1, a5, a10, a30, mrr = compute_metrics(res, test_pairs)
    hits_k_max[1] = max(hits_k_max[1], a1)
    hits_k_max[5] = max(hits_k_max[5], a5)
    hits_k_max[10] = max(hits_k_max[10], a10)
    hits_k_max[30] = max(hits_k_max[30], a30)
    mrr_max = max(mrr_max, mrr)

    for key in hits_k_max:
        final_hits_list[key].append(hits_k_max[key])
    final_mrr_list.append(mrr_max)
    time_cost = time_ed - time_st

    print('{} Edge Noise:{} Feat Noise:{} Type:{} Bases:{} GW beta:{} ss:{}, ep:{}'.format(
        args.dataset, args.edge_noise, args.feat_noise, args.noise_type, args.bases, args.gw_beta, args.step_size,
        args.epoch))
    print('H@1 %.2f%% H@5 %.2f%% H@10 %.2f%% H@30 %.2f%% Time: %.2fs' % (
        a1 * 100, a5 * 100, a10 * 100, a30 * 100, time_cost))
    # with open('result.txt', 'a+') as f:
    #     f.write('{} Edge Noise:{} Feat Noise:{} Type:{} Bases:{} GW beta:{} ss:{}, ep:{}\n'.format(
    #         args.dataset, args.edge_noise, args.feat_noise, args.noise_type, args.bases, args.gw_beta, args.step_size, args.epoch))
    #     f.write('H@1 %.2f%% H@5 %.2f%% H@10 %.2f%% H@30 %.2f%% Time: %.2fs\n ' % (a1 * 100, a5 * 100, a10 * 100, a30 * 100, time_cost))

topk = list(final_hits_list.keys())
final_hits = dict()
final_hits_robust = dict()
final_hits_std = dict()
final_hits_robust_std = dict()
for k in topk:
    hits_k_list = np.array(final_hits_list[k])
    final_hits[k] = np.mean(hits_k_list)
    final_hits_std[k] = np.std(hits_k_list)
    hits_k_list = rm_out(hits_k_list)
    final_hits_robust[k] = np.mean(hits_k_list)
    final_hits_robust_std[k] = np.std(hits_k_list)
final_mrr = np.mean(final_mrr_list)
final_mrr_std = np.std(final_mrr_list)
final_mrr_list = rm_out(final_mrr_list)
final_mrr_robust = np.mean(final_mrr_list)
final_mrr_robust_std = np.std(final_mrr_list)

if args.record:
    exp_name = args.exp_name
    if not os.path.exists("results"):
        os.makedirs("results")
    out_path = f"results/{exp_name}_results.csv"
    if not os.path.exists(out_path):
        with open(out_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([""] + [f"Hit@{k}" for k in topk] + ["MRR"] + [f"std@{k}" for k in topk] + ["std_MRR"])

    with open(out_path, "a", newline='') as f:
        writer = csv.writer(f)
        if args.edge_noise_rate > 0:
            header = f"{args.dataset}_(edge-{args.edge_noise_rate:.1f})"
        else:
            header = f"{args.dataset}_(attr-{args.attr_noise_rate:.1f}{'_strong' if args.strong_noise else ''})"
        writer.writerow(
            [header] + [f"{final_hits[k]:.3f}" for k in topk] + [f"{final_mrr:.3f}"] + [f"{final_hits_std[k]:.3f}" for k
                                                                                        in topk] + [
                f"{final_mrr_std:.3f}"])
        if args.robust:
            writer.writerow(
                [header + "_robust"] + [f"{final_hits_robust[k]:.3f}" for k in topk] + [f"{final_mrr_robust:.3f}"] + [
                    f"{final_hits_robust_std[k]:.3f}" for k in topk] + [f"{final_mrr_robust_std:.3f}"])
