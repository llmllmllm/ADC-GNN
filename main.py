import torch
import torch.nn.functional as F
import argparse
import time
from dataset import Dataset
from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_score
from ADCGNN import *
from sklearn.model_selection import train_test_split
import pickle as pkl
import numpy as np
from diffusion import GaussianDiffusion, make_beta_schedule

def compute_gmean(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    recall_pos = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    recall_neg = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    return np.sqrt(recall_pos * recall_neg)

class ModelWrapper(torch.nn.Module):
    def __init__(self, base_model, num_classes, in_feats):
        super().__init__()
        self.base_model = base_model
        self.noise_head = torch.nn.Linear(num_classes, in_feats)
    
    def forward(self, x, t=None):
        logits = self.base_model(x)
        noise_pred = self.noise_head(logits)
        return noise_pred, logits

def  train(model, g, args):
    features = g.ndata['feature']
    labels = g.ndata['label']
    
    device = features.device  
    in_feats = features.shape[1]
    print("in_feats:", in_feats)
    print("g.ndata['feature'].shape:", features.shape)
    
    features_reshaped = features
    print("features_reshaped:", features_reshaped.shape)
    betas = make_beta_schedule(schedule="cosine", n_timestep=1500, cosine_s=12e-3)
    diffusion = GaussianDiffusion(betas).to(device)
    if 'train_mask' not in g.ndata:
        num_nodes = labels.shape[0]
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        indices = np.arange(num_nodes)
        train_idx, rest_idx = train_test_split(indices, train_size=args.train_ratio, stratify=labels.cpu().numpy(), random_state=2)
        val_idx, test_idx = train_test_split(rest_idx, test_size=0.67, stratify=labels.cpu().numpy()[rest_idx], random_state=2)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
    else:
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']

    print('train/dev/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    best_loss = 1e6
    final_trec = final_tpre = final_tmf1 = final_tauc = final_tgmean = 0.0

    weight_val = (1 - labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    num_classes = 2
    weight_tensor = torch.ones(num_classes, device=device)
    weight_tensor[1] = weight_val
    print('cross entropy weight:', weight_tensor)
    
    time_start = time.time()
    for e in range(1, args.epoch + 1):
        model.train()
        n, logits = model(features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask].long(), weight=weight_tensor)
        
        batch_size = features_reshaped.shape[0]
        t = torch.randint(0, diffusion.num_timesteps, (batch_size,)).to(device)
        loss_diff = diffusion.p_loss_contrastive(model, features_reshaped, t, temperature=0.1, lambda_contrast=0.1)
        total_loss = loss + 0.1 * loss_diff
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            _, logits_val = model(features)
            loss_val = F.cross_entropy(logits_val[val_mask], labels[val_mask].long(), weight=weight_tensor)
            probs = logits_val.softmax(1)
            f1, thres = get_best_f1(labels[val_mask], probs[val_mask])
            preds = torch.zeros_like(labels).to(device)
            preds[probs[:, 1] > thres] = 1
            
            y_true = labels[test_mask].cpu().numpy()
            y_pred = preds[test_mask].cpu().numpy()
            trec = recall_score(y_true, y_pred)
            tpre = precision_score(y_true, y_pred)
            tmf1 = f1_score(y_true, y_pred, average='macro')
            tauc = roc_auc_score(y_true, probs[test_mask][:, 1].cpu().numpy())
            tgmean = compute_gmean(y_true, y_pred)

        if loss_val <= best_loss:
            best_loss = loss_val
            final_trec = trec
            final_tpre = tpre
            final_tmf1 = tmf1
            final_tauc = tauc
            final_tgmean = tgmean
            pred_y = probs

        print('Epoch {}, loss: {:.4f}, val mf1: {:.4f}'.format(e, loss_val, f1))
        if args.del_ratio == 0 and e % 20 == 0:
            with open(f'probs_{args.dataset}_ADCGNN', 'wb') as f:
                pkl.dump(pred_y, f)
                
    time_end = time.time()
    print('time cost: ', time_end - time_start, 's')
    
    result = 'REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f} GMean {:.2f}'.format(
        final_trec * 100, final_tpre * 100, final_tmf1 * 100, final_tauc * 100, final_tgmean * 100
    )
    with open('result.txt', 'a+') as f:
        f.write(f'{result}\n')
    return final_tmf1, final_tauc, final_tgmean

def get_best_f1(labels, probs):
    if labels.is_cuda:
        labels = labels.cpu().numpy()
    else:
        labels = labels.numpy()
    if probs.is_cuda:
        probs = probs.cpu().numpy()
    else:
        probs = probs.numpy()
    
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:, 1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    print("seed:", seed, "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ADCGNN')
    parser.add_argument("--dataset", type=str, default="amazon",
                        help="Dataset for this model (yelp/amazon/tfinance/tsocial)")
    parser.add_argument("--train_ratio", type=float, default=0.01, help="Training ratio")
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--order", type=int, default=2, help="Order C in Beta Wavelet")
    parser.add_argument("--epoch", type=int, default=100, help="The max number of epochs")
    parser.add_argument("--run", type=int, default=1, help="Running times")
    parser.add_argument("--del_ratio", type=float, default=0., help="delete ratios")
    parser.add_argument("--adj_type", type=str, default='sym', help="sym or rw")
    parser.add_argument("--load_epoch", type=int, default=100, help="load epoch prediction")
    parser.add_argument("--data_path", type=str, default='./data', help="data path")

    args = parser.parse_args()
    print("Training ratio :", args.train_ratio)
    print("Deletion ratio :", args.del_ratio)
    print(args)
    
    dataset_name = args.dataset
    del_ratio = args.del_ratio
    if args.dataset == 'yelp':
        homo = 0
    else :
        homo = args.dataset
    
    order = args.order
    h_feats = args.hid_dim
    adj_type = args.adj_type
    load_epoch = args.load_epoch
    data_path = args.data_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = Dataset(load_epoch, dataset_name, del_ratio, homo, data_path, adj_type=adj_type).graph
    graph = graph.to(device)
    in_feats = graph.ndata['feature'].shape[1]
    num_classes = 2

    set_random_seed(72)
    
    if args.run == 1:
        if homo == 'amazon':
            base_model = ADCGNN_amazon(in_feats, h_feats, num_classes, graph, d=order)
        elif homo == 0:
            base_model = ADCGNN_yelp(in_feats, h_feats, num_classes, graph, d=order)
        elif homo == 'tfinance':
            base_model = ADCGNN_tfinance(in_feats, h_feats, num_classes, graph, d=order)
        elif homo == 'CM': 
            base_model = ADCGNN_CM(in_feats, h_feats, num_classes, graph, d=order)
        base_model.to(device)
        model = ModelWrapper(base_model, num_classes, in_feats).to(device)
        train(model, graph, args)
    else:
        final_mf1s, final_aucs, final_gmeans = [], [], []
        for tt in range(args.run):
            if homo == 'amazon':
                base_model = ADCGNN_amazon(in_feats, h_feats, num_classes, graph, d=order)
            elif homo == 0:
                base_model = ADCGNN_yelp(in_feats, h_feats, num_classes, graph, d=order)
            elif homo == 'tfinance':
                base_model = ADCGNN_tfinance(in_feats, h_feats, num_classes, graph, d=order)
            elif homo == 'CM': 
                base_model = ADCGNN_CM(in_feats, h_feats, num_classes, graph, d=order)
            base_model.to(device)
            model = ModelWrapper(base_model, num_classes, in_feats).to(device)
            mf1, auc, gmean = train(model, graph, args)
            final_mf1s.append(mf1)
            final_aucs.append(auc)
            final_gmeans.append(gmean)
        final_mf1s = np.array(final_mf1s)
        final_aucs = np.array(final_aucs)
        result = 'MF1-mean: {:.2f}, AUC-mean: {:.2f}, GMean-mean: {:.2f}'.format(
            100 * np.mean(final_mf1s),
            100 * np.mean(final_aucs),
            100 * np.mean(final_gmeans)
        )
        print(result)
