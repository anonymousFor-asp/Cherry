import sys
sys.path.insert(0,'..')
sys.path.insert(0,'..')
sys.path.insert(0,'../pytorch/utils/')
sys.path.insert(0,'../pytorch/micro_batch_train/')
sys.path.insert(0,'../pytorch/models/')
import dgl
from dgl.data.utils import save_graphs
import numpy as np
from statistics import mean
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

import dgl.nn.pytorch as dglnn
import time
import argparse
import random
from graphsage_model import GraphSAGE
from gcn_model_cherry import GCN
from gat_model_cherry import GAT
from load_graph import load_reddit, load_ogb, load_cora, load_karate, prepare_data, load_pubmed, load_amazon

from load_graph import load_ogbn_dataset
from memory_usage import see_memory_usage, nvidia_smi_usage

import pickle
from utils import Logger





def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.device >= 0:
		torch.cuda.manual_seed_all(args.seed)
		torch.cuda.manual_seed(args.seed)
		torch.backends.cudnn.enabled = False
		torch.backends.cudnn.deterministic = True
		dgl.seed(args.seed)
		dgl.random.seed(args.seed)

def CPU_DELTA_TIME(tic, str1):
	toc = time.time()
	print(str1 + ' spend:  {:.6f}'.format(toc - tic))
	return toc


def compute_acc(pred, labels):
	"""
	Compute the accuracy of prediction given the labels.
	"""
	labels = labels.long()
	return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, nfeats, labels, train_nid, val_nid, test_nid, device, args):
	"""
	Evaluate the model on the validation set specified by ``val_nid``.
	g : The entire graph.
	inputs : The features of all the nodes.
	labels : The labels of all the nodes.
	val_nid : the node Ids for validation.
	device : The GPU device to evaluate on.
	"""
	nfeats=nfeats.to(device)
	g=g.to(device)
	model.eval()
	with torch.no_grad():
		pred = model.inference(g, nfeats,  args, device)
	model.train()
	
	train_acc= compute_acc(pred[train_nid], labels[train_nid].to(pred.device))
	val_acc=compute_acc(pred[val_nid], labels[val_nid].to(pred.device))
	test_acc=compute_acc(pred[test_nid], labels[test_nid].to(pred.device))
	return (train_acc, val_acc, test_acc)


def load_subtensor(nfeat, labels, seeds, input_nodes, device):
	"""
	Extracts features and labels for a subset of nodes
	"""
	batch_inputs = nfeat[input_nodes].to(device)
	batch_labels = labels[seeds].to(device)
	return batch_inputs, batch_labels

def load_block_subtensor(nfeat, labels, blocks, device):
	"""
	Extracts features and labels for a subset of nodes
	"""
	
	batch_inputs = nfeat[blocks[0].srcdata[dgl.NID]].to(device)
	batch_labels = labels[blocks[-1].dstdata[dgl.NID]].to(device)
	return batch_inputs, batch_labels

def get_compute_num_nids(blocks):
	res=0
	for b in blocks:
		res+=len(b.srcdata['_ID'])
	return res

def get_FL_output_num_nids(blocks):
	
	output_fl =len(blocks[0].dstdata['_ID'])
	return output_fl

def gen_model(args, in_feats, out_feats, device):
	if args.model == "SAGE":
		model = GraphSAGE(
			in_feats,
			args.num_hidden,
			out_feats,
			args.aggre,
			args.num_layers,
			F.relu,
			args.dropout
		).to(device)
	elif args.model == "GCN":
		model = GCN(
			in_feats,
			args.num_hidden,
			out_feats,
			args.num_layers,
			F.relu,
			args.dropout
		).to(device)
	elif args.model == "GAT":
		model = GAT(
			in_feats,
			args.num_hidden,
			out_feats,
			args.num_heads,
			args.num_layers,
			F.relu,
			args.dropout
		).to(device)
	
	return model

#### Entry point
def run(args, device, data):
	# Unpack data
	g, nfeats, labels, n_classes, train_nid, val_nid, test_nid = data
	in_feats = len(nfeats[0])
	print('nfeats:', nfeats.shape)
	print('in feats: ', in_feats)
	print('train_nid:', train_nid.shape)
	
	sampler = dgl.dataloading.MultiLayerNeighborSampler(
		[int(fanout) for fanout in args.fan_out.split(',')])
	if args.num_batch == 1:
		args.batch_size = len(train_nid)
	if args.batch_size == 0:
		if len(train_nid)%args.num_batch==0:
			args.batch_size = len(train_nid)//args.num_batch
		else:
			args.batch_size = len(train_nid)//args.num_batch + 1
	
	args.num_workers = 0 # when features on GPU, the number of workers should set 0 
	batch_dataloader = dgl.dataloading.DataLoader(
		g,
		train_nid,
		sampler,
		batch_size=args.batch_size,
		shuffle=True,
		drop_last=False,
		num_workers=args.num_workers)
	
	model = gen_model(args, in_feats, n_classes, device)
	loss_fcn = nn.CrossEntropyLoss()
	
	logger = Logger(args.num_runs, args)
	
	for run in range(args.num_runs):
		model.reset_parameters()
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		for epoch in range(args.num_epochs):
			total_t = time.time()

			num_src_node =0
			num_out_node_FL=0
			num_input_nids=0
			model.train()

			load_block_time = []
			block_move_time = []
			model_time = []
			loss_time = []
			opt_time = []

			print('Enter Batch-------------------------------')
			for step, (input_nodes, seeds, blocks) in enumerate(batch_dataloader):
				torch.cuda.reset_max_memory_allocated()
				num_input_nids	+= len(input_nodes)
				num_src_node+=get_compute_num_nids(blocks)
				num_out_node_FL+=get_FL_output_num_nids(blocks)
				
				t1 = time.time()
				batch_inputs, batch_labels = load_block_subtensor(nfeats, labels, blocks, device)
				t3 = time.time()
				load_block_time.append(t3 - t1)
				blocks = [block.int().to(device) for block in blocks]
				t4 = time.time()
				block_move_time.append(t4 - t3)
				batch_pred = model(blocks, batch_inputs)
				t5 = time.time()
				model_time.append(t5 - t4)

				if args.dataset=='ogbn-papers100M':
					loss = loss_fcn(batch_pred, batch_labels.long())
				else:
					loss = loss_fcn(batch_pred, batch_labels)
				loss.backward()
				t6 = time.time()
				loss_time.append(t6 - t5)

				optimizer.step()
				optimizer.zero_grad()
				opt_time.append(time.time() - t6)

				max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)
				print("Mini-batch-", step, " max memory allocated: ", max_memory_allocated, " GB")

			if args.eval:
				train_acc, val_acc, test_acc = evaluate(model, g, nfeats, labels, train_nid, val_nid, test_nid, device, args)
				logger.add_result(run, (train_acc, val_acc, test_acc))
				print("Run {:02d} | Epoch {:05d} | Loss {:.4f} | Train {:.4f} | Val {:.4f} | Test {:.4f}".format(run, epoch, loss.item(), train_acc, val_acc, test_acc))
			else:
				print(' Run '+str(run)+'| Epoch '+ str( epoch)+' |')

			print("TIME RECORD-----------------------------")
			print("load_block_time: ", sum(load_block_time))
			print("block_move_time: ", sum(block_move_time))
			print("model_time: ", sum(model_time))
			print("loss_time: ", sum(loss_time))
			print("optimizer_time: ", sum(opt_time))
			print("total_time: ", time.time() - total_t)
			print("NODES RECORD----------------------------")
			print('Number of nodes for computation during this epoch: ', num_src_node)
			print('Number of input nodes during this epoch: ', num_input_nids)
			print('Number of first layer output nodes during this epoch: ', num_out_node_FL)
	if args.GPUmem:
		see_memory_usage("-----------------------------------------after mini batch train ")


def main():
	# get_memory("-----------------------------------------main_start***************************")
	tt = time.time()
	print("main start at this time " + str(tt))
	argparser = argparse.ArgumentParser("multi-gpu training")
	argparser.add_argument('--device', type=int, default=0,
		help="GPU device ID. Use -1 for CPU training")
	argparser.add_argument('--seed', type=int, default=1236)
	argparser.add_argument('--setseed', type=bool, default=True)
	argparser.add_argument('--GPUmem', type=bool, default=True)
	argparser.add_argument('--load-full-batch', type=bool, default=False)
	# argparser.add_argument('--root', type=str, default='../my_full_graph/')
	# argparser.add_argument('--dataset', type=str, default='ogbn-arxiv')
	# argparser.add_argument('--dataset', type=str, default='ogbn-mag')
	argparser.add_argument('--dataset', type=str, default='ogbn-products')
	# argparser.add_argument('--aggre', type=str, default='lstm')
	# argparser.add_argument('--dataset', type=str, default='cora')
	# argparser.add_argument('--dataset', type=str, default='karate')
	# argparser.add_argument('--dataset', type=str, default='reddit')
	argparser.add_argument('--aggre', type=str, default='mean')
	
	argparser.add_argument('--num-runs', type=int, default=1)
	# argparser.add_argument('--num-epochs', type=int, default=)
	# argparser.add_argument('--num-runs', type=int, default=10)
	argparser.add_argument('--num-epochs', type=int, default=500)
	argparser.add_argument('--num-hidden', type=int, default=256)

	argparser.add_argument('--num-layers', type=int, default=3)
	argparser.add_argument('--fan-out', type=str, default='25,35,40')

	#---------------------------------------------------------------------------------------
	argparser.add_argument('--num-batch', type=int, default=2)
	# argparser.add_argument('--num-batch', type=int, default=2)
	argparser.add_argument('--batch-size', type=int, default=0) 
 
	# argparser.add_argument('--batch-size', type=int, default=2) # karate
	# argparser.add_argument('--batch-size', type=int, default=70) # cora
	# argparser.add_argument('--batch-size', type=int, default=30) # pubmed
	# argparser.add_argument('--batch-size', type=int, default=76716) # reddit
	# argparser.add_argument('--batch-size', type=int, default=45471) # ogbn-arxiv 2 batches
	# argparser.add_argument('--batch-size', type=int, default=22736) # ogbn-arxiv 4 batches

	#--------------------------------------------------------------------------------------

	argparser.add_argument('--lr', type=float, default=0.003)
	argparser.add_argument('--dropout', type=float, default=0.5)
	argparser.add_argument("--weight-decay", type=float, default=5e-4,
						help="Weight for L2 loss")
	argparser.add_argument("--eval", action='store_true', 
						help='If not set, we will only do the training part.')

	argparser.add_argument('--num-workers', type=int, default=0,
		help="Number of sampling processes. Use 0 for no extra process.")
	argparser.add_argument("--eval-batch-size", type=int, default=100000,
						help="evaluation batch size")
	argparser.add_argument("--R", type=int, default=5,
						help="number of hops")
	argparser.add_argument('--log-indent', type=int, default=2)
	argparser.add_argument('--device-number', type=str, default='0')
	argparser.add_argument('--num-heads', type=int, default=4)
	argparser.add_argument('--model', type=str, default='SAGE')


	args = argparser.parse_args()

	os.environ["CUDA_VISIBLE_DEVICES"] = args.device_number
	if args.setseed:
		set_seed(args)
	device = "cpu"
	if args.GPUmem:
		see_memory_usage("-----------------------------------------before load data ")
	if args.dataset=='karate':
		g, n_classes = load_karate()
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='cora':
		g, n_classes = load_cora()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='pubmed':
		g, n_classes = load_pubmed()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='reddit':
		g, n_classes = load_reddit()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
	elif args.dataset == 'ogbn-arxiv':
		data = load_ogbn_dataset(args.dataset,  args)
		device = "cuda:0"

	elif args.dataset=='ogbn-products':
		g, n_classes = load_ogb(args.dataset,args)
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='ogbn-papers100M':
		g, n_classes = load_ogb(args.dataset,args)
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='amazon':
		g, n_classes = load_amazon()
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	else:
		raise Exception('unknown dataset')
	if args.GPUmem:
		see_memory_usage("-----------------------------------------after load data ")
	best_test = run(args, device, data)
	

if __name__=='__main__':
	main()
