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
import gc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from block_dataloader import generate_dataloader_block

import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm

import random
from graphsage_model import GraphSAGE
from gcn_model_cherry import GCN
from gat_model_cherry import GAT
import dgl.function as fn
from load_graph import load_reddit, load_ogb, load_cora, load_karate, prepare_data, load_pubmed, load_amazon

from load_graph import load_ogbn_dataset, load_ogb
from memory_usage import see_memory_usage, nvidia_smi_usage
import tracemalloc
from cpu_mem_usage import get_memory
from statistics import mean
from my_utils import parse_results

import pickle
from utils import Logger
import os 
import numpy
from collections import Counter




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
	# train_nid = train_nid.to(device)
	# val_nid=val_nid.to(device)
	# test_nid=test_nid.to(device)
	nfeats=nfeats.to(device)
	g=g.to(device)
	# print('device ', device)
	model.eval()
	with torch.no_grad():
		# pred = model(g=g, x=nfeats)
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

def load_block_subtensor(nfeat, labels, blocks, device,args):
	"""
	Extracts features and labels for a subset of nodes
	"""

	# if args.GPUmem:
		# see_memory_usage("----------------------------------------before batch input features to device")
	batch_inputs = nfeat[blocks[0].srcdata[dgl.NID]].to(device)
	# if args.GPUmem:
		# see_memory_usage("----------------------------------------after batch input features to device")
	batch_labels = labels[blocks[-1].dstdata[dgl.NID]].to(device)
	# if args.GPUmem:
		# see_memory_usage("----------------------------------------after  batch labels to device")
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
	# if args.GPUmem:
		# see_memory_usage("----------------------------------------start of run function ")
	# Unpack data
	g, nfeats, labels, n_classes, train_nid, val_nid, test_nid = data
	in_feats = len(nfeats[0])
	print('in feats: ', in_feats)
	nvidia_smi_list=[]
	if args.selection_method =='metis':
		args.o_graph = dgl.node_subgraph(g, train_nid)


	sampler = dgl.dataloading.MultiLayerNeighborSampler(
		[int(fanout) for fanout in args.fan_out.split(',')])
	full_batch_size = len(train_nid)
	

	args.num_workers = 0
	full_batch_dataloader = dgl.dataloading.DataLoader(
		g,
		train_nid,
		sampler,
		# device='cpu',
		batch_size=full_batch_size,
		shuffle=True,
		drop_last=False,
		num_workers=args.num_workers)


	model = gen_model(args, in_feats, n_classes, device)
					
	logger = Logger(args.num_runs, args)
	for run in range(args.num_runs):
		model.reset_parameters()
		loss_fcn = nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
		for epoch in range(args.num_epochs):
			print("EPOCH BEGIN-----------------------------")
			total_t = time.time()

			num_src_node =0
			num_out_node_FL=0
			num_input_nids=0
			model.train()
			loss_sum=0
			pseudo_mini_loss = torch.tensor([], dtype=torch.long)

			load_block_time = []
			block_move_time = []
			model_time = []
			loss_time = []
			opt_time = []
			
			betty_t1 = time.time()
			# if args.load_full_batch:
			# 	full_batch_dataloader=[]
			# 	file_name=r'./../../dataset/fan_out_'+args.fan_out+'/'+args.dataset+'_'+str(epoch)+'_items.pickle'
			# 	with open(file_name, 'rb') as handle:
			# 		item=pickle.load(handle)
			# 		full_batch_dataloader.append(item)
			
			block_dataloader, weights_list, time_collection = generate_dataloader_block(g, full_batch_dataloader, args)
			betty_time = time.time() - betty_t1
			connect_check_time, block_gen_time_total, batch_blocks_gen_time, partition_time, gen_dist_time =time_collection
			
			print("conncect_check_time: ", connect_check_time)
			print('block_gen_time_total: ', block_gen_time_total)
			print('batch_blocks_gen_time: ', batch_blocks_gen_time)
			print('partition_time: ', partition_time)
			print('gen dist nodes time: ', gen_dist_time)
			
			for step, (input_nodes, seeds, blocks) in enumerate(block_dataloader):
				torch.cuda.reset_max_memory_allocated()
				torch.cuda.synchronize() # synchronized

				num_input_nids	+= len(input_nodes)
				num_src_node+=get_compute_num_nids(blocks)
				num_out_node_FL+=get_FL_output_num_nids(blocks)

				t1 = time.time()
				batch_inputs, batch_labels = load_block_subtensor(nfeats, labels, blocks, device,args)
				t3 = time.time()
				load_block_time.append(t3 - t1)
				blocks = [block.int().to(device) for block in blocks]
				torch.cuda.synchronize() # synchronized
				t4 = time.time()
				block_move_time.append(t4 - t3)
				batch_pred = model(blocks, batch_inputs)
				torch.cuda.synchronize() # synchronized
				t5 = time.time()
				model_time.append(t5 - t4)
				
				if args.dataset=='ogbn-papers100M':
					pseudo_mini_loss = loss_fcn(batch_pred, batch_labels.long())
				else:
					pseudo_mini_loss = loss_fcn(batch_pred, batch_labels)
				pseudo_mini_loss = pseudo_mini_loss*weights_list[step]
				pseudo_mini_loss.backward()
				
				loss_sum += pseudo_mini_loss
				
				torch.cuda.synchronize() # synchronized
				t2 = time.time()
				loss_time.append(t2 - t5)
				
				max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)
				print("Micro-batch-", step, " max memory allocated: ", max_memory_allocated, " GB")
			
			torch.cuda.synchronize() # synchronized
			opt_t = time.time()
			optimizer.step()
			optimizer.zero_grad()
			torch.cuda.synchronize() # synchronized
			opt_time.append(time.time() - opt_t)

			if args.GPUmem:
					see_memory_usage("-----------------------------------------after optimizer zero grad")
			if args.eval:
				
				args.batch_size = len(train_nid)//args.num_batch +1
			
				train_acc, val_acc, test_acc = evaluate(model, g, nfeats, labels, train_nid, val_nid, test_nid, device, args)

				logger.add_result(run, (train_acc, val_acc, test_acc))
					
				print("Run {:02d} | Epoch {:05d} | Loss {:.4f} | Train {:.4f} | Val {:.4f} | Test {:.4f}".format(run, epoch, loss_sum.item(), train_acc, val_acc, test_acc))
			else:
				print(' Run '+str(run)+'| Epoch '+ str( epoch)+' |')

			print("TIME RECORD-----------------------------")
			print("Betty prepare time: ", betty_time)
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
	
def count_parameters(model):
	pytorch_total_params = sum(torch.numel(p) for p in model.parameters())
	print('total model parameters size ', pytorch_total_params)
	print('trainable parameters')
	
	for name, param in model.named_parameters():
		if param.requires_grad:
			print (name + ', '+str(param.data.shape))
	print('-'*40)
	print('un-trainable parameters')
	for name, param in model.named_parameters():
		if not param.requires_grad:
			print (name, param.data.shape)

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
	argparser.add_argument('--load-full-batch', type=bool, default=True)
	# argparser.add_argument('--root', type=str, default='../my_full_graph/')
	argparser.add_argument('--dataset', type=str, default='ogbn-arxiv')
	# argparser.add_argument('--dataset', type=str, default='ogbn-mag')
	# argparser.add_argument('--dataset', type=str, default='ogbn-products')
	# argparser.add_argument('--dataset', type=str, default='cora')
	# argparser.add_argument('--dataset', type=str, default='karate')
	# argparser.add_argument('--dataset', type=str, default='reddit')
	# argparser.add_argument('--aggre', type=str, default='lstm')
	argparser.add_argument('--aggre', type=str, default='mean')
	# argparser.add_argument('--selection-method', type=str, default='range')
	# argparser.add_argument('--selection-method', type=str, default='random')
	# argparser.add_argument('--selection-method', type=str, default='metis')
	argparser.add_argument('--selection-method', type=str, default='REG')
	argparser.add_argument('--num-batch', type=int, default=1)
	argparser.add_argument('--batch-size', type=int, default=0)

	argparser.add_argument('--re-partition-method', type=str, default='REG')
	# argparser.add_argument('--re-partition-method', type=str, default='random')
	argparser.add_argument('--num-re-partition', type=int, default=0)

	# argparser.add_argument('--balanced_init_ratio', type=float, default=0.2)
	argparser.add_argument('--num-runs', type=int, default=1)
	argparser.add_argument('--num-epochs', type=int, default=1)

	argparser.add_argument('--num-hidden', type=int, default=16)

	argparser.add_argument('--num-layers', type=int, default=1)
	argparser.add_argument('--fan-out', type=str, default='10')
	
	argparser.add_argument('--log-indent', type=float, default=0)
#--------------------------------------------------------------------------------------
	

	argparser.add_argument('--lr', type=float, default=1e-2)
	argparser.add_argument('--dropout', type=float, default=0.5)
	argparser.add_argument("--weight-decay", type=float, default=5e-4,
						help="Weight for L2 loss")
	argparser.add_argument("--eval", action='store_true', 
						help='If not set, we will only do the training part.')

	argparser.add_argument('--num-workers', type=int, default=4,
		help="Number of sampling processes. Use 0 for no extra process.")
	

	argparser.add_argument('--log-every', type=int, default=5)
	argparser.add_argument('--eval-every', type=int, default=5)
	argparser.add_argument('--device-number', type=str, default='0')
	argparser.add_argument('--num-heads', type=int, default=4)
	argparser.add_argument('--model', type=str, default='SAGE')
	
	args = argparser.parse_args()

	os.environ["CUDA_VISIBLE_DEVICES"] = args.device_number
	if args.setseed:
		set_seed(args)
	device = "cpu"
	# if args.GPUmem:
		# see_memory_usage("-----------------------------------------before load data ")
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
	elif args.dataset=='ogbn-mag':
		# data = prepare_data_mag(device, args)
		data = load_ogbn_mag(args)
		device = "cuda:0"
		# run_mag(args, device, data)
		# return
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
		
	
	best_test = run(args, device, data)
	

if __name__=='__main__':
	main()

