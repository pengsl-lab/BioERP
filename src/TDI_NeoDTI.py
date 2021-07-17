# -*- coding: utf-8 -*-
"""

This part of codes from: https://github.com/FangpingWan/NeoDTI

Cross validation for BioERP.
"""
import numpy as np
#from sets import Set
import pickle
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
#from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold
from utils import *
from tflearn.activations import relu
#from sklearn.cross_validation import train_test_split,StratifiedKFold
from sklearn.model_selection import train_test_split,StratifiedKFold
import sys
from optparse import OptionParser
from sklearn import svm



parser = OptionParser()
parser.add_option("-d", "--d", default=768, help="The embedding dimension d")
parser.add_option("-n","--n",default=1, help="global norm to be clipped")
parser.add_option("-k","--k",default=512,help="The dimension of project matrices k")
parser.add_option("-t","--t",default = "o",help="Test scenario")
parser.add_option("-r","--r",default = "ten",help="positive negative ratio")

(opts, args) = parser.parse_args()


def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)

def row_normalize(a_matrix, substract_self_loop):
    if substract_self_loop == True:
        np.fill_diagonal(a_matrix,0)
    a_matrix = a_matrix.astype(float)
    row_sums = a_matrix.sum(axis=1)+1e-12
    new_matrix = a_matrix / row_sums[:, np.newaxis]
    new_matrix[np.isnan(new_matrix) | np.isinf(new_matrix)] = 0.0
    return new_matrix

#load network
network_path = "../data/"

drug_drug = np.loadtxt(network_path+'mat_drug_drug.txt')
#print 'loaded drug drug', check_symmetric(drug_drug), np.shape(drug_drug)
true_drug = 708 # First [0:708] are drugs, the rest are compounds retrieved from ZINC15 database
drug_chemical = np.loadtxt(network_path+'Similarity_Matrix_Drugs.txt')
drug_chemical=drug_chemical[:true_drug,:true_drug]
#print 'loaded drug chemical', check_symmetric(drug_chemical), np.shape(drug_chemical)
drug_disease = np.loadtxt(network_path+'mat_drug_disease.txt')
#print 'loaded drug disease', np.shape(drug_disease)
drug_sideeffect = np.loadtxt(network_path+'mat_drug_se.txt')
#print 'loaded drug sideffect', np.shape(drug_sideeffect)
disease_drug = drug_disease.T
sideeffect_drug = drug_sideeffect.T

protein_protein = np.loadtxt(network_path+'mat_protein_protein.txt')
#print 'loaded protein protein', check_symmetric(protein_protein), np.shape(protein_protein)
protein_sequence = np.loadtxt(network_path+'Similarity_Matrix_Proteins.txt')
#print 'loaded protein sequence', check_symmetric(protein_sequence), np.shape(protein_sequence)
protein_disease = np.loadtxt(network_path+'mat_protein_disease.txt')
#print 'loaded protein disease', np.shape(protein_disease)
disease_protein = protein_disease.T


#normalize network for mean pooling aggregation
drug_drug_normalize = row_normalize(drug_drug,True)
drug_chemical_normalize = row_normalize(drug_chemical,True)
drug_disease_normalize = row_normalize(drug_disease,False)
drug_sideeffect_normalize = row_normalize(drug_sideeffect,False)

protein_protein_normalize = row_normalize(protein_protein,True)
protein_sequence_normalize = row_normalize(protein_sequence,True)
protein_disease_normalize = row_normalize(protein_disease,False)

disease_drug_normalize = row_normalize(disease_drug,False)
disease_protein_normalize = row_normalize(disease_protein,False)
sideeffect_drug_normalize = row_normalize(sideeffect_drug,False)



#define computation graph
num_drug = len(drug_drug_normalize)
num_protein = len(protein_protein_normalize)
num_disease = len(disease_protein_normalize)
num_sideeffect = len(sideeffect_drug_normalize)

dim_drug = int(opts.d)
dim_protein = int(opts.d)
dim_disease = int(opts.d)
dim_sideeffect = int(opts.d)
dim_pred = int(opts.k)
dim_pass = int(opts.d)

class Model(object):
    def __init__(self):
        self._build_model()
        self.kk=0
    def _build_model(self):
        #inputs
        self.drug_drug = tf.placeholder(tf.float32, [num_drug, num_drug])
        self.drug_drug_normalize = tf.placeholder(tf.float32, [num_drug, num_drug])

        self.drug_chemical = tf.placeholder(tf.float32, [num_drug, num_drug])
        self.drug_chemical_normalize = tf.placeholder(tf.float32, [num_drug, num_drug])

        self.drug_disease = tf.placeholder(tf.float32, [num_drug, num_disease])
        self.drug_disease_normalize = tf.placeholder(tf.float32, [num_drug, num_disease])

        self.drug_sideeffect = tf.placeholder(tf.float32, [num_drug, num_sideeffect])
        self.drug_sideeffect_normalize = tf.placeholder(tf.float32, [num_drug, num_sideeffect])

        
        self.protein_protein = tf.placeholder(tf.float32, [num_protein, num_protein])
        self.protein_protein_normalize = tf.placeholder(tf.float32, [num_protein, num_protein])

        self.protein_sequence = tf.placeholder(tf.float32, [num_protein, num_protein])
        self.protein_sequence_normalize = tf.placeholder(tf.float32, [num_protein, num_protein])

        self.protein_disease = tf.placeholder(tf.float32, [num_protein, num_disease])
        self.protein_disease_normalize = tf.placeholder(tf.float32, [num_protein, num_disease])
        
        self.disease_drug = tf.placeholder(tf.float32, [num_disease, num_drug])
        self.disease_drug_normalize = tf.placeholder(tf.float32, [num_disease, num_drug])

        self.disease_protein = tf.placeholder(tf.float32, [num_disease, num_protein])
        self.disease_protein_normalize = tf.placeholder(tf.float32, [num_disease, num_protein])

        self.sideeffect_drug = tf.placeholder(tf.float32, [num_sideeffect, num_drug])
        self.sideeffect_drug_normalize = tf.placeholder(tf.float32, [num_sideeffect, num_drug])

        self.drug_protein = tf.placeholder(tf.float32, [num_drug, num_protein])
        self.drug_protein_normalize = tf.placeholder(tf.float32, [num_drug, num_protein])

        self.protein_drug = tf.placeholder(tf.float32, [num_protein, num_drug])
        self.protein_drug_normalize = tf.placeholder(tf.float32, [num_protein, num_drug])

        self.protein_drug_mask = tf.placeholder(tf.float32, [num_protein, num_drug])

        feature_list = extract_finalembedding()

        drug_feature  = tf.convert_to_tensor(feature_list[0:708])
        protein_feature = tf.convert_to_tensor(feature_list[708:2220])
        disease_feature = tf.convert_to_tensor(feature_list[2220:7823])
        sideeffect_feature = tf.convert_to_tensor(feature_list[7823:12015])
        
        drug_vector1 = tf.nn.l2_normalize(drug_feature, dim=1)
        protein_vector1 = tf.nn.l2_normalize(protein_feature, dim=1)
        disease_vector1 = tf.nn.l2_normalize(disease_feature, dim=1)
        sideeffect_vector1 = tf.nn.l2_normalize(sideeffect_feature, dim=1)

        self.drug_representation = drug_vector1
        self.protein_representation = protein_vector1
        self.disease_representation = disease_vector1
        self.sideeffect_representation = sideeffect_vector1

        #reconstructing networks
        self.drug_drug_reconstruct = bi_layer(self.drug_representation,self.drug_representation, sym=True, dim_pred=dim_pred)
        self.drug_drug_reconstruct_loss = tf.reduce_sum(tf.multiply((self.drug_drug_reconstruct-self.drug_drug), (self.drug_drug_reconstruct-self.drug_drug)))

        self.drug_chemical_reconstruct = bi_layer(self.drug_representation,self.drug_representation, sym=True, dim_pred=dim_pred)
        self.drug_chemical_reconstruct_loss = tf.reduce_sum(tf.multiply((self.drug_chemical_reconstruct-self.drug_chemical), (self.drug_chemical_reconstruct-self.drug_chemical)))

        self.drug_disease_reconstruct = bi_layer(self.drug_representation,self.disease_representation, sym=False, dim_pred=dim_pred)
        self.drug_disease_reconstruct_loss = tf.reduce_sum(tf.multiply((self.drug_disease_reconstruct-self.drug_disease), (self.drug_disease_reconstruct-self.drug_disease)))


        self.drug_sideeffect_reconstruct = bi_layer(self.drug_representation,self.sideeffect_representation, sym=False, dim_pred=dim_pred)
        self.drug_sideeffect_reconstruct_loss = tf.reduce_sum(tf.multiply((self.drug_sideeffect_reconstruct-self.drug_sideeffect), (self.drug_sideeffect_reconstruct-self.drug_sideeffect)))

        self.protein_protein_reconstruct = bi_layer(self.protein_representation,self.protein_representation, sym=True, dim_pred=dim_pred)
        self.protein_protein_reconstruct_loss = tf.reduce_sum(tf.multiply((self.protein_protein_reconstruct-self.protein_protein), (self.protein_protein_reconstruct-self.protein_protein)))

        self.protein_sequence_reconstruct = bi_layer(self.protein_representation,self.protein_representation, sym=True, dim_pred=dim_pred)
        self.protein_sequence_reconstruct_loss = tf.reduce_sum(tf.multiply((self.protein_sequence_reconstruct-self.protein_sequence), (self.protein_sequence_reconstruct-self.protein_sequence)))


        self.protein_disease_reconstruct = bi_layer(self.protein_representation,self.disease_representation, sym=False, dim_pred=dim_pred)
        self.protein_disease_reconstruct_loss = tf.reduce_sum(tf.multiply((self.protein_disease_reconstruct-self.protein_disease), (self.protein_disease_reconstruct-self.protein_disease)))


        self.protein_drug_reconstruct = bi_layer(self.protein_representation, self.drug_representation, sym=False, dim_pred=dim_pred)
        tmp = tf.multiply(self.protein_drug_mask, (self.protein_drug_reconstruct-self.protein_drug))
        self.protein_drug_reconstruct_loss = tf.reduce_sum(tf.multiply(tmp, tmp))

        self.l2_loss = tf.add_n(tf.get_collection("l2_reg"))

        self.loss = self.protein_drug_reconstruct_loss + 1.0*(self.drug_drug_reconstruct_loss+self.drug_chemical_reconstruct_loss+
                                                            self.drug_disease_reconstruct_loss+self.drug_sideeffect_reconstruct_loss+
                                                            self.protein_protein_reconstruct_loss+self.protein_sequence_reconstruct_loss+
                                                            self.protein_disease_reconstruct_loss) + self.l2_loss
graph = tf.get_default_graph()
with graph.as_default():
    model = Model()
    learning_rate = tf.placeholder(tf.float32, [])
    total_loss = model.loss
    dti_loss = model.protein_drug_reconstruct_loss

    optimize = tf.train.AdamOptimizer(learning_rate)
    gradients, variables = zip(*optimize.compute_gradients(total_loss))
    gradients, _ = tf.clip_by_global_norm(gradients, int(opts.n))
    optimizer = optimize.apply_gradients(zip(gradients, variables))

    eval_pred = model.protein_drug_reconstruct

def train_and_evaluate(DTItrain, DTIvalid, DTItest, graph, verbose=True, num_steps = 4000):
    protein_drug = np.zeros((num_protein,num_drug))  #708*1512
    mask = np.zeros((num_protein,num_drug))

    for ele in DTItrain:
        protein_drug[ele[0],ele[1]] = ele[2]
        mask[ele[0],ele[1]] = 1
    drug_protein = protein_drug.T

    protein_drug_normalize = row_normalize(protein_drug,False)
    drug_protein_normalize = row_normalize(drug_protein,False)

    lr = 0.001

    best_valid_aupr = 0
    best_valid_auc = 0
    test_aupr = 0
    test_auc = 0

    with tf.Session(graph=graph) as sess:
        tf.initialize_all_variables().run()
        for i in range(num_steps):
            _, tloss, dtiloss, results = sess.run([optimizer,total_loss,dti_loss,eval_pred], \
                                        feed_dict={model.drug_drug:drug_drug, model.drug_drug_normalize:drug_drug_normalize,\
                                        model.drug_chemical:drug_chemical, model.drug_chemical_normalize:drug_chemical_normalize,\
                                        model.drug_disease:drug_disease, model.drug_disease_normalize:drug_disease_normalize,\
                                        model.drug_sideeffect:drug_sideeffect, model.drug_sideeffect_normalize:drug_sideeffect_normalize,\
                                        model.protein_protein:protein_protein, model.protein_protein_normalize:protein_protein_normalize,\
                                        model.protein_sequence:protein_sequence, model.protein_sequence_normalize:protein_sequence_normalize,\
                                        model.protein_disease:protein_disease, model.protein_disease_normalize:protein_disease_normalize,\
                                        model.disease_drug:disease_drug, model.disease_drug_normalize:disease_drug_normalize,\
                                        model.disease_protein:disease_protein, model.disease_protein_normalize:disease_protein_normalize,\
                                        model.sideeffect_drug:sideeffect_drug, model.sideeffect_drug_normalize:sideeffect_drug_normalize,\
                                        model.drug_protein:drug_protein, model.drug_protein_normalize:drug_protein_normalize,\
                                        model.protein_drug:protein_drug, model.protein_drug_normalize:protein_drug_normalize,\
                                        model.protein_drug_mask:mask,\
                                        learning_rate: lr})

            #every 25 steps of gradient descent, evaluate the performance, other choices of this number are possible
            if i % 25 == 0 and verbose == True:
                print ('step',i,'total and dtiloss',tloss, dtiloss)

                pred_list = []
                ground_truth = []
                for ele in DTIvalid:
                    pred_list.append(results[ele[0],ele[1]])
                    ground_truth.append(ele[2])
                valid_auc = roc_auc_score(ground_truth, pred_list)
                valid_aupr = average_precision_score(ground_truth, pred_list)
                if valid_aupr >= best_valid_aupr:
                    best_valid_aupr = valid_aupr
                    best_valid_auc = valid_auc
                    pred_list = []
                    ground_truth = []
                    test_list = []
                    for ele in DTItest:
                        pred_list.append(results[ele[0],ele[1]])
                        ground_truth.append(ele[2])
                        test_list = pred_list
                    test_auc = roc_auc_score(ground_truth, pred_list)
                    test_aupr = average_precision_score(ground_truth, pred_list)
                print( 'valid auc aupr,', valid_auc, valid_aupr, 'test auc aupr', test_auc, test_aupr)
    return best_valid_auc, best_valid_aupr, test_auc, test_aupr, test_list

test_auc_round = []
test_aupr_round = []
for r in range(1):
    print ('sample round',r+1)
    if opts.t == 'o':
        dti_o = np.loadtxt(network_path+'mat_protein_drug.txt')
    else:
        dti_o = np.loadtxt(network_path+'mat_protein_drug'+opts.t+'.txt')

    whole_positive_index = []
    whole_negative_index = []
    for i in range(np.shape(dti_o)[0]):
        for j in range(np.shape(dti_o)[1]):
            if int(dti_o[i][j]) == 1:
                whole_positive_index.append([i, j])
            elif int(dti_o[i][j]) == 0:
                whole_negative_index.append([i, j])

    if opts.r == 'ten':
        negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),size=10*len(whole_positive_index),replace=False)
    elif opts.r == 'all':
        negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),size=len(whole_negative_index),replace=False)
    else:
        print( 'wrong positive negative ratio')
        break

    data_set = np.zeros((len(negative_sample_index)+len(whole_positive_index),3),dtype=int)
    count = 0
    for i in whole_positive_index:
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 1
        count += 1
    for i in negative_sample_index:
        data_set[count][0] = whole_negative_index[i][0]
        data_set[count][1] = whole_negative_index[i][1]
        data_set[count][2] = 0
        count += 1

    if opts.t == 'unique':
        whole_positive_index_test = []
        whole_negative_index_test = []
        for i in range(np.shape(dti_o)[0]):
            for j in range(np.shape(dti_o)[1]):
                if int(dti_o[i][j]) == 3:
                    whole_positive_index_test.append([i,j])
                elif int(dti_o[i][j]) == 2:
                    whole_negative_index_test.append([i,j])

        if opts.r == 'ten':
            negative_sample_index_test = np.random.choice(np.arange(len(whole_negative_index_test)),size=10*len(whole_positive_index_test),replace=False)
        elif opts.r == 'all':
            negative_sample_index_test = np.random.choice(np.arange(len(whole_negative_index_test)),size=whole_negative_index_test,replace=False)
        else:
            print( 'wrong positive negative ratio')
            break
        data_set_test = np.zeros((len(negative_sample_index_test)+len(whole_positive_index_test),3),dtype=int)
        count = 0
        for i in whole_positive_index_test:
            data_set_test[count][0] = i[0]
            data_set_test[count][1] = i[1]
            data_set_test[count][2] = 1
            count += 1
        for i in negative_sample_index_test:
            data_set_test[count][0] = whole_negative_index_test[i][0]
            data_set_test[count][1] = whole_negative_index_test[i][1]
            data_set_test[count][2] = 0
            count += 1

        DTItrain = data_set
        rs = np.random.randint(0,1000,1)[0]
        DTItrain, DTIvalid =  train_test_split(DTItrain, test_size=0.05, random_state=rs)
        v_auc, v_aupr, t_auc, t_aupr, pred_list = train_and_evaluate(DTItrain=DTItrain, DTIvalid=DTIvalid, DTItest=DTItest, graph=graph, num_steps=3000)  #num_steps=3000

        test_auc_round.append(t_auc)
        test_aupr_round.append(t_aupr)
        np.savetxt('test_auc', test_auc_round)
        np.savetxt('test_aupr', test_aupr_round)

    else:
        test_auc_fold = []
        test_aupr_fold = []
        rs = np.random.randint(0, 1000, 1)[0]
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=rs)

        for train_index, test_index in kf.split(data_set[:, 0:2], data_set[:, 2]):
            DTItrain, DTItest = data_set[train_index], data_set[test_index]
            DTItrain, DTIvalid = train_test_split(DTItrain, test_size=0.05, random_state=rs)
            v_auc, v_aupr, t_auc, t_aupr, test_list = train_and_evaluate(DTItrain=DTItrain, DTIvalid=DTIvalid, DTItest=DTItest, graph=graph, num_steps=3000)
            test_auc_fold.append(t_auc)
            test_aupr_fold.append(t_aupr)

        test_auc_round.append(np.mean(test_auc_fold))
        test_aupr_round.append(np.mean(test_aupr_fold))
        print("auc:", test_auc_round)
        print("aupr:", test_aupr_round)

        np.savetxt('TDI_test_auc_NeoDTI_'+ str(dim_drug), test_auc_fold)
        np.savetxt('TDI_test_aupr_NeoDTI_'+ str(dim_drug), test_aupr_fold)