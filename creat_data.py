import pandas as pd
import numpy as np
import os
import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *
import csv
from sklearn.model_selection import train_test_split


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    # 边信息转成图
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index

# 将数据分为训练集和测试集两个csv文件（3:7的比例）

df = pd.read_csv('NEWSMILESDH.csv')
df = np.array(df)
allurl_fea=[d[0] for d in df]
df1=df[:int(0.7*len(allurl_fea))]
df1=pd.DataFrame(df1,columns=['IL_SMILES','Temperature(K)','Pressure(mPa)','xCO2(mole fraction)'])
df1.to_csv("./train_jrh_new.csv",index=False)
df2=df[int(0.7*len(allurl_fea)):]
df2=pd.DataFrame(df2,columns=['IL_SMILES','Temperature(K)','Pressure(mPa)','xCO2(mole fraction)'])
df2.to_csv("./test_jrh_new.csv",index=False)

# 分别转化smiles图

compound_iso_smiles_1 = []
compound_iso_smiles_2 = []

compound_iso_smiles_1 += list(df1['IL_SMILES'])
compound_iso_smiles_1 = set(compound_iso_smiles_1)
smile_graph_1 = {}
for smile in compound_iso_smiles_1:
    g_1 = smile_to_graph(smile)
    smile_graph_1[smile] = g_1

compound_iso_smiles_2 += list(df2['IL_SMILES'])
compound_iso_smiles_2 = set(compound_iso_smiles_2)
smile_graph_2 = {}
for smile in compound_iso_smiles_2:
    g_2 = smile_to_graph(smile)
    smile_graph_2[smile] = g_2

# 转化成pt文件等待调用

train_jrh = pd.read_csv('train_jrh_new.csv')
train_drugs, train_t, train_p, train_Y = list(train_jrh['IL_SMILES']), list(train_jrh['Temperature(K)']), list(train_jrh['Pressure(mPa)']), \
                                     list(train_jrh['xCO2(mole fraction)'])
train_drugs, train_t, train_p, train_Y = np.asarray(train_drugs), np.asarray(train_t), np.asarray(train_p), np.asarray(train_Y)
train_jrh_data = TestbedDataset(root='data', dataset='train' + '_jrh_new', xd=train_drugs, t=train_t, p=train_p, y=train_Y,
                          smile_graph=smile_graph_1)

test_jrh = pd.read_csv('test_jrh_new.csv')
test_drugs, test_t, test_p, test_Y = list(test_jrh['IL_SMILES']), list(test_jrh['Temperature(K)']), list(test_jrh['Pressure(mPa)']), \
                                     list(test_jrh['xCO2(mole fraction)'])
test_drugs, test_t, test_p, test_Y = np.asarray(test_drugs), np.asarray(test_t), np.asarray(test_p), np.asarray(test_Y)
test_jrh_data = TestbedDataset(root='data', dataset='test' + '_jrh_new', xd=test_drugs, t=test_t, p=test_p, y=test_Y,
                          smile_graph=smile_graph_2)