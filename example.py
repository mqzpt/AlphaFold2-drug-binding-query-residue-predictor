##############################################################
###                                                        ###
###             CYCLICA ALPHAFOLD 2 CHALLENGE              ###
###                                                        ###
###  We have developed a binary classification model to    ###
###  predict 'drug binding' or 'non-drug binding' for any  ###
###  query residue within a protein. We are using an       ###
###  AlphaFold2 predicted protien model for our training   ###
###  data, with data confirmed by Cyclica showing if a     ###
###  paticular site (amino acid QR) is 'drug binding' or   ###
###  'non-drug binding'.                                   ###
###                                                        ###
##############################################################
###                                                        ###
###            Authors: Matthew Athanasopoulous            ###
###                                                        ###
##############################################################

# alpha.py cleans our data and prepares it for use in a neural network

# Importing pandas and numpy for functions

import xgboost
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
data = pd.read_csv("af2_dataset_training_labeled.csv")

# Organizing the data (removing redundant features)

# We will be dropping some uneccesary features using the pandas Dataframe.drop() function. We will be left with all of our features representing indivual 2-D arrays, and then piece them together using the numpy empty and append functions.

##################################
#
#  Legend:
#
# x1 -> indexing & annotation_sequence
#   x1_label1_list -> Represents the alpha-carboxyl group's pKa.
#   x1_label2_list -> Represents the alpha-ammonium ion's pKa.
#   x1_label3_list -> Represents the side chain group's pKa.
#   x1_label4_list -> Represents the isoelectronic point.
#   x1_label5_list -> Represents the hydrophilicity of each amino acid.
# x2 -> feat_PHI -> Protein chain bonding angle, computed with [Biopython].
# x3 -> feat_PSI -> Protein chain bonding angle, computed with [Biopython].
# x4 -> feat_TAU -> Protein chain bonding angle, computed with [Biopython].
# x5 -> feat_THETA -> Protein chain bonding angle, computed with [Biopython].
# x6 -> feat_BBSASA -> Protein chain bonding angle, computed with [Biopython].
# x7 -> feat_SCSASA -> Protein chain bonding angle, computed with [Biopython].
#   For reference: (https://biopython.org/docs/1.75/api/Bio.PDB.Polypeptide.html)
# x8 -> feat_pLDDT -> AlphaFold2 residue-level prediction confidence value.
# x9 -> feat_DSSP_H -> Secondary structure assignment by [DSSP].
# x10 -> feat_DSSP_B -> Secondary structure assignment by [DSSP].
# x11 -> feat_DSSP_E -> Secondary structure assignment by [DSSP].
# x12 -> feat_DSSP_G -> Secondary structure assignment by [DSSP].
# x13 -> feat_DSSP_I -> Secondary structure assignment by [DSSP].
# x14 -> feat_DSSP_T -> Secondary structure assignment by [DSSP].
# x15 -> feat_DSSP_S -> Secondary structure assignment by [DSSP].
#   For reference: -> (https://en.wikipedia.org/wiki/DSSP_(algorithm))
# x16 -> feat_DSSP_6 -> Backbone structural feature describing backbone hydrogen bonding networks.
# x17 -> feat_DSSP_7 -> Backbone structural feature describing backbone hydrogen bonding networks.
# x18 -> feat_DSSP_8 -> Backbone structural feature describing backbone hydrogen bonding networks.
# x19 -> feat_DSSP_9 -> Backbone structural feature describing backbone hydrogen bonding networks.
# x20 -> feat_DSSP_10 -> Backbone structural feature describing backbone hydrogen bonding networks.
# x21 -> feat_DSSP_11 -> Backbone structural feature describing backbone hydrogen bonding networks.
# x22 -> feat_DSSP_12 -> Backbone structural feature describing backbone hydrogen bonding networks.
# x23 -> feat_DSSP_13 -> Backbone structural feature describing backbone hydrogen bonding networks.
# x24 -> y_Ligand -> Indicates if the residue (row) belongs to a known binding site or not.
#
###################################

x1 = data.drop(['annotation_atomrec', 'feat_A', 'feat_C', 'feat_D', 'feat_E', 'feat_F', 'feat_G', 'feat_H', 'feat_I', 'feat_K', 'feat_L', 'feat_M', 'feat_N', 'feat_P', 'feat_Q', 'feat_R', 'feat_S', 'feat_T', 'feat_V', 'feat_W', 'feat_Y', 'feat_PHI', 'feat_PSI', 'feat_TAU', 'feat_THETA', 'feat_BBSASA', 'feat_SCSASA', 'feat_pLDDT', 'feat_DSSP_H', 'feat_DSSP_B', 'feat_DSSP_E', 'feat_DSSP_G', 'feat_DSSP_I', 'feat_DSSP_T', 'feat_DSSP_S', 'feat_DSSP_6', 'feat_DSSP_7', 'feat_DSSP_8', 'feat_DSSP_9', 'feat_DSSP_10', 'feat_DSSP_11', 'feat_DSSP_12', 'feat_DSSP_13', 'coord_X', 'coord_Y', 'coord_Z', 'entry', 'entry_index', 'y_Ligand'],
               axis=1)

x1_np_array = x1.to_numpy()

x2 = data.drop(['annotation_sequence', 'feat_A', 'feat_C', 'feat_D', 'feat_E', 'feat_F', 'feat_G', 'feat_H', 'feat_I', 'feat_K', 'feat_L', 'feat_M', 'feat_N', 'feat_P', 'feat_Q', 'feat_R', 'feat_S', 'feat_T', 'feat_V', 'feat_W', 'feat_Y', 'annotation_atomrec', 'feat_PSI', 'feat_TAU', 'feat_THETA', 'feat_BBSASA', 'feat_SCSASA', 'feat_pLDDT', 'feat_DSSP_H', 'feat_DSSP_B', 'feat_DSSP_E', 'feat_DSSP_G', 'feat_DSSP_I', 'feat_DSSP_T', 'feat_DSSP_S', 'feat_DSSP_6', 'feat_DSSP_7', 'feat_DSSP_8', 'feat_DSSP_9', 'feat_DSSP_10', 'feat_DSSP_11', 'feat_DSSP_12', 'feat_DSSP_13', 'coord_X', 'coord_Y', 'coord_Z', 'entry', 'entry_index', 'y_Ligand'],
               axis=1)

x2_np_array = x2.to_numpy()
x2_np_array = np.delete(x2_np_array, 0, 1)

x3 = data.drop(['annotation_sequence', 'feat_A', 'feat_C', 'feat_D', 'feat_E', 'feat_F', 'feat_G', 'feat_H', 'feat_I', 'feat_K', 'feat_L', 'feat_M', 'feat_N', 'feat_P', 'feat_Q', 'feat_R', 'feat_S', 'feat_T', 'feat_V', 'feat_W', 'feat_Y', 'annotation_atomrec', 'feat_PHI', 'feat_TAU', 'feat_THETA', 'feat_BBSASA', 'feat_SCSASA', 'feat_pLDDT', 'feat_DSSP_H', 'feat_DSSP_B', 'feat_DSSP_E', 'feat_DSSP_G', 'feat_DSSP_I', 'feat_DSSP_T', 'feat_DSSP_S', 'feat_DSSP_6', 'feat_DSSP_7', 'feat_DSSP_8', 'feat_DSSP_9', 'feat_DSSP_10', 'feat_DSSP_11', 'feat_DSSP_12', 'feat_DSSP_13', 'coord_X', 'coord_Y', 'coord_Z', 'entry', 'entry_index', 'y_Ligand'],
               axis=1)

x3_np_array = x3.to_numpy()
x3_np_array = np.delete(x3_np_array, 0, 1)

x4 = data.drop(['annotation_sequence', 'feat_A', 'feat_C', 'feat_D', 'feat_E', 'feat_F', 'feat_G', 'feat_H', 'feat_I', 'feat_K', 'feat_L', 'feat_M', 'feat_N', 'feat_P', 'feat_Q', 'feat_R', 'feat_S', 'feat_T', 'feat_V', 'feat_W', 'feat_Y', 'annotation_atomrec', 'feat_PHI', 'feat_PSI', 'feat_THETA', 'feat_BBSASA', 'feat_SCSASA', 'feat_pLDDT', 'feat_DSSP_H', 'feat_DSSP_B', 'feat_DSSP_E', 'feat_DSSP_G', 'feat_DSSP_I', 'feat_DSSP_T', 'feat_DSSP_S', 'feat_DSSP_6', 'feat_DSSP_7', 'feat_DSSP_8', 'feat_DSSP_9', 'feat_DSSP_10', 'feat_DSSP_11', 'feat_DSSP_12', 'feat_DSSP_13', 'coord_X', 'coord_Y', 'coord_Z', 'entry', 'entry_index', 'y_Ligand'],
               axis=1)

x4_np_array = x4.to_numpy()
x4_np_array = np.delete(x4_np_array, 0, 1)

x5 = data.drop(['annotation_sequence', 'feat_A', 'feat_C', 'feat_D', 'feat_E', 'feat_F', 'feat_G', 'feat_H', 'feat_I', 'feat_K', 'feat_L', 'feat_M', 'feat_N', 'feat_P', 'feat_Q', 'feat_R', 'feat_S', 'feat_T', 'feat_V', 'feat_W', 'feat_Y', 'annotation_atomrec', 'feat_PHI', 'feat_PSI', 'feat_TAU', 'feat_BBSASA', 'feat_SCSASA', 'feat_pLDDT', 'feat_DSSP_H', 'feat_DSSP_B', 'feat_DSSP_E', 'feat_DSSP_G', 'feat_DSSP_I', 'feat_DSSP_T', 'feat_DSSP_S', 'feat_DSSP_6', 'feat_DSSP_7', 'feat_DSSP_8', 'feat_DSSP_9', 'feat_DSSP_10', 'feat_DSSP_11', 'feat_DSSP_12', 'feat_DSSP_13', 'coord_X', 'coord_Y', 'coord_Z', 'entry', 'entry_index', 'y_Ligand'],
               axis=1)

x5_np_array = x5.to_numpy()
x5_np_array = np.delete(x5_np_array, 0, 1)

x6 = data.drop(['annotation_sequence', 'feat_A', 'feat_C', 'feat_D', 'feat_E', 'feat_F', 'feat_G', 'feat_H', 'feat_I', 'feat_K', 'feat_L', 'feat_M', 'feat_N', 'feat_P', 'feat_Q', 'feat_R', 'feat_S', 'feat_T', 'feat_V', 'feat_W', 'feat_Y', 'annotation_atomrec', 'feat_PHI', 'feat_PSI', 'feat_TAU', 'feat_THETA', 'feat_SCSASA', 'feat_pLDDT', 'feat_DSSP_H', 'feat_DSSP_B', 'feat_DSSP_E', 'feat_DSSP_G', 'feat_DSSP_I', 'feat_DSSP_T', 'feat_DSSP_S', 'feat_DSSP_6', 'feat_DSSP_7', 'feat_DSSP_8', 'feat_DSSP_9', 'feat_DSSP_10', 'feat_DSSP_11', 'feat_DSSP_12', 'feat_DSSP_13', 'coord_X', 'coord_Y', 'coord_Z', 'entry', 'entry_index', 'y_Ligand'],
               axis=1)

x6_np_array = x6.to_numpy()
x6_np_array = np.delete(x6_np_array, 0, 1)

x7 = data.drop(['annotation_sequence', 'feat_A', 'feat_C', 'feat_D', 'feat_E', 'feat_F', 'feat_G', 'feat_H', 'feat_I', 'feat_K', 'feat_L', 'feat_M', 'feat_N', 'feat_P', 'feat_Q', 'feat_R', 'feat_S', 'feat_T', 'feat_V', 'feat_W', 'feat_Y', 'annotation_atomrec', 'feat_PHI', 'feat_PSI', 'feat_TAU', 'feat_THETA', 'feat_BBSASA', 'feat_pLDDT', 'feat_DSSP_H', 'feat_DSSP_B', 'feat_DSSP_E', 'feat_DSSP_G', 'feat_DSSP_I', 'feat_DSSP_T', 'feat_DSSP_S', 'feat_DSSP_6', 'feat_DSSP_7', 'feat_DSSP_8', 'feat_DSSP_9', 'feat_DSSP_10', 'feat_DSSP_11', 'feat_DSSP_12', 'feat_DSSP_13', 'coord_X', 'coord_Y', 'coord_Z', 'entry', 'entry_index', 'y_Ligand'],
               axis=1)

x7_np_array = x7.to_numpy()
x7_np_array = np.delete(x7_np_array, 0, 1)

x8 = data.drop(['annotation_sequence', 'feat_A', 'feat_C', 'feat_D', 'feat_E', 'feat_F', 'feat_G', 'feat_H', 'feat_I', 'feat_K', 'feat_L', 'feat_M', 'feat_N', 'feat_P', 'feat_Q', 'feat_R', 'feat_S', 'feat_T', 'feat_V', 'feat_W', 'feat_Y', 'annotation_atomrec', 'feat_PHI', 'feat_PSI', 'feat_TAU', 'feat_THETA', 'feat_BBSASA', 'feat_SCSASA', 'feat_DSSP_H', 'feat_DSSP_B', 'feat_DSSP_E', 'feat_DSSP_G', 'feat_DSSP_I', 'feat_DSSP_T', 'feat_DSSP_S', 'feat_DSSP_6', 'feat_DSSP_7', 'feat_DSSP_8', 'feat_DSSP_9', 'feat_DSSP_10', 'feat_DSSP_11', 'feat_DSSP_12', 'feat_DSSP_13', 'coord_X', 'coord_Y', 'coord_Z', 'entry', 'entry_index', 'y_Ligand'],
               axis=1)

x8_np_array = x8.to_numpy()
x8_np_array = np.delete(x8_np_array, 0, 1)

x9 = data.drop(['annotation_sequence', 'feat_A', 'feat_C', 'feat_D', 'feat_E', 'feat_F', 'feat_G', 'feat_H', 'feat_I', 'feat_K', 'feat_L', 'feat_M', 'feat_N', 'feat_P', 'feat_Q', 'feat_R', 'feat_S', 'feat_T', 'feat_V', 'feat_W', 'feat_Y', 'annotation_atomrec', 'feat_PHI', 'feat_PSI', 'feat_TAU', 'feat_THETA', 'feat_BBSASA', 'feat_SCSASA', 'feat_pLDDT', 'feat_DSSP_B', 'feat_DSSP_E', 'feat_DSSP_G', 'feat_DSSP_I', 'feat_DSSP_T', 'feat_DSSP_S', 'feat_DSSP_6', 'feat_DSSP_7', 'feat_DSSP_8', 'feat_DSSP_9', 'feat_DSSP_10', 'feat_DSSP_11', 'feat_DSSP_12', 'feat_DSSP_13', 'coord_X', 'coord_Y', 'coord_Z', 'entry', 'entry_index', 'y_Ligand'],
               axis=1)

x9_np_array = x9.to_numpy()
x9_np_array = np.delete(x9_np_array, 0, 1)

x10 = data.drop(['annotation_sequence', 'feat_A', 'feat_C', 'feat_D', 'feat_E', 'feat_F', 'feat_G', 'feat_H', 'feat_I', 'feat_K', 'feat_L', 'feat_M', 'feat_N', 'feat_P', 'feat_Q', 'feat_R', 'feat_S', 'feat_T', 'feat_V', 'feat_W', 'feat_Y', 'annotation_atomrec', 'feat_PHI', 'feat_PSI', 'feat_TAU', 'feat_THETA', 'feat_BBSASA', 'feat_SCSASA', 'feat_pLDDT', 'feat_DSSP_H', 'feat_DSSP_E', 'feat_DSSP_G', 'feat_DSSP_I', 'feat_DSSP_T', 'feat_DSSP_S', 'feat_DSSP_6', 'feat_DSSP_7', 'feat_DSSP_8', 'feat_DSSP_9', 'feat_DSSP_10', 'feat_DSSP_11', 'feat_DSSP_12', 'feat_DSSP_13', 'coord_X', 'coord_Y', 'coord_Z', 'entry', 'entry_index', 'y_Ligand'],
                axis=1)

x10_np_array = x10.to_numpy()
x10_np_array = np.delete(x10_np_array, 0, 1)

x11 = data.drop(['annotation_sequence', 'feat_A', 'feat_C', 'feat_D', 'feat_E', 'feat_F', 'feat_G', 'feat_H', 'feat_I', 'feat_K', 'feat_L', 'feat_M', 'feat_N', 'feat_P', 'feat_Q', 'feat_R', 'feat_S', 'feat_T', 'feat_V', 'feat_W', 'feat_Y', 'annotation_atomrec', 'feat_PHI', 'feat_PSI', 'feat_TAU', 'feat_THETA', 'feat_BBSASA', 'feat_SCSASA', 'feat_pLDDT', 'feat_DSSP_H', 'feat_DSSP_B', 'feat_DSSP_G', 'feat_DSSP_I', 'feat_DSSP_T', 'feat_DSSP_S', 'feat_DSSP_6', 'feat_DSSP_7', 'feat_DSSP_8', 'feat_DSSP_9', 'feat_DSSP_10', 'feat_DSSP_11', 'feat_DSSP_12', 'feat_DSSP_13', 'coord_X', 'coord_Y', 'coord_Z', 'entry', 'entry_index', 'y_Ligand'],
                axis=1)

x11_np_array = x11.to_numpy()
x11_np_array = np.delete(x11_np_array, 0, 1)

x12 = data.drop(['annotation_sequence', 'feat_A', 'feat_C', 'feat_D', 'feat_E', 'feat_F', 'feat_G', 'feat_H', 'feat_I', 'feat_K', 'feat_L', 'feat_M', 'feat_N', 'feat_P', 'feat_Q', 'feat_R', 'feat_S', 'feat_T', 'feat_V', 'feat_W', 'feat_Y', 'annotation_atomrec', 'feat_PHI', 'feat_PSI', 'feat_TAU', 'feat_THETA', 'feat_BBSASA', 'feat_SCSASA', 'feat_pLDDT', 'feat_DSSP_H', 'feat_DSSP_B', 'feat_DSSP_E', 'feat_DSSP_I', 'feat_DSSP_T', 'feat_DSSP_S', 'feat_DSSP_6', 'feat_DSSP_7', 'feat_DSSP_8', 'feat_DSSP_9', 'feat_DSSP_10', 'feat_DSSP_11', 'feat_DSSP_12', 'feat_DSSP_13', 'coord_X', 'coord_Y', 'coord_Z', 'entry', 'entry_index', 'y_Ligand'],
                axis=1)

x12_np_array = x12.to_numpy()
x12_np_array = np.delete(x12_np_array, 0, 1)

x13 = data.drop(['annotation_sequence', 'feat_A', 'feat_C', 'feat_D', 'feat_E', 'feat_F', 'feat_G', 'feat_H', 'feat_I', 'feat_K', 'feat_L', 'feat_M', 'feat_N', 'feat_P', 'feat_Q', 'feat_R', 'feat_S', 'feat_T', 'feat_V', 'feat_W', 'feat_Y', 'annotation_atomrec', 'feat_PHI', 'feat_PSI', 'feat_TAU', 'feat_THETA', 'feat_BBSASA', 'feat_SCSASA', 'feat_pLDDT', 'feat_DSSP_H', 'feat_DSSP_B', 'feat_DSSP_E', 'feat_DSSP_G', 'feat_DSSP_T', 'feat_DSSP_S', 'feat_DSSP_6', 'feat_DSSP_7', 'feat_DSSP_8', 'feat_DSSP_9', 'feat_DSSP_10', 'feat_DSSP_11', 'feat_DSSP_12', 'feat_DSSP_13', 'coord_X', 'coord_Y', 'coord_Z', 'entry', 'entry_index', 'y_Ligand'],
                axis=1)

x13_np_array = x13.to_numpy()
x13_np_array = np.delete(x13_np_array, 0, 1)

x14 = data.drop(['annotation_sequence', 'feat_A', 'feat_C', 'feat_D', 'feat_E', 'feat_F', 'feat_G', 'feat_H', 'feat_I', 'feat_K', 'feat_L', 'feat_M', 'feat_N', 'feat_P', 'feat_Q', 'feat_R', 'feat_S', 'feat_T', 'feat_V', 'feat_W', 'feat_Y', 'annotation_atomrec', 'feat_PHI', 'feat_PSI', 'feat_TAU', 'feat_THETA', 'feat_BBSASA', 'feat_SCSASA', 'feat_pLDDT', 'feat_DSSP_H', 'feat_DSSP_B', 'feat_DSSP_E', 'feat_DSSP_G', 'feat_DSSP_I', 'feat_DSSP_S', 'feat_DSSP_6', 'feat_DSSP_7', 'feat_DSSP_8', 'feat_DSSP_9', 'feat_DSSP_10', 'feat_DSSP_11', 'feat_DSSP_12', 'feat_DSSP_13', 'coord_X', 'coord_Y', 'coord_Z', 'entry', 'entry_index', 'y_Ligand'],
                axis=1)

x14_np_array = x14.to_numpy()
x14_np_array = np.delete(x14_np_array, 0, 1)

x15 = data.drop(['annotation_sequence', 'feat_A', 'feat_C', 'feat_D', 'feat_E', 'feat_F', 'feat_G', 'feat_H', 'feat_I', 'feat_K', 'feat_L', 'feat_M', 'feat_N', 'feat_P', 'feat_Q', 'feat_R', 'feat_S', 'feat_T', 'feat_V', 'feat_W', 'feat_Y', 'annotation_atomrec', 'feat_PHI', 'feat_PSI', 'feat_TAU', 'feat_THETA', 'feat_BBSASA', 'feat_SCSASA', 'feat_pLDDT', 'feat_DSSP_H', 'feat_DSSP_B', 'feat_DSSP_E', 'feat_DSSP_G', 'feat_DSSP_I', 'feat_DSSP_T', 'feat_DSSP_6', 'feat_DSSP_7', 'feat_DSSP_8', 'feat_DSSP_9', 'feat_DSSP_10', 'feat_DSSP_11', 'feat_DSSP_12', 'feat_DSSP_13', 'coord_X', 'coord_Y', 'coord_Z', 'entry', 'entry_index', 'y_Ligand'],
                axis=1)

x15_np_array = x15.to_numpy()
x15_np_array = np.delete(x15_np_array, 0, 1)

x16 = data.drop(['annotation_sequence', 'feat_A', 'feat_C', 'feat_D', 'feat_E', 'feat_F', 'feat_G', 'feat_H', 'feat_I', 'feat_K', 'feat_L', 'feat_M', 'feat_N', 'feat_P', 'feat_Q', 'feat_R', 'feat_S', 'feat_T', 'feat_V', 'feat_W', 'feat_Y', 'annotation_atomrec', 'feat_PHI', 'feat_PSI', 'feat_TAU', 'feat_THETA', 'feat_BBSASA', 'feat_SCSASA', 'feat_pLDDT', 'feat_DSSP_H', 'feat_DSSP_B', 'feat_DSSP_E', 'feat_DSSP_G', 'feat_DSSP_I', 'feat_DSSP_T', 'feat_DSSP_S', 'feat_DSSP_7', 'feat_DSSP_8', 'feat_DSSP_9', 'feat_DSSP_10', 'feat_DSSP_11', 'feat_DSSP_12', 'feat_DSSP_13', 'coord_X', 'coord_Y', 'coord_Z', 'entry', 'entry_index', 'y_Ligand'],
                axis=1)

x16_np_array = x16.to_numpy()
x16_np_array = np.delete(x16_np_array, 0, 1)

x17 = data.drop(['annotation_sequence', 'feat_A', 'feat_C', 'feat_D', 'feat_E', 'feat_F', 'feat_G', 'feat_H', 'feat_I', 'feat_K', 'feat_L', 'feat_M', 'feat_N', 'feat_P', 'feat_Q', 'feat_R', 'feat_S', 'feat_T', 'feat_V', 'feat_W', 'feat_Y', 'annotation_atomrec', 'feat_PHI', 'feat_PSI', 'feat_TAU', 'feat_THETA', 'feat_BBSASA', 'feat_SCSASA', 'feat_pLDDT', 'feat_DSSP_H', 'feat_DSSP_B', 'feat_DSSP_E', 'feat_DSSP_G', 'feat_DSSP_I', 'feat_DSSP_T', 'feat_DSSP_S', 'feat_DSSP_6', 'feat_DSSP_8', 'feat_DSSP_9', 'feat_DSSP_10', 'feat_DSSP_11', 'feat_DSSP_12', 'feat_DSSP_13', 'coord_X', 'coord_Y', 'coord_Z', 'entry', 'entry_index', 'y_Ligand'],
                axis=1)

x17_np_array = x17.to_numpy()
x17_np_array = np.delete(x17_np_array, 0, 1)

x18 = data.drop(['annotation_sequence', 'feat_A', 'feat_C', 'feat_D', 'feat_E', 'feat_F', 'feat_G', 'feat_H', 'feat_I', 'feat_K', 'feat_L', 'feat_M', 'feat_N', 'feat_P', 'feat_Q', 'feat_R', 'feat_S', 'feat_T', 'feat_V', 'feat_W', 'feat_Y', 'annotation_atomrec', 'feat_PHI', 'feat_PSI', 'feat_TAU', 'feat_THETA', 'feat_BBSASA', 'feat_SCSASA', 'feat_pLDDT', 'feat_DSSP_H', 'feat_DSSP_B', 'feat_DSSP_E', 'feat_DSSP_G', 'feat_DSSP_I', 'feat_DSSP_T', 'feat_DSSP_S', 'feat_DSSP_6', 'feat_DSSP_7', 'feat_DSSP_9', 'feat_DSSP_10', 'feat_DSSP_11', 'feat_DSSP_12', 'feat_DSSP_13', 'coord_X', 'coord_Y', 'coord_Z', 'entry', 'entry_index', 'y_Ligand'],
                axis=1)

x18_np_array = x18.to_numpy()
x18_np_array = np.delete(x18_np_array, 0, 1)

x19 = data.drop(['annotation_sequence', 'feat_A', 'feat_C', 'feat_D', 'feat_E', 'feat_F', 'feat_G', 'feat_H', 'feat_I', 'feat_K', 'feat_L', 'feat_M', 'feat_N', 'feat_P', 'feat_Q', 'feat_R', 'feat_S', 'feat_T', 'feat_V', 'feat_W', 'feat_Y', 'annotation_atomrec', 'feat_PHI', 'feat_PSI', 'feat_TAU', 'feat_THETA', 'feat_BBSASA', 'feat_SCSASA', 'feat_pLDDT', 'feat_DSSP_H', 'feat_DSSP_B', 'feat_DSSP_E', 'feat_DSSP_G', 'feat_DSSP_I', 'feat_DSSP_T', 'feat_DSSP_S', 'feat_DSSP_6', 'feat_DSSP_7', 'feat_DSSP_8', 'feat_DSSP_10', 'feat_DSSP_11', 'feat_DSSP_12', 'feat_DSSP_13', 'coord_X', 'coord_Y', 'coord_Z', 'entry', 'entry_index', 'y_Ligand'],
                axis=1)

x19_np_array = x19.to_numpy()
x19_np_array = np.delete(x19_np_array, 0, 1)

x20 = data.drop(['annotation_sequence', 'feat_A', 'feat_C', 'feat_D', 'feat_E', 'feat_F', 'feat_G', 'feat_H', 'feat_I', 'feat_K', 'feat_L', 'feat_M', 'feat_N', 'feat_P', 'feat_Q', 'feat_R', 'feat_S', 'feat_T', 'feat_V', 'feat_W', 'feat_Y', 'annotation_atomrec', 'feat_PHI', 'feat_PSI', 'feat_TAU', 'feat_THETA', 'feat_BBSASA', 'feat_SCSASA', 'feat_pLDDT', 'feat_DSSP_H', 'feat_DSSP_B', 'feat_DSSP_E', 'feat_DSSP_G', 'feat_DSSP_I', 'feat_DSSP_T', 'feat_DSSP_S', 'feat_DSSP_6', 'feat_DSSP_7', 'feat_DSSP_8', 'feat_DSSP_9', 'feat_DSSP_11', 'feat_DSSP_12', 'feat_DSSP_13', 'coord_X', 'coord_Y', 'coord_Z', 'entry', 'entry_index', 'y_Ligand'],
                axis=1)

x20_np_array = x20.to_numpy()
x20_np_array = np.delete(x20_np_array, 0, 1)

x21 = data.drop(['annotation_sequence', 'feat_A', 'feat_C', 'feat_D', 'feat_E', 'feat_F', 'feat_G', 'feat_H', 'feat_I', 'feat_K', 'feat_L', 'feat_M', 'feat_N', 'feat_P', 'feat_Q', 'feat_R', 'feat_S', 'feat_T', 'feat_V', 'feat_W', 'feat_Y', 'annotation_atomrec', 'feat_PHI', 'feat_PSI', 'feat_TAU', 'feat_THETA', 'feat_BBSASA', 'feat_SCSASA', 'feat_pLDDT', 'feat_DSSP_H', 'feat_DSSP_B', 'feat_DSSP_E', 'feat_DSSP_G', 'feat_DSSP_I', 'feat_DSSP_T', 'feat_DSSP_S', 'feat_DSSP_6', 'feat_DSSP_7', 'feat_DSSP_8', 'feat_DSSP_9', 'feat_DSSP_10', 'feat_DSSP_12', 'feat_DSSP_13', 'coord_X', 'coord_Y', 'coord_Z', 'entry', 'entry_index', 'y_Ligand'],
                axis=1)

x21_np_array = x21.to_numpy()
x21_np_array = np.delete(x21_np_array, 0, 1)

x22 = data.drop(['annotation_sequence', 'feat_A', 'feat_C', 'feat_D', 'feat_E', 'feat_F', 'feat_G', 'feat_H', 'feat_I', 'feat_K', 'feat_L', 'feat_M', 'feat_N', 'feat_P', 'feat_Q', 'feat_R', 'feat_S', 'feat_T', 'feat_V', 'feat_W', 'feat_Y', 'annotation_atomrec', 'feat_PHI', 'feat_PSI', 'feat_TAU', 'feat_THETA', 'feat_BBSASA', 'feat_SCSASA', 'feat_pLDDT', 'feat_DSSP_H', 'feat_DSSP_B', 'feat_DSSP_E', 'feat_DSSP_G', 'feat_DSSP_I', 'feat_DSSP_T', 'feat_DSSP_S', 'feat_DSSP_6', 'feat_DSSP_7', 'feat_DSSP_8', 'feat_DSSP_9', 'feat_DSSP_10', 'feat_DSSP_11', 'feat_DSSP_13', 'coord_X', 'coord_Y', 'coord_Z', 'entry', 'entry_index', 'y_Ligand'],
                axis=1)

x22_np_array = x22.to_numpy()
x22_np_array = np.delete(x22_np_array, 0, 1)

x23 = data.drop(['annotation_sequence', 'feat_A', 'feat_C', 'feat_D', 'feat_E', 'feat_F', 'feat_G', 'feat_H', 'feat_I', 'feat_K', 'feat_L', 'feat_M', 'feat_N', 'feat_P', 'feat_Q', 'feat_R', 'feat_S', 'feat_T', 'feat_V', 'feat_W', 'feat_Y', 'annotation_atomrec', 'feat_PHI', 'feat_PSI', 'feat_TAU', 'feat_THETA', 'feat_BBSASA', 'feat_SCSASA', 'feat_pLDDT', 'feat_DSSP_H', 'feat_DSSP_B', 'feat_DSSP_E', 'feat_DSSP_G', 'feat_DSSP_I', 'feat_DSSP_T', 'feat_DSSP_S', 'feat_DSSP_6', 'feat_DSSP_7', 'feat_DSSP_8', 'feat_DSSP_9', 'feat_DSSP_10', 'feat_DSSP_11', 'feat_DSSP_12', 'coord_X', 'coord_Y', 'coord_Z', 'entry', 'entry_index', 'y_Ligand'],
                axis=1)

x23_np_array = x23.to_numpy()
x23_np_array = np.delete(x23_np_array, 0, 1)

x24 = data.drop(['annotation_sequence', 'feat_A', 'feat_C', 'feat_D', 'feat_E', 'feat_F', 'feat_G', 'feat_H', 'feat_I', 'feat_K', 'feat_L', 'feat_M', 'feat_N', 'feat_P', 'feat_Q', 'feat_R', 'feat_S', 'feat_T', 'feat_V', 'feat_W', 'feat_Y', 'annotation_atomrec', 'feat_PHI', 'feat_PSI', 'feat_TAU', 'feat_THETA', 'feat_BBSASA', 'feat_SCSASA', 'feat_pLDDT', 'feat_DSSP_H', 'feat_DSSP_B', 'feat_DSSP_E', 'feat_DSSP_G', 'feat_DSSP_I', 'feat_DSSP_T', 'feat_DSSP_S', 'feat_DSSP_6', 'feat_DSSP_7', 'feat_DSSP_8', 'feat_DSSP_9', 'feat_DSSP_10', 'feat_DSSP_11', 'feat_DSSP_12', 'feat_DSSP_13', 'coord_X', 'coord_Y', 'coord_Z', 'entry', 'entry_index'],
                axis=1)

x24_np_array = x24.to_numpy()
x24_np_array = np.delete(x24_np_array, 0, 1)

# All columns have been seperated, we will now make adjustments to columns which are not "machine-readable".

x1_label1_list = []
x1_label2_list = []
x1_label3_list = []
x1_label4_list = []
x1_label5_list = []

x9_label_list = []
x10_label_list = []
x11_label_list = []
x12_label_list = []
x13_label_list = []
x14_label_list = []
x15_label_list = []

x24_label_list = []

# Making a combination of pKa1, pKa2, pKa3, and pl values for each amino acid. For reference pKa1= α-carboxyl group, pKa2 = α-ammonium ion, and pKa3 = side chain group. Also, pl is just the isoelectronic point. Researach from the University of Calgary: https://www.chem.ucalgary.ca/courses/350/Carey5th/Ch27/ch27-1-4-2.html

# Also adds a hydrophilicity value for each amino acid. Research for values from IMGT: https://www.imgt.org/IMGTeducation/Aide-memoire/_UK/aminoacids/IMGTclasses.html

for i in range(0, data.shape[0]):
    if x1_np_array[i][1] == 'A':
        x1_label1_list.append(2.34)
        x1_label2_list.append(9.69)
        x1_label3_list.append(0)
        x1_label4_list.append(6.00)
        x1_label5_list.append(1.8)
    elif x1_np_array[i][1] == 'C':
        x1_label1_list.append(1.96)
        x1_label2_list.append(8.18)
        x1_label3_list.append(0)
        x1_label4_list.append(5.07)
        x1_label5_list.append(2.5)
    elif x1_np_array[i][1] == 'D':
        x1_label1_list.append(1.88)
        x1_label2_list.append(9.60)
        x1_label3_list.append(3.65)
        x1_label4_list.append(2.77)
        x1_label5_list.append(-3.5)
    elif x1_np_array[i][1] == 'E':
        x1_label1_list.append(2.19)
        x1_label2_list.append(9.67)
        x1_label3_list.append(4.25)
        x1_label4_list.append(3.22)
        x1_label5_list.append(-3.5)
    elif x1_np_array[i][1] == 'F':
        x1_label1_list.append(1.83)
        x1_label2_list.append(9.13)
        x1_label3_list.append(0)
        x1_label4_list.append(5.48)
        x1_label5_list.append(2.8)
    elif x1_np_array[i][1] == 'G':
        x1_label1_list.append(2.34)
        x1_label2_list.append(9.60)
        x1_label3_list.append(0)
        x1_label4_list.append(5.97)
        x1_label5_list.append(-0.4)
    elif x1_np_array[i][1] == 'H':
        x1_label1_list.append(1.82)
        x1_label2_list.append(9.17)
        x1_label3_list.append(6.00)
        x1_label4_list.append(7.59)
        x1_label5_list.append(-3.2)
    elif x1_np_array[i][1] == 'I':
        x1_label1_list.append(2.36)
        x1_label2_list.append(9.60)
        x1_label3_list.append(0)
        x1_label4_list.append(5.98)
        x1_label5_list.append(4.5)
    elif x1_np_array[i][1] == 'K':
        x1_label1_list.append(2.18)
        x1_label2_list.append(8.95)
        x1_label3_list.append(10.53)
        x1_label4_list.append(9.74)
        x1_label5_list.append(-3.9)
    elif x1_np_array[i][1] == 'L':
        x1_label1_list.append(2.36)
        x1_label2_list.append(9.60)
        x1_label3_list.append(0)
        x1_label4_list.append(5.98)
        x1_label5_list.append(3.8)
    elif x1_np_array[i][1] == 'M':
        x1_label1_list.append(2.28)
        x1_label2_list.append(9.21)
        x1_label3_list.append(0)
        x1_label4_list.append(5.74)
        x1_label5_list.append(1.9)
    elif x1_np_array[i][1] == 'N':
        x1_label1_list.append(2.02)
        x1_label2_list.append(8.80)
        x1_label3_list.append(0)
        x1_label4_list.append(5.41)
        x1_label5_list.append(-3.5)
    elif x1_np_array[i][1] == 'P':
        x1_label1_list.append(1.99)
        x1_label2_list.append(10.60)
        x1_label3_list.append(0)
        x1_label4_list.append(6.30)
        x1_label5_list.append(-1.6)
    elif x1_np_array[i][1] == 'Q':
        x1_label1_list.append(2.17)
        x1_label2_list.append(9.13)
        x1_label3_list.append(0)
        x1_label4_list.append(5.65)
        x1_label5_list.append(-3.5)
    elif x1_np_array[i][1] == 'R':
        x1_label1_list.append(2.17)
        x1_label2_list.append(9.04)
        x1_label3_list.append(12.48)
        x1_label4_list.append(10.76)
        x1_label5_list.append(-4.5)
    elif x1_np_array[i][1] == 'S':
        x1_label1_list.append(2.21)
        x1_label2_list.append(9.15)
        x1_label3_list.append(0)
        x1_label4_list.append(5.68)
        x1_label5_list.append(-0.8)
    elif x1_np_array[i][1] == 'T':
        x1_label1_list.append(2.09)
        x1_label2_list.append(9.10)
        x1_label3_list.append(0)
        x1_label4_list.append(5.60)
        x1_label5_list.append(-0.7)
    elif x1_np_array[i][1] == 'V':
        x1_label1_list.append(2.32)
        x1_label2_list.append(9.62)
        x1_label3_list.append(0)
        x1_label4_list.append(5.96)
        x1_label5_list.append(4.2)
    elif x1_np_array[i][1] == 'W':
        x1_label1_list.append(2.83)
        x1_label2_list.append(9.39)
        x1_label3_list.append(0)
        x1_label4_list.append(5.89)
        x1_label5_list.append(-0.9)
    elif x1_np_array[i][1] == 'Y':
        x1_label1_list.append(2.20)
        x1_label2_list.append(9.11)
        x1_label3_list.append(0)
        x1_label4_list.append(5.66)
        x1_label5_list.append(-1.3)

for i in range(0, data.shape[0]):
    if x9_np_array[i] == False:
        x9_label_list.append(0)
    elif x9_np_array[i] == True:
        x9_label_list.append(1)

for i in range(0, data.shape[0]):
    if x10_np_array[i] == False:
        x10_label_list.append(0)
    elif x10_np_array[i] == True:
        x10_label_list.append(1)

for i in range(0, data.shape[0]):
    if x11_np_array[i] == False:
        x11_label_list.append(0)
    elif x11_np_array[i] == True:
        x11_label_list.append(1)

for i in range(0, data.shape[0]):
    if x12_np_array[i] == False:
        x12_label_list.append(0)
    elif x12_np_array[i] == True:
        x12_label_list.append(1)

for i in range(0, data.shape[0]):
    if x13_np_array[i] == False:
        x13_label_list.append(0)
    elif x13_np_array[i] == True:
        x13_label_list.append(1)

for i in range(0, data.shape[0]):
    if x14_np_array[i] == False:
        x14_label_list.append(0)
    elif x14_np_array[i] == True:
        x14_label_list.append(1)

for i in range(0, data.shape[0]):
    if x15_np_array[i] == False:
        x15_label_list.append(0)
    elif x15_np_array[i] == True:
        x15_label_list.append(1)

for i in range(0, data.shape[0]):
    if x24_np_array[i] == False:
        x24_label_list.append(0)
    elif x24_np_array[i] == True:
        x24_label_list.append(1)


# The data has been numerized. We will now make it back into a big 2D matrix so we can so machine learning and stuff

x1_label1 = np.empty([len(x1_label1_list), 0], float)
x1_label1 = np.append(x1_label1, np.array([x1_label1_list]).transpose(),
                      axis=1)

x1_label2 = np.empty([len(x1_label2_list), 0], float)
x1_label2 = np.append(x1_label2, np.array([x1_label2_list]).transpose(),
                      axis=1)

x1_label3 = np.empty([len(x1_label3_list), 0], float)
x1_label3 = np.append(x1_label3, np.array([x1_label3_list]).transpose(),
                      axis=1)

x1_label4 = np.empty([len(x1_label4_list), 0], float)
x1_label4 = np.append(x1_label4, np.array([x1_label4_list]).transpose(),
                      axis=1)

x1_label5 = np.empty([len(x1_label5_list), 0], float)
x1_label5 = np.append(x1_label5, np.array([x1_label5_list]).transpose(),
                      axis=1)

x9_label = np.empty([len(x9_label_list), 0], float)
x9_label = np.append(x9_label, np.array([x9_label_list]).transpose(),
                     axis=1)

x10_label = np.empty([len(x10_label_list), 0], float)
x10_label = np.append(x10_label, np.array([x10_label_list]).transpose(),
                      axis=1)

x11_label = np.empty([len(x11_label_list), 0], float)
x11_label = np.append(x11_label, np.array([x11_label_list]).transpose(),
                      axis=1)

x12_label = np.empty([len(x12_label_list), 0], float)
x12_label = np.append(x12_label, np.array([x12_label_list]).transpose(),
                      axis=1)

x13_label = np.empty([len(x13_label_list), 0], float)
x13_label = np.append(x13_label, np.array([x13_label_list]).transpose(),
                      axis=1)

x14_label = np.empty([len(x14_label_list), 0], float)
x14_label = np.append(x14_label, np.array([x14_label_list]).transpose(),
                      axis=1)

x15_label = np.empty([len(x15_label_list), 0], float)
x15_label = np.append(x15_label, np.array([x15_label_list]).transpose(),
                      axis=1)

x24_label = np.empty([len(x24_label_list), 0], float)
label = np.append(x24_label, np.array([x24_label_list]).transpose(),
                  axis=1)

# Creating the 2D output matrix with hstack;
# This matrix is optimized for our neural net build.

parameters = np.hstack((x1_label1, x1_label2, x1_label3, x1_label4, x1_label5, x2_np_array, x3_np_array, x4_np_array, x5_np_array, x6_np_array, x7_np_array, x8_np_array, x9_label, x10_label,
                        x11_label, x12_label, x13_label, x14_label, x15_label, x16_np_array, x17_np_array, x18_np_array, x19_np_array, x20_np_array, x21_np_array, x22_np_array, x23_np_array))


X_train, X_test, y_train, y_test = train_test_split(
    parameters, label, test_size=0.2, random_state=42)

without_categorical_columns = [
    col for col in X_train.columns if X_train[col].dtype != "O"]

xgb = xgboost.XGBClassifier()
xgb.fit(X_train, y_train)

y_test_pred = xgb.predict(X_test[without_categorical_columns])

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_pred, pos_label=1)
auc_roc = metrics.auc(fpr, tpr)

precision, recall, _ = metrics.precision_recall_curve(y_test, y_test_pred)
auc_pr = metrics.auc(recall, precision)

print(f"ROC-AUC: {auc_roc} \n PR-AUC {auc_pr}")
