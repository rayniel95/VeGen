from typing import List, Dict, Iterator, Any
import numpy as np
import math
import utils



def create_term_document_matrix(line_tuples, document_names, vocab) -> np.ndarray:
	'''Returns a numpy array containing the term document matrix for the input lines.

	Inputs:
	line_tuples: A list of tuples, containing the name of the document and
	a tokenized line from that document.
	document_names: A list of the document names
	vocab: A list of the tokens in the vocabulary

	# NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:39 PM.

	Let m = len(vocab) and n = len(document_names).

	Returns:
	td_matrix: A mxn numpy array where the number of rows is the number of words
		and each column corresponds to a document. A_ij contains the
		frequency with which word i occurs in document j.
	'''

	vocab_to_id = dict(zip(vocab, range(0, len(vocab))))
	docname_to_id = dict(zip(document_names, range(0, len(document_names))))

	# YOUR CODE HERE
	# print(line_tuples)
	# print(vocab)
	# print(document_names)
	td_matrix = np.ndarray(shape=(len(vocab), len(document_names)), dtype=int)

	for index1 in range(len(vocab)):
		for index2 in range(len(document_names)):
			td_matrix[index1, index2] = 0

	for line in line_tuples:
		for word in line[1]:
			td_matrix[vocab_to_id[word], docname_to_id[line[0]]] += 1

	return td_matrix


def create_tf_idf_matrix(term_document_matrix: np.ndarray):
	'''Given the term document matrix, output a tf-idf weighted version.

	  See section 15.2.1 in the textbook.

	  Hint: Use numpy matrix and vector operations to speed up implementation.

	  Input:
		term_document_matrix: Numpy array where each column represents a document
		and each row, the frequency of a word in that document.

	  Returns:
		A numpy array with the same dimension as term_document_matrix, where
		A_ij is weighted by the inverse document frequency of document h.
	  '''

	# YOUR CODE HERE
	tf_matrix = np.ndarray((term_document_matrix.shape[0], term_document_matrix.shape[1]))
	for row in range(term_document_matrix.shape[0]):
		for col in range(term_document_matrix.shape[1]):
			tf_matrix[row, col] = math.log10(term_document_matrix[row, col] + 1)

	df_array = compute_df_array(term_document_matrix)

	idf_array = compute_idf_array(term_document_matrix.shape[1], df_array)
	# print(df_array)
	# print(idf_array)
	tf_idf_matrix = np.ndarray((term_document_matrix.shape[0], term_document_matrix.shape[1]))
	for row in range(term_document_matrix.shape[0]):
		for col in range(term_document_matrix.shape[1]):
			tf_idf_matrix[row, col] = term_document_matrix[row, col] * idf_array[row]

	return tf_idf_matrix


def compute_df_array(term_document_matrix):
	return np.array([sum(int(bool(term_document_matrix[row, col])) for col in
							 range(term_document_matrix.shape[1]))
						for row in range(term_document_matrix.shape[0])])


def compute_idf_array(documents_count, df_array):
	return [math.log10(documents_count / df) if df > 0 else 0 for df in df_array]


def compute_cosine_similarity(vector1, vector2):
	'''Computes the cosine similarity of the two input vectors.

	Inputs:
	vector1: A nx1 numpy array
	vector2: A nx1 numpy array

	Returns:
	A scalar similarity value.
	'''

	# YOUR CODE HERE
	div = math.sqrt(sum(math.pow(vec1, 2) for vec1 in vector1)) * math.sqrt(sum(
		math.pow(vec2, 2) for vec2 in vector2))
	if div == 0: return 0

	return sum(elem1 * elem2 for elem1, elem2 in zip(vector1, vector2)) / div


def compute_jaccard_similarity(vector1, vector2):
	'''Computes the cosine similarity of the two input vectors.

	Inputs:
	vector1: A nx1 numpy array
	vector2: A nx1 numpy array

	Returns:
	A scalar similarity value.
	'''

	# YOUR CODE HERE
	set1 = set(index for index in range(len(vector1)) if vector1[index] > 0)
	set2 = set(index for index in range(len(vector2)) if vector2[index] > 0)
	div = len(set1.union(set2))
	if div == 0: return 0
	return len(set1.intersection(set2)) / div


def compute_dice_similarity(vector1, vector2):
	'''Computes the cosine similarity of the two input vectors.

	Inputs:
	vector1: A nx1 numpy array
	vector2: A nx1 numpy array

	Returns:
	A scalar similarity value.
	'''
	# YOUR CODE HERE
	set1 = set(index for index in range(len(vector1)) if vector1[index] > 0)
	set2 = set(index for index in range(len(vector2)) if vector2[index] > 0)
	div = len(set1) + len(set2)
	if div == 0: return 0
	return (2 * len(set1.intersection(set2))) / div


def rank_plays(target_play_index, term_document_matrix, similarity_fn):
	''' Ranks the similarity of all of the plays to the target play.

	# NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:51 PM.

	Inputs:
	target_play_index: The integer index of the play we want to compare all others against.
	term_document_matrix: The term-document matrix as a mxn numpy array.
	similarity_fn: Function that should be used to compared vectors for two
	  documents. Either compute_dice_similarity, compute_jaccard_similarity, or
	  compute_cosine_similarity.

	Returns:
	A length-n list of integer indices corresponding to play names,
	ordered by decreasing similarity to the play indexed by target_play_index
	'''

	# YOUR CODE HERE
	ranks = [similarity_fn(utils.get_column_vector(term_document_matrix, target_play_index),
						   utils.get_column_vector(term_document_matrix, index))
			 for index in range(term_document_matrix.shape[1])]

	return list(reversed(sorted([index for index in range(term_document_matrix.shape[1])],
				  key=lambda index: ranks[index])))


def rank_words(target_word_index, matrix, similarity_fn):

	''' Ranks the similarity of all of the words to the target word.

	# NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:51 PM.

	Inputs:
	target_word_index: The index of the word we want to compare all others against.
	matrix: Numpy matrix where the ith row represents a vector embedding of the ith word.
	similarity_fn: Function that should be used to compared vectors for two word
	  ebeddings. Either compute_dice_similarity, compute_jaccard_similarity, or
	  compute_cosine_similarity.

	Returns:
	A length-n list of integer word indices, ordered by decreasing similarity to the
	target word indexed by word_index
	'''

	# YOUR CODE HERE
	ranks = [similarity_fn(utils.get_row_vector(matrix, target_word_index),
		utils.get_row_vector(matrix, index)) for index in range(matrix.shape[0])]

	return list(reversed(sorted([index for index in range(matrix.shape[0])],
				  key=lambda index: ranks[index])))


def to_zero_array(array: np.ndarray):
	len_iter = len(array)
	for index in range(len_iter):
		array[index] = 0
	return array


def to_zero_matrix(matrix: np.ndarray):
	for index1 in range(matrix.shape[0]):
		for index2 in range(matrix.shape[1]):
			matrix[index1, index2] = 0
	return matrix


def create_term_consult_vector(vocab: List[str], consult: List[str]):
	array = to_zero_array(np.ndarray((len(vocab),)))

	for word in consult:
		array[vocab.index(word)] += 1

	return array


def normalize_matrix(matrix: np.ndarray):
	max = utils.max_matrix(matrix)
	if max == 0: raise Exception('max is 0, impossible normalize')

	new_matrix = np.ndarray((matrix.shape[0], matrix.shape[1],))
	for index1 in range(new_matrix.shape[0]):
		for index2 in range(new_matrix.shape[1]):
			new_matrix[index1, index2] = matrix[index1, index2] / max
	return new_matrix


def normalize_array(array: np.ndarray):
	max = utils.max_array(array)
	if max == 0: raise Exception('cannot normalize, max is 0')
	new_array = np.ndarray((array.shape[0],))
	for index in range(new_array.shape[0]):
		new_array[index] = array[index] / max
	return new_array


def consult_weights_array(term_frecuency: np.ndarray, idf_array: List[float]) -> np.ndarray:
	array = to_zero_array(np.ndarray((term_frecuency.shape[0])))

	max_value = utils.max_array(term_frecuency)

	if max_value == 0: return array # todo ver esta parte donde se retorna 0
	for index in range(term_frecuency.shape[0]):
		array[index] = (((term_frecuency[index] / max_value) * 0.5) + 0.5) * \
					   idf_array[index]

	return array


def vegen(tf_idfmatrix, tf_idfvector):
	mintermtree = utils.MintermTree()
	mintermtree.construct(tf_idfmatrix, tf_idfmatrix.shape[1])
	# todo tener cuidado al dividir por 0 o crear arrays sucios
	correlationmatrix = np.ndarray((tf_idfmatrix.shape[0], mintermtree.count,),
								   dtype=float)
	for index1 in range(correlationmatrix.shape[0]):
		for index2 in range(correlationmatrix.shape[1]):
			value = 0
			for indexdoc in mintermtree.patterndocs[index2]:
				value += tf_idfmatrix[index1, indexdoc]

			correlationmatrix[index1, index2] = value
	# print(correlationmatrix)
	Narray = np.ndarray((tf_idfmatrix.shape[0],), dtype=float)

	for index1 in range(Narray.shape[0]):
		value = 0
		for index2 in range(correlationmatrix.shape[1]):
			value += math.pow(correlationmatrix[index1, index2], 2)
		Narray[index1] = math.sqrt(value)
	# print(Narray)
	# todo puede que no quepa en la memoria
	vectorialindex = to_zero_matrix(np.ndarray((int(tf_idfmatrix.shape[0]), int(math.pow(tf_idfmatrix.shape[0], 2)),),
								dtype=float))

	for index1 in range(vectorialindex.shape[0]):
		for index2 in range(correlationmatrix.shape[1]):
			if Narray[index1] == 0:
				vectorialindex[index1, mintermtree.binary[index2]] = 0
			else:
				vectorialindex[index1, mintermtree.binary[index2]] = \
					correlationmatrix[index1, index2] / Narray[index1]

	vectorialdocs = to_zero_matrix(np.ndarray((tf_idfmatrix.shape[1],
											   vectorialindex.shape[1],)))

	for vectordoc in range(vectorialdocs.shape[0]):
		for vectorterm in range(vectorialindex.shape[0]):
			for component in range(vectorialdocs.shape[1]):
				vectorialdocs[vectordoc, component] += \
					vectorialindex[vectorterm, component] * \
					tf_idfmatrix[vectorterm, vectordoc]
	print(vectorialdocs)

	vectorq = to_zero_array(np.ndarray((vectorialindex.shape[1],)))
	for vectorterm in range(vectorialindex.shape[0]):
		for component in range(vectorq.shape[0]):
			vectorq[component] += vectorialindex[vectorterm, component] * \
				tf_idfvector[vectorterm]

	print(vectorq)


def main(vocab, docs, consult):
	raise NotImplementedError()


if __name__ == '__main__':
	matrix = np.array([
		[0.41, 0, 0.41, 0.27],
		[0.31, 0.06, 0, 0.41],
		[0, 1, 0.45, 0]
	])

	vegen(matrix, [0.41, 0.41, 1])







