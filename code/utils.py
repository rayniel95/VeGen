from typing import Any, Dict, Iterable


def get_row_vector(matrix, row_id):
	return matrix[row_id, :]


def get_column_vector(matrix, col_id):
	return matrix[:, col_id]


def max_array(array):
	value = array[0]
	for index in range(array.shape[0]):
		if array[index] > value: value = array[index]
	return value


def max_matrix(matrix):
	max = matrix[0, 0]
	for index1 in range(matrix.shape[0]):
		for index2 in range(matrix.shape[1]):
			if matrix[index1, index2] > max:
				max = matrix[index1, index2]
	return max


class MintermNode:

	def __init__(self, value: Any, mintermindex=-1):
		self.value = value
		self.sons: Dict[Any, 'MintermNode'] = {}
		self.mintermindex = mintermindex


class MintermTree:

	def __init__(self):
		self.root = MintermNode('')
		self.count = 0
		self.patterndocs = {}
		self.binary = {}

	def insert(self, iterable: Iterable[Any], indexdoc: int):
		actual_node = self.root
		inserted = False
		for value in iterable: # O(t)
			if actual_node.sons.get(value, False):
				actual_node = actual_node.sons[value]
			else:
				actual_node.sons[value] = MintermNode(value) # O(d)
				actual_node = actual_node.sons[value]
				inserted = True
		# el ciclo puede ser considerado O(t)
		if inserted:
			actual_node.mintermindex = self.count
			self.count += 1

		try:
			self.patterndocs[actual_node.mintermindex].append(indexdoc)
		except: self.patterndocs[actual_node.mintermindex] = [indexdoc]

		return inserted, actual_node.mintermindex

	def construct(self, matrix, seconddimension: int):
		# se puede construir los minterms activos usando la matriz de pesos, si
		# si el peso es 0 entonces la frecuenci tambien, si el peso es > 0
		# entonces la frecuencia tambien, por lo que da igual cual matriz se use

		for index2 in range(seconddimension): # O(d)
			minterm = [1 if value > 0 else 0 for value in
					   get_column_vector(matrix, index2)] # O(t+t*t) por la list
			inserted, index =  self.insert(minterm, index2) # O(t)
			if inserted:
				strminterm = [str(value) for value in minterm] # O(t)
				# O(t + t*t + t) puede ser O(t)
				self.binary[index] = int(''.join(reversed(strminterm)), base=2)
				# todo sumarle 1 al resultado

		# en general O(t*d)

