import numpy as np
import random
import copy as cp
import time


class ILSNueva:

	def __init__(self, data, ml_const, cl_const, nb_clust, segment_percent, ls_max_neighbors, pbt_inherit, xi):

		self._data = data
		self._ml = ml_const
		self._cl = cl_const
		self._dim = data.shape[0]
		self._result_nb_clust = nb_clust
		self._evals_done = 0
		self._segment_size = int(np.ceil(segment_percent * self._dim))
		self._max_neighbors = ls_max_neighbors
		self._pbt_inherit = pbt_inherit
		self._xi = xi

	def init_ils(self):

		self._best_solution = np.random.randint(0, self._result_nb_clust, (2, self._dim))
		self._best_fitness = np.empty(2)
		self._best_fitness[0] = self.get_single_fitness(self._best_solution[0, :])[0]
		self._best_fitness[1] = self.get_single_fitness(self._best_solution[1, :])[0]


	# Funcionalidad para evaluar un unico individuo decodificado
	def get_single_fitness(self, cromosome):

		current_clustering = cromosome
		# Inicializamos la distancia media de las instancias de los clusters
		total_mean_distance = 0
		# Obtenemos el numero de clusters del clustering actual
		nb_clusters = len(set(current_clustering))

		# Para cada cluster en el clustering actual
		for j in set(current_clustering):
			# Obtener las instancias asociadas al cluster
			clust = self._data[current_clustering == j, :]

			if clust.shape[0] > 1:

				# Obtenemos la distancia media intra-cluster
				tot = 0.
				for k in range(clust.shape[0] - 1):
					tot += ((((clust[k + 1:] - clust[k]) ** 2).sum(1)) ** .5).sum()

				if ((clust.shape[0] - 1) * (clust.shape[0]) / 2.) == 0.0:
					print(tot)
					print(cromosome)
					print(clust)

				avg = tot / ((clust.shape[0] - 1) * (clust.shape[0]) / 2.)

				# Acumular la distancia media
				total_mean_distance += avg

		# Inicializamos el numero de restricciones que no se satisfacen
		infeasability = 0

		# Calculamos el numero de restricciones must-link que no se satisfacen
		for c in range(np.shape(self._ml)[0]):

			if current_clustering[self._ml[c][0]] != current_clustering[self._ml[c][1]]:
				infeasability += 1

		# Calculamos el numero de restricciones cannot-link que no se satisfacen
		for c in range(np.shape(self._cl)[0]):

			if current_clustering[self._cl[c][0]] == current_clustering[self._cl[c][1]]:
				infeasability += 1

		# Calcular el valor de la funcion fitness
		distance = total_mean_distance / nb_clusters
		# penalty = self._mu * self._dim * infeasability
		penalty = distance * infeasability
		fitness = distance + penalty

		# Aumentar en uno el contador de evaluacions de la funcion objetivo
		self._evals_done += 1

		return fitness, distance, penalty

	def segment_mutation_operator(self, chromosome):

		segment_start = np.random.randint(self._dim)
		segment_end = (segment_start + self._segment_size) % self._dim
		new_segment = np.random.randint(0, self._result_nb_clust, self._segment_size)
		if segment_start < segment_end:

			chromosome[segment_start:segment_end] = new_segment

		else:

			chromosome[segment_start:] = new_segment[:self._dim - segment_start]
			#np.random.randint(0, self._result_nb_clust, self._dim - segment_start)
			chromosome[:segment_end] = new_segment[self._dim - segment_start:]
			#np.random.randint(0, self._result_nb_clust, segment_end)

		return chromosome

	def random_mutation_operator(self, chromosome):

		pos = np.random.choice(self._dim, self._segment_size, replace=False)
		new_labels = np.random.randint(0, self._result_nb_clust, self._segment_size)

		chromosome[pos] = new_labels

		return chromosome

	# Operador de cruce aleatorio
	def uniform_crossover_operator(self, parent1, parent2):

		# Obtenemos el vector de probabilidades de herdar de parent1 y resolvemos las probabilidades
		v = np.where(np.random.rand(self._dim) > self._pbt_inherit)[0]

		# Creamos el nuevo cromosoma como una copia de parent1
		new_cromosome = cp.deepcopy(parent1)

		# Copiamos los genes de parent2 indicados por las probabilidades obtenidas
		new_cromosome[v] = parent2[v]

		return new_cromosome


	def local_search(self, chromosome):

		generated = 0
		improvement = True
		random_index_list = np.array(range(self._dim))
		random.shuffle(random_index_list)
		ril_ind = 0
		fitness = self.get_single_fitness(chromosome)[0]

		#while improvement and generated < self._max_neighbors:
		while generated < self._max_neighbors:

			object_index = random_index_list[ril_ind]
			improvement = False
			original_label = chromosome[object_index]
			other_labels = np.delete(np.array(range(self._result_nb_clust)), original_label)
			random.shuffle(other_labels)

			for label in other_labels:

				generated += 1
				chromosome[object_index] = label
				new_fitness = self.get_single_fitness(chromosome)[0]

				if new_fitness < fitness:

					fitness = new_fitness
					improvement = True
					break

				else:

					chromosome[object_index] = original_label


			if ril_ind == self._dim - 1:

				random.shuffle(random_index_list)
				ril_ind = 0

			else:

				ril_ind += 1

		return chromosome, fitness


	def run(self, max_evals):

		self.init_ils()
		gen_times = []

		while self._evals_done < max_evals:
			start_gen = time.time()
			worst = np.argmax(self._best_fitness)
			best = (worst + 1) % 2

			new_chromosome = self.uniform_crossover_operator(self._best_solution[best], self._best_solution[worst])
			mutant = self.segment_mutation_operator(new_chromosome)
			improved_mutant, improved_mutant_fitness = self.local_search(mutant)

			if improved_mutant_fitness < self._best_fitness[worst]:

				self._best_solution[worst] = mutant
				self._best_fitness[worst] = improved_mutant_fitness

			if self._best_fitness[best] - self._best_fitness[worst] > self._best_fitness[best] * self._xi:

				worst = np.argmax(self._best_fitness)

				self._best_solution[worst, :] = np.random.randint(0, self._result_nb_clust, self._dim)
				self._best_fitness[worst] = self.get_single_fitness(self._best_solution[worst, :])[0]

			gen_times.append(time.time() - start_gen)

		best = np.argmin(self._best_fitness)
		return self._best_solution[best, :], np.sum(gen_times)













