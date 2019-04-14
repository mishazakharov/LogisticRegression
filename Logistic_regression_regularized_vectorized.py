import numpy as np  
import pandas as pd


def step_gradient(old_theta,vektor_x,vektor_y,learningRate,regularizeRate):
	m = len(vektor_y)
	# Вектор тэта
	theta = old_theta
	# Вектор дэльта начальный
	delta = np.zeros((3,1))
	for i in range(0,m):
		# Создание вектора n+1 РАЗМЕРНОСТИ(ДОБАВИЛ x0!)
		x_i = vektor_x
		y = vektor_y
		# Н-размерный вектор, содержащий все признаки для 1 экземпляра!
		x_i = x_i[:3,i].reshape(3,1)
		# Число - целевое значение на m-ом экземпляре!
		y_i = y[i]
		# Аргумент функции гипотезы!
		z = np.dot(theta.T,x_i)
		# Гипотеза логистической регрессии
		hypothesis = 1/(1+np.exp(-z))
		# Расчет констант, почему-то иначе не работает!!!!
		constant = (1/m) * (float(hypothesis) - y_i)
		# Новый вектор дельта!
		delta += (x_i * constant)
		# Регуляризация
		regularize = 1 - ((regularizeRate*learningRate)/m)
	# Формула шага градиентного спуска в векторной форме(ругляризованная)!!!
	new_theta = (old_theta*regularize) - (learningRate * delta)
	return new_theta
	# При вызове функции в качестве old_theta будет передаваться new_theta
	# предыдущего шага, так осуществляется правильное обновление вектора тэта.

def gradient_runner(initial_theta,num_iterations,vektor_x,
								vektor_y,learningRate,regularizeRate):
	theta = initial_theta
	for i in range(num_iterations):
		theta = step_gradient(theta,vektor_x,vektor_y,learningRate,
														regularizeRate)
	return theta

def run():
	# read csv
	data = pd.read_csv('marks.txt')
	# create numpy array out of pandas data
	data = data.values
	# Создание вектора n+1 размерности(ДОБАВИЛ x0!)
	x_i = data.T[:2]
	vektor_x = np.insert(x_i,0,[1],axis=0)
	vektor_y = data[:,2]
	initial_theta = np.zeros((3,1))
	learningRate = 0.001
	# Параметр регуляризации!
	regularizeRate = 0.01
	num_iterations = 3000
	theta = gradient_runner(initial_theta,num_iterations,vektor_x,vektor_y,
												learningRate,regularizeRate)
	print(theta)
	z = np.dot(theta.T,vektor_x[:3,92].reshape(3,1))
	print(z)
	prediction = 1/(1+np.exp(-z))
	print(prediction,'this is my prediction!!')

if __name__ == "__main__":
	run()
