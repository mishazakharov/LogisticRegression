import numpy as np
import pandas as pd
import math

def compute_error(theta_o,theta_one,
				theta_two,points):
	totalError = 0
	N = len(points)
	for i in range(0,len(points)):
		x1 = points[i,0]
		x2 = points[i,1]
		y = points[i,2]
		z = theta_o + theta_one*x1 + theta_two*x2
		hypothesis = 1/(1+np.exp(-z))
		totalError += ((-y) * np.log(hypothesis))
	
def step_gradient(theta_o_current,theta_one_current,theta_two_current,
							points,learningRate):
	theta_o = 0
	theta_one = 0
	theta_two = 0
	N = len(points)
	for i in range(0,N):
		x1 = points[i,0]
		x2 = points[i,1]
		y = points[i,2]

		z = theta_o_current + theta_one_current*x1 + theta_two_current * x2
		hypothesis = 1/(1+np.exp(-z))
		theta_o += (1/N) * (hypothesis - y)
		theta_one += (1/N) * x1 * (hypothesis - y)
		theta_two += (1/N) * x2 * (hypothesis - y)

	new_theta_o = theta_o_current - (learningRate * theta_o)
	new_theta_one = theta_one_current - (learningRate * theta_one)
	new_theta_two = theta_two_current - (learningRate * theta_two)
	return new_theta_o,new_theta_one,new_theta_two

def gradient_runner(starting_o,starting_one,starting_two,num_iterations,
									points,learningRate):
	theta_o = starting_o
	theta_one = starting_one
	theta_two = starting_two
	for i in range(num_iterations):
		theta_o,theta_one,theta_two = step_gradient(theta_o,theta_one,
										theta_two,points,learningRate)
	return theta_o,theta_one,theta_two

def run():
	points = np.genfromtxt('marks.txt',delimiter=',')
	num_iterations = 3000
	learningRate = 0.001
	initial_theta_o = 0
	initial_theta_one = 0
	initial_theta_two = 0
	first_error = compute_error(initial_theta_o,initial_theta_one,
				initial_theta_two,points)
	[theta_o,theta_one,theta_two] = gradient_runner(initial_theta_o,
		initial_theta_one,initial_theta_two,num_iterations,
										points,learningRate)
	z = theta_o + (theta_one*74.492692) + (theta_two *84.845137)
	hypothesis = 1/(1+np.exp(-z))
	end_error = compute_error(theta_o,theta_one,theta_two,points)
	print('1 - ',first_error)
	print('2 - ',end_error)
	print('z - ',z)
	print(hypothesis)
	print(theta_o,theta_one,theta_two)

if __name__ == '__main__':
	run()