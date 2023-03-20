import copy
import time

#Grids 1-5 are 2x2
grid1 = [
		[1, 0, 4, 2],
		[4, 2, 1, 3],
		[2, 1, 3, 4],
		[3, 4, 2, 1]]

grid2 = [
		[1, 0, 4, 2],
		[4, 2, 1, 3],
		[2, 1, 0, 4],
		[3, 4, 2, 1]]

grid3 = [
		[1, 0, 4, 2],
		[4, 2, 1, 0],
		[2, 1, 0, 4],
		[0, 4, 2, 1]]

grid4 = [
		[1, 0, 4, 2],
		[0, 2, 1, 0],
		[2, 1, 0, 4],
		[0, 4, 2, 1]]

grid5 = [
		[1, 0, 0, 2],
		[0, 0, 1, 0],
		[0, 1, 0, 4],
		[0, 0, 0, 1]]

grid6 = [
		[0, 0, 6, 0, 0, 3],
		[5, 0, 0, 0, 0, 0],
		[0, 1, 3, 4, 0, 0],
		[0, 0, 0, 0, 0, 6],
		[0, 0, 1, 0, 0, 0],
		[0, 5, 0, 0, 6, 4]]

grid7 = [
	    [1, 6, 8, 0, 0, 0, 9, 0, 2],
        [0, 0, 0, 3, 0, 1, 0, 0, 0],
        [0, 3, 0, 6, 2, 0, 0, 0, 0],
        [0, 0, 9, 0, 0, 0, 1, 0, 6],
        [0, 0, 1, 0, 0, 0, 3, 7, 0],
        [0, 4, 3, 5, 0, 0, 0, 0, 9],
        [0, 0, 0, 8, 0, 2, 6, 0, 0],
        [0, 0, 0, 9, 0, 5, 0, 2, 3],
        [2, 0, 6, 0, 3, 0, 7, 0, 0]
	    ]

#grids = [(grid1, 2, 2), (grid2, 2, 2), (grid3, 2, 2), (grid4, 2, 2), (grid5, 2, 2), (grid6, 2, 3), (grid7, 3, 3)]
grids = [(grid7, 3, 3)]

'''
===================================
DO NOT CHANGE CODE ABOVE THIS LINE
===================================
'''
import csv

def check_section(section, n):

	if len(set(section)) == len(section) and sum(section) == sum([i for i in range(n+1)]):
		return True
	return False

def reader(data_file): # function to read the data from the csv file
    with open(data_file) as data:
        all_data = csv.reader(data) # read the data
        return list(all_data)   # return the data as a list

def get_squares(grid, n_rows, n_cols):
	squares = []
	for i in range(n_cols):
		rows = (i*n_rows, (i+1)*n_rows)
		for j in range(n_rows):
			cols = (j*n_cols, (j+1)*n_cols)
			square = []
			for k in range(rows[0], rows[1]):
				line = grid[k][cols[0]:cols[1]]
				square +=line
			squares.append(square)
	return(squares)

def check_solution(grid, n_rows, n_cols):
	'''
	This function is used to check whether a sudoku board has been correctly solved
	args: grid - representation of a suduko board as a nested list.
	returns: True (correct solution) or False (incorrect solution)
	'''
	n = n_rows*n_cols

	for row in grid:
		if check_section(row, n) == False:
			return False

	for i in range(n_rows):
		column = []
		for row in grid:
			column.append(row[i])
		if check_section(column, n) == False:
			return False

	squares = get_squares(grid, n_rows, n_cols)
	for square in squares:
		if check_section(square, n) == False:
			return False

	return True

# function to solve the sudoku board
def recursive_solve(grid, n_rows, n_cols, priority_array):
	'''A recursive function to both enter and test possible values in the grid
	Inputs:
	grid: initial grid to solve
	n_rows: number of boxes horizontally
	n_cols: number of boxes vertically'''
	#N is the maximum integer considered in this board
	n = n_rows*n_cols
	if priority_array:
		row = priority_array[0][0]
		column = priority_array[0][1]
		for k in priority_array[0][2]: # k is the number we are trying to put in the cell
			if valid(grid, row, column, k, n_rows, n_cols): # test that the value entered could be part of a valid solution
				grid[row][column] = k # we put k in the cell
				recursive_solve(grid, n_rows, n_cols, priority_array[1:]) # we call the function recursively
				if check_solution(grid, n_rows, n_cols): # if the grid is correct, we return it
					return(grid)
		grid[row][column] = 0 # if we have tried all the numbers and none of them work, we return the grid to its original state
	return(grid)
# we return the grid if it is already solved

# we check if the number is valid in the row, column and box
def valid(grid, row_index, column_index, number, n_rows, n_cols):
	'''Tests if the number to be input is a valid input to the existing grid, i.e. does not clash with another value in the row, column, or box
	Inputs: 
		grid: current grid to be tested against
		row_index, column_index: location of value in grid
		number: value of number to be tested
		n_rows: number of boxes horizontally
		n_cols: number of boxes vertically
	Outputs: 
		Boolean True/False'''
	if number in grid[row_index]: # if the number is already in the row, it is not valid
		return False
	column = [grid[n][column_index] for n in range(0,len(grid))] # we create a list of the numbers in the column
	if number in column: # if the number is already in the column, it is not valid
		return False
	boxes = get_squares(grid,n_rows,n_cols) # we create a list of the numbers in the box
	box = boxes[n_rows*(row_index//n_rows)+column_index//n_cols] # we find the box in which the cell is
	if number in box: # if the number is already in the box, it is not valid
		return False
	return True # if the number is not in the row, column or box, it is valid

def priority_length(term):
	return(len(term[2]))

def to_tuple(lst):
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)

def create_priority(grid, n_rows, n_cols):
	'''Creates an array of the number of valid values & a separate array with empty values replaced with an array of all valid values
    Inputs: 
        grid: current grid to be solved
	    n_rows: number of boxes horizontally
		n_cols: number of boxes vertically
	Outputs:
        priority_array: array of coordinates in the format [row_no, col_no, possible_values_count]
	    valid_array: array of original grid with all 0-values replaced with viable values'''
	priority_array = []
	valid_array = []
	for line in grid:
		valid_array.append(list(line))
	n = n_rows*n_cols
	grid = to_tuple(grid) # Tuple is created here
	for row in range(0, n): # i is the row
		for column in range(0, n): # j is the column
			if grid[row][column] == 0: # if the cell is empty
				valid_array[row][column] = []
				priority_array.append([row, column, []])
				for value in range(1, n+1): # k is the number we are trying to put in the cell
					if valid(grid, row, column, value, n_rows, n_cols): # test that the value entered could be part of a valid solution
						valid_array[row][column].append(value)
						priority_array[-1][2].append(value)
	priority_array.sort(key=priority_length)
	return priority_array, valid_array

def solve(grid, n_rows, n_cols):
	'''
	Solve function for Sudoku coursework.
	Comment out one of the lines below to either use the random or recursive solver
	'''
	priority_array, valid_array = create_priority(grid, n_rows, n_cols)
	#return random_solve(grid, n_rows, n_cols)
	return recursive_solve(grid, n_rows, n_cols, priority_array)


'''
===================================
DO NOT CHANGE CODE BELOW THIS LINE
===================================
'''
def main():

	points = 0

	print("Running test script for coursework 1")
	print("====================================")
	
	for (i, (grid, n_rows, n_cols)) in enumerate(grids):
		print("Solving grid: %d" % (i+1))
		start_time = time.time()
		solution = solve(grid, n_rows, n_cols)
		elapsed_time = time.time() - start_time
		print("Solved in: %f seconds" % elapsed_time)
		for line in solution:
			print(line)
		if check_solution(solution, n_rows, n_cols):
			print("grid %d correct" % (i+1))
			points = points + 10
		else:
			print("grid %d incorrect" % (i+1))

	print("====================================")
	print("Test script complete, Total points: %d" % points)


if __name__ == "__main__":
	main()