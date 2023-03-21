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

grid7 = [[1, 6, 8, 0, 0, 0, 9, 0, 2],
         [0, 0, 0, 3, 0, 1, 0, 0, 0],
         [0, 3, 0, 6, 2, 0, 0, 0, 0],
         [0, 0, 9, 0, 0, 0, 1, 0, 6],
         [0, 0, 1, 0, 0, 0, 3, 7, 0],
         [0, 4, 3, 5, 0, 0, 0, 0, 9],
         [0, 0, 0, 8, 0, 2, 6, 0, 0],
         [0, 0, 0, 9, 0, 5, 0, 2, 3],
         [2, 0, 6, 0, 3, 0, 7, 0, 0]]


grids = [(grid1, 2, 2), (grid2, 2, 2), (grid3, 2, 2), (grid4, 2, 2), (grid5, 2, 2),(grid6, 2, 3),(grid7, 3, 3)]
'''
===================================
DO NOT CHANGE CODE ABOVE THIS LINE
===================================
'''
def check_section(section, n):

	if len(set(section)) == len(section) and sum(section) == sum([i for i in range(n+1)]):
		return True
	return False


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
	n_rows - number of rows in each square
	n_cols - number of columns in each square

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


def recursive_solve_old(grid, n_rows, n_cols):
	# reach the maximum depth of recursion
	# check if the grid is correct
	# if it is, return grid
	# if it is not, move on to the next number

	#N is the maximum integer considered in this board
	n = n_rows*n_cols
	for y in range(n):
		for x in range(n):
			# if at max depth code skipped and moves back up a recursion level as no 0 values are found
			if grid[y][x] == 0:
				for z in range(1,n+1):
					grid[y][x] = z

					# Call recursion to reach maximum depth
					recursive_solve(grid, n_rows, n_cols)
					
					# if the grid has been solved this will pass the results all the way out of the recursion
					if check_solution(grid, n_rows, n_cols):
						return grid
				
				# if none of the numbers 1 to n are correct, reset the value of the cell to 0 and return the grid up a recursion level
				grid[y][x] = 0
				return grid

	
def recursive_solve(grid,n_rows,n_cols): 
    '''
	This function is used to solve a sudoku board using recursion

	args: grid - representation of a suduko board as a nested list.
	n_rows - number of rows in each square
	n_cols - number of columns in each square

	returns: grid - representation of a suduko board as a nested list.
	'''
	# n is the maximum integer considered in this board
    n = n_rows*n_cols 
    
    # find the next zero in the grid
    zeros = find_zeros(grid)

	# if there are no zeros left in the grid then the grid is solved and this will start passing results up the recursion	
    if not zeros:	
        return grid 
    else:
		# row and col are the coordinates of the next zero in the grid
        row, col = zeros 

	# try all numbers 1 to n
    for i in range(1,n+1): 
	
		# check if the number is possible in the current position
		# if it is possible, set the value of the cell to the number and call recursion
        if possible(grid, i, (row, col),n_rows,n_cols): 
            grid[row][col] = i 

			# call recursion to find next zero
			# if the grid has been solved this will pass the results all the way out of the recursion
            if recursive_solve(grid,n_rows,n_cols): 
                return grid	

			# if the number is not possible, reset the value of the cell to 0 and try the next number
            grid[row][col] = 0 

	# return False to move back up the recursion if none of the numbers 1 to n are possible
    return False 


def possible(grid, num, pos,n_rows,n_cols):
    '''
	This function is used to check whether a number is possible in a given position on a sudoku board

	args: grid - representation of a suduko board as a nested list.
	num - the number to be checked
	pos - the coordinates of the position to be checked
	n_rows - number of rows in each square
	n_cols - number of columns in each square

	returns: True (possible) or False (not possible)
	'''
    # Check row
    for i in range(len(grid[0])):
        if grid[pos[0]][i] == num and pos[1] != i:
            return False

    # Check column
    for i in range(len(grid)):
        if grid[i][pos[1]] == num and pos[0] != i:
            return False

	# Check sub grid
	# find the coordinates of the top left corner of the sub grid
    grid_x = pos[1] // n_cols
    grid_y = pos[0] // n_rows

	# check each cell in the sub grid to see if the number is already present
    for i in range(grid_y*n_rows, grid_y*n_rows + n_rows):
        for j in range(grid_x * n_cols, grid_x*n_cols + n_cols):
            if grid[i][j] == num and (i,j) != pos:
                return False

	# if the number is not present in the row, column or sub grid then it is possible
    return True


def find_zeros(grid):
    '''
	This function is used to find the next zero in a sudoku board

	args: grid - representation of a suduko board as a nested list.

	returns: (row, col) - the coordinates of the next zero in the grid
	none - if there are no zeros left in the grid
	'''
	# loop through each cell in the grid
    for i in range(len(grid)):
        for j in range(len(grid[0])):
			
			# if a zero is found return the coordinates
            if grid[i][j] == 0:
                return (i, j)  # row, col

    return None
				
def random_solve(grid, n_rows, n_cols, max_tries=500):

	return grid


def solve(grid, n_rows, n_cols):

	'''
	Solve function for Sudoku coursework.
	Comment out one of the lines below to either use the random or recursive solver
	'''
	
	#return random_solve(grid, n_rows, n_cols)
	return recursive_solve(grid, n_rows, n_cols)


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
		print(solution)
		if check_solution(solution, n_rows, n_cols):
			print("grid %d correct" % (i+1))
			points = points + 10
		else:
			print("grid %d incorrect" % (i+1))

	print("====================================")
	print("Test script complete, Total points: %d" % points)


if __name__ == "__main__":
	main()