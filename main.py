import time
import csv
import argparse
import copy
import timeit
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import random
import itertools
import sys
from dataclasses import dataclass

# Grids 1-5 are 2x2
grid1 = [[1, 0, 4, 2], [4, 2, 1, 3], [2, 1, 3, 4], [3, 4, 2, 1]]

grid2 = [[1, 0, 4, 2], [4, 2, 1, 3], [2, 1, 0, 4], [3, 4, 2, 1]]

grid3 = [[1, 0, 4, 2], [4, 2, 1, 0], [2, 1, 0, 4], [0, 4, 2, 1]]

grid4 = [[1, 0, 4, 2], [0, 2, 1, 0], [2, 1, 0, 4], [0, 4, 2, 1]]

grid5 = [[1, 0, 0, 2], [0, 0, 1, 0], [0, 1, 0, 4], [0, 0, 0, 1]]

grid6 = [
    [0, 0, 6, 0, 0, 3],
    [5, 0, 0, 0, 0, 0],
    [0, 1, 3, 4, 0, 0],
    [0, 0, 0, 0, 0, 6],
    [0, 0, 1, 0, 0, 0],
    [0, 5, 0, 0, 6, 4],
]

grid7 = [
    [1, 6, 8, 0, 0, 0, 9, 0, 2],
    [0, 0, 0, 3, 0, 1, 0, 0, 0],
    [0, 3, 0, 6, 2, 0, 0, 0, 0],
    [0, 0, 9, 0, 0, 0, 1, 0, 6],
    [0, 0, 1, 0, 0, 0, 3, 7, 0],
    [0, 4, 3, 5, 0, 0, 0, 0, 9],
    [0, 0, 0, 8, 0, 2, 6, 0, 0],
    [0, 0, 0, 9, 0, 5, 0, 2, 3],
    [2, 0, 6, 0, 3, 0, 7, 0, 0],
]

grid8 = [
    [0, 0, 0, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 0, 8, 0, 4, 0, 0],
    [0, 9, 4, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 9],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 8, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 7],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]

# 4x4 grid 1, 2 missing digits
grid9 = [[1, 2, 0, 0], [3, 0, 0, 0], [0, 0, 1, 2], [2, 1, 0, 0]]

# 4x4 grid 2, 4 missing digits
grid10 = [[2, 1, 0, 0], [0, 3, 0, 0], [1, 0, 0, 0], [0, 0, 0, 4]]

# 4x4 grid 3, 6 missing digits
grid11 = [[0, 2, 0, 0], [0, 0, 0, 4], [1, 0, 0, 0], [0, 0, 3, 0]]

# 4x4 grid 4, 8 missing digits
grid12 = [[0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

# 4x4 grid 5, 10 missing digits
grid13 = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

# 6x6 grid 1
grid14 = [
    [1, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0],
    [0, 0, 3, 0, 0, 0],
    [0, 0, 0, 4, 0, 0],
    [0, 0, 0, 0, 5, 0],
    [0, 0, 0, 0, 0, 6],
]
# 6x6 grid 4, 18 missing digits
grid15 = [
    [0, 0, 0, 5, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [6, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 4, 0],
    [0, 0, 4, 0, 0, 0],
    [0, 0, 0, 0, 0, 6],
]

# 6x6 grid 5, 21 missing digits
grid16 = [
    [0, 0, 0, 0, 0, 0],
    [5, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [6, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]

# 9x9 grid 4, 40 missing digits
grid17 = [
    [0, 0, 0, 2, 6, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 8, 0, 0, 1],
    [0, 8, 0, 0, 0, 0, 2, 3, 0],
    [0, 0, 6, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 9, 0, 7, 0],
    [0, 5, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 4, 0, 0, 0],
    [0, 1, 0, 6, 0, 0, 0, 0, 0],
    [0, 0, 7, 0, 0, 0, 0, 5, 0],
]

# 9x9 grid 5, 45 missing digits
grid18 = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]

grids = [
    (grid1, 2, 2),
    (grid2, 2, 2),
    (grid3, 2, 2),
    (grid4, 2, 2),
    (grid5, 2, 2),
    (grid6, 2, 3),
    (grid7, 3, 3),
    (grid8, 3, 3),
    (grid9, 2, 2),
    (grid10, 2, 2),
    (grid11, 2, 2),
    (grid12, 2, 2),
    (grid13, 2, 2),
    (grid14, 2, 3),
    (grid15, 2, 3),
    (grid16, 2, 3),
    (grid17, 3, 3),
    (grid18, 3, 3),
]


class TooManyHintsError(Exception):
    """Raised when user asks for more hints than there are zeroes in Sudoku grid"""

    pass


@dataclass
class ProfileResults:
    """Class for storing parameters to plot profiling results"""

    difficulty: int
    n_rows: int
    n_cols: int
    timeit_results: list


def _getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--explain",
        help="Provide set of instructions for solving grid",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--file",
        nargs=2,
        metavar=("INPUT", "OUTPUT"),
        help="Input file for grid to solve and output file for solved grid. If 'explain' is set then include explanations",
    )
    parser.add_argument(
        "--hint",
        help="Integer value for number of values to fill",
        default=None,
    )
    parser.add_argument(
        "--profile",
        help="Measure performance and produce plots using 'timeit'",
        default=False,
        action=argparse.BooleanOptionalAction,
    )

    args = parser.parse_args()
    return args


def check_section(section, n):
    if len(set(section)) == len(section) and sum(section) == sum(
        [i for i in range(n + 1)]
    ):
        return True
    return False


def get_squares(grid, n_rows, n_cols):
    squares = []
    for i in range(n_cols):
        rows = (i * n_rows, (i + 1) * n_rows)
        for j in range(n_rows):
            cols = (j * n_cols, (j + 1) * n_cols)
            square = []
            for k in range(rows[0], rows[1]):
                line = grid[k][cols[0] : cols[1]]
                square += line
            squares.append(square)
    return squares


def check_solution(grid, n_rows, n_cols):
    """
    This function checks whether a sudoku board has been correctly solved.

    args: grid - representation of a suduko board as a nested list
    returns: True (correct solution) or False (incorrect solution)
    """
    n = n_rows * n_cols

    for row in grid:
        if check_section(row, n) is False:
            return False

    for i in range(n_rows):
        column = []
        for row in grid:
            column.append(row[i])
        if check_section(column, n) is False:
            return False

    squares = get_squares(grid, n_rows, n_cols)
    for square in squares:
        if check_section(square, n) is False:
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
	if priority_array:
		row = priority_array[0][0]
		column = priority_array[0][1]
		for value in priority_array[0][2]: # k is the number we are trying to put in the cell
			if valid(grid, row, column, value, n_rows, n_cols): # test that the value entered could be part of a valid solution
				grid[row][column] = value # we put k in the cell
				recursive_solve(grid, n_rows, n_cols, priority_array[1:]) # we call the function recursively
				if check_solution(grid, n_rows, n_cols): # if the grid is correct, we return it
					return(grid)
		grid[row][column] = 0 # if we have tried all the numbers and none of them work, we return the grid to its original state
	return(grid)
# we return the grid if it is already solved


# we check if the number is valid in the row, column and box
def valid(grid, row_index, column_index, number, n_rows, n_cols):
    """
    Tests if the number to be input is a valid input to the existing grid, i.e. does not clash with another value in the row, column, or box.

    Inputs:
            grid: current grid to be tested against
            row_index, column_index: location of value in grid
            number: value of number to be tested
            n_rows: number of boxes horizontally
            n_cols: number of boxes vertically
    Outputs:
            Boolean True/False"""
    if (
        number in grid[row_index]
    ):  # if the number is already in the row, it is not valid
        return False
    column = [
        grid[n][column_index] for n in range(0, len(grid))
    ]  # we create a list of the numbers in the column
    if (
        number in column
    ):  # if the number is already in the column, it is not valid
        return False
    boxes = get_squares(
        grid, n_rows, n_cols
    )  # we create a list of the numbers in the box
    box = boxes[
        n_rows * (row_index // n_rows) + column_index // n_cols
    ]  # we find the box in which the cell is
    if number in box:  # if the number is already in the box, it is not valid
        return False
    return True  # if the number is not in the row, column or box, it is valid


def priority_length(term):
    return len(term[2])


def to_tuple(lst):
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)


def create_priority(grid, n_rows, n_cols):
    """
    Creates an array of the number of valid values and a separate array with empty values replaced with an array of all valid values.

    Inputs:
        grid: current grid to be solved
        n_rows: number of boxes horizontally
        n_cols: number of boxes vertically
    Outputs:
        priority_array: array of coordinates in the format [row_no, col_no, possible_values_count]
        valid_array: array of original grid with all 0-values replaced with viable values
    """
    priority_array = []
    valid_array = []
    for line in grid:
        valid_array.append(list(line))
    n = n_rows * n_cols
    grid = to_tuple(grid)  # Tuple is created here
    for row in range(0, n):  # i is the row
        for column in range(0, n):  # j is the column
            if grid[row][column] == 0:  # if the cell is empty
                valid_array[row][column] = []
                priority_array.append([row, column, []])
                for value in range(
                    1, n + 1
                ):  # k is the number we are trying to put in the cell
                    if valid(
                        grid, row, column, value, n_rows, n_cols
                    ):  # test that the value entered could be part of a valid solution
                        valid_array[row][column].append(value)
                        priority_array[-1][2].append(value)
    priority_array.sort(key=priority_length)
    return priority_array, valid_array


def solve(grid, n_rows, n_cols):
    """
    Solve function for Sudoku coursework.
    Comment out one of the lines below to either use the random or recursive solver
    """

    priority_array, valid_array = create_priority(grid, n_rows, n_cols)
    solved_grid = recursive_solve(grid, n_rows, n_cols, priority_array)
    return solved_grid


def explain(original_grid, solved_grid, to_terminal):
    changes = []
    for index, row in enumerate(original_grid):
        changes.append(
            [
                (i, updated)
                for i, (zero, updated) in enumerate(
                    zip(row, solved_grid[index])
                )
                if zero != updated
            ]
        )
    if to_terminal:
        for row_number, row in enumerate(changes):
            for element in row:
                print(
                    f"Put {element[0]} in location ({row_number}, {element[1]})"
                )
    return changes

def plot(results):
    import matplotlib.pyplot as plt

    plt.style.use("ggplot")
    plt.figure(figsize=(10, 5))
    plt.title("Time taken to solve Sudoku grids")
    plt.xlabel("Difficulty")
    plt.ylabel("Time taken (s)")
    plt.xticks(range(0, 81, 5))
    plt.yticks(range(0, 2))
    plt.grid(True)
    grid_number = 0
    for grid in results: # we plot the results for each grid
        grid_number += 1
        difficulty = grid['difficulty'] # we get the difficulty of the grid
        average_time = sum(grid['results']) / len(grid['results']) # we calculate the average time taken to solve the grid
        print(difficulty, average_time)            
        plt.bar(difficulty, average_time, label=f"Grid {grid_number}: {grid['n_rows']}x{grid['n_cols']}", width=0.5)
    plt.legend(loc="upper right")
    plt.show()
    
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

def plot1(results):
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Time taken to solve Sudoku puzzles")
    ax.set_xlabel("Number of missing values")
    ax.set_ylabel("Time taken (s)")
    ax.set_xticks(range(0, 81, 5))
    ax.set_yticks(np.arange(0, 2.5, 0.25))
    ax.grid(True)
    
    # Define color map for difficulty levels
    cmap = plt.get_cmap("viridis")
    
    grid_number = 0
    for grid in results: # we plot the results for each grid
        grid_number += 1
        difficulty = grid['difficulty'] # we get the difficulty of the grid
        average_time = np.mean(grid['results']) # we calculate the average time taken to solve the grid
        std_dev = np.std(grid['results']) # we calculate the standard deviation of the time taken to solve the grid
        
        # Add error bars to bar plot
        ax.bar(difficulty, average_time, yerr=std_dev, capsize=5, 
               color=cmap(difficulty/80), label=f"Grid {grid_number}: {grid['n_rows']}x{grid['n_cols']}")
    
    # Increase font size of axis labels
    ax.tick_params(axis="both", labelsize=12)
    
    # Set y-axis to logarithmic scale
    ax.set_yscale("log")
    
    # Add subtitle with additional information about the data
    fig.text(0.5, 0.001, "Data based on 18 Sudoku puzzles solved 7 times each using a recursive algorithm", ha="center", fontsize=12)
    
    # Add color bar to indicate difficulty levels
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=80))
    cbar = fig.colorbar(sm, ax=ax, ticks=range(0, 81, 10), shrink=0.8)
    cbar.set_label("Difficulty", fontsize=12)
    
    plt.legend(bbox_to_anchor=(1.2, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
def to_file(args, solved_grid, changes):
    with open(args.file[1], "w") as output:
        output.write("Solved Grid:\n")
        writer = csv.writer(output)
        writer.writerows(solved_grid)
    if args.explain:
        with open(args.file[1], "a") as output:
            output.write("\n\nExplanation:\n")
            for row_number, row in enumerate(changes):
                for element in row:
                    output.write(
                        f"Put {element[0]} in location ({row_number}, {element[1]})\n"
                    )


def hint(hints, solved_grid, original_grid, n_rows, n_cols):
    ranges = [range(0, n_rows * n_cols) for i in range(2)]
    perms = list(itertools.product(*ranges))
    valid_perms = [
        perm for perm in perms if original_grid[perm[0]][perm[1]] == 0
    ]
    if hints > sum(row.count(0) for row in original_grid):
        raise TooManyHintsError
    chosen_hints = random.sample(valid_perms, hints)
    grid_to_show = copy.deepcopy(original_grid)
    for perm in chosen_hints:
        grid_to_show[perm[0]][perm[1]] = solved_grid[perm[0]][perm[1]]
    return grid_to_show


# function to plot the results of the profiling as a bar chart

def plot2(results):

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Time taken to solve Sudoku puzzles")
    ax.set_xlabel("Number of missing values")
    ax.set_ylabel("Time taken (s)")
    ax.set_xticks(range(0, 81, 5))
    ax.set_yticks(np.arange(0, 2.5, 0.25))
    ax.grid(True)

    # Define color map for difficulty levels
    cmap = plt.get_cmap("viridis")

    grid_number = 0
    for grid in results: # we plot the results for each grid
        grid_number += 1
        difficulty = grid['difficulty'] # we get the difficulty of the grid
        average_time = np.mean(grid['results']) # we calculate the average time taken to solve the grid
        std_dev = np.std(grid['results']) # we calculate the standard deviation of the time taken to solve the grid

        # Add error bars to scatter plot
        ax.errorbar(difficulty, average_time, yerr=std_dev, fmt='x', capsize=5, color=cmap(difficulty/80), label=f"Grid {grid_number}: {grid['n_rows']}x{grid['n_cols']}")
    
	 # Compute and plot line of best fit
    x = [grid['difficulty'] for grid in results]
    y = [np.mean(grid['results']) for grid in results]
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    ax.plot(x, slope*np.array(x)+intercept, color='black', label=f"Line of best fit: y={slope:.3f}x+{intercept:.3f}")

    # Increase font size of axis labels
    ax.tick_params(axis="both", labelsize=12)

    # Set y-axis to logarithmic scale
    ax.set_yscale("log") # does this need changing? ###### because the one poor result

    # Add subtitle with additional information about the data
    fig.text(0.5, 0.001, "Data based on 18 Sudoku puzzles solved 7 times each using a recursive algorithm", ha="center", fontsize=12)

    # Add color bar to indicate difficulty levels
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=80))
    cbar = fig.colorbar(sm, ax=ax, ticks=range(0, 81, 10), shrink=0.8)
    cbar.set_label("Difficulty", fontsize=12)

    plt.legend(bbox_to_anchor=(1.2, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    


def plot(results):
    import matplotlib.pyplot as plt

    plt.style.use("ggplot")
    plt.figure(figsize=(10, 5))
    plt.title("Time taken to solve Sudoku grids")
    plt.xlabel("Difficulty")
    plt.ylabel("Time taken (s)")
    plt.xticks(range(0, 81, 5))
    plt.yticks(range(0, 2))
    plt.grid(True)
    grid_number = 0
    for grid in results:  # we plot the results for each grid
        grid_number += 1
        difficulty = grid["difficulty"]  # we get the difficulty of the grid
        average_time = sum(grid["results"]) / len(
            grid["results"]
        )  # we calculate the average time taken to solve the grid
        print(difficulty, average_time)
        plt.bar(
            difficulty,
            average_time,
            label=f"Grid {grid_number}: {grid['n_rows']}x{grid['n_cols']}",
            width=0.5,
        )
    plt.legend(loc="upper right")
    plt.show()


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress


def profiling(grid, n_cols, n_rows, repeat):
    SETUP = """
import copy
grid_to_test = copy.deepcopy(grid)
"""  # Deepcopy required to prevent mutation of grid variable for subsequent runs. Setup code is not included in execution time.

    STMT = """
priority_array, valid_array = create_priority(grid_to_test, n_rows, n_cols)
solved_grid = recursive_solve(grid_to_test, n_rows, n_cols, priority_array)
"""

    difficulty = sum(row.count(0) for row in grid)
    results = timeit.repeat(
        stmt=STMT,
        setup=SETUP,
        repeat=repeat,
        number=1,
        globals={
            "grid": grid,
            "n_rows": n_cols,
            "n_cols": n_rows,
            "create_priority": create_priority,
            "recursive_solve": recursive_solve,
        },
    )
    return ProfileResults(difficulty, n_rows, n_cols, results)


def barplot(results, repeats):
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Time taken to solve Sudoku puzzles")
    ax.set_xlabel("Number of missing values")
    ax.set_ylabel("Time taken (s)")
    ax.set_xticks(range(0, 81, 5))
    ax.set_yticks(np.arange(0, 2.5, 0.25))
    ax.grid(True)

    # Define color map for difficulty levels
    cmap = plt.get_cmap("rainbow")

    # sorting the grids by difficulty
    results.sort(key=lambda x: x["difficulty"])

    grid_number = 0
    for grid in results:  # we plot the results for each grid
        grid_number += 1
        difficulty = grid.difficulty  # we get the difficulty of the grid
        average_time = np.mean(
            grid.timeit_results
        )  # we calculate the average time taken to solve the grid
        std_dev = np.std(
            grid.timeit_results
        )  # we calculate the standard deviation of the time taken to solve the grid

        # Adding error bars to bar plot
        ax.bar(
            difficulty,
            average_time,
            yerr=std_dev,
            capsize=5,
            color=cmap(difficulty / 80),
            label=f"Grid {grid_number}: {grid.n_rows}x{grid.n_cols}",
        )

    # Increase font size of axis labels
    ax.tick_params(axis="both", labelsize=12)

    # Set y-axis to logarithmic scale
    ax.set_yscale("log")

    # Add subtitle with additional information about the data
    fig.text(
        0.5,
        0.001,
        f"Data based on 18 Sudoku puzzles solved {repeats} times each using a recursive algorithm",
        ha="center",
        fontsize=12,
    )

    # Add color bar to indicate difficulty levels
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=80))
    cbar = fig.colorbar(sm, ax=ax, ticks=range(0, 81, 10), shrink=0.8)
    cbar.set_label("Difficulty", fontsize=12)

    # Extract the x and y values from the results
    x = np.array([grid.difficulty for grid in results])
    y = np.array([np.mean(grid.timeit_results) for grid in results])

    # sorting x and y values by ascending y values
    x = x[np.argsort(x)]
    y = y[np.argsort(x)]

    # Fit a logarithmic polynomial to the data
    z = np.polyfit(np.log(x), np.log(y), 1)
    f = np.poly1d(z)

    # Plot the logarithmic line of best fit
    ax.plot(
        x,
        np.exp(f(np.log(x))),
        color="black",
        linestyle="--",
        label="Line of best fit (log-log)",
    )
    # Add legend
    plt.legend(bbox_to_anchor=(1.2, 1), loc="upper left")
    plt.tight_layout()
    plt.show()  # we show the plot


# creating a similar plot using a scatter graph


def plot2(results, repeats):
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Time taken to solve Sudoku puzzles")
    ax.set_xlabel("Number of missing values")
    ax.set_ylabel("Time taken (s)")
    ax.set_xticks(range(0, 81, 5))
    ax.set_yticks(np.arange(0, 2.5, 0.25))
    ax.grid(True)

    # Define color map for difficulty levels
    cmap = plt.get_cmap("viridis")

    grid_number = 0  # TODO Remove this and use enumerate
    for grid in results:  # we plot the results for each grid
        grid_number += 1
        difficulty = grid.difficulty  # we get the difficulty of the grid
        average_time = np.mean(
            grid.timeit_results
        )  # we calculate the average time taken to solve the grid
        std_dev = np.std(
            grid.timeit_results
        )  # we calculate the standard deviation of the time taken to solve the grid

        # Add error bars to scatter plot
        ax.errorbar(
            difficulty,
            average_time,
            yerr=std_dev,
            fmt="x",
            capsize=5,
            color=cmap(difficulty / 80),
            label=f"Grid {grid_number}: {grid.n_rows}x{grid.n_cols}",
        )

    # Compute and plot line of best fit
    x = [grid.difficulty for grid in results]
    y = [np.mean(grid.timeit_results) for grid in results]
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    ax.plot(
        x,
        slope * np.array(x) + intercept,
        color="black",
        label=f"Line of best fit: y={slope:.3f}x+{intercept:.3f}",
    )

    # Increase font size of axis labels
    ax.tick_params(axis="both", labelsize=12)

    # Set y-axis to logarithmic scale
    ax.set_yscale(
        "log"
    )  # does this need changing? ###### because the one poor result

    # Add subtitle with additional information about the data
    fig.text(
        0.5,
        0.001,
        f"Data based on 18 Sudoku puzzles solved {repeats} times each using a recursive algorithm",
        ha="center",
        fontsize=12,
    )

    # Add color bar to indicate difficulty levels
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=80))
    cbar = fig.colorbar(sm, ax=ax, ticks=range(0, 81, 10), shrink=0.8)
    cbar.set_label("Difficulty", fontsize=12)

    plt.legend(bbox_to_anchor=(1.2, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def _main():
    points = 0
    args = _getArgs()
    if args.file:
        dims = {"4": (2, 2), "6": (2, 3), "9": (3, 3)}
        with open(args.file[0], "r", newline="") as gridfile:
            reader = csv.reader(gridfile)
            grid_input = [
                [int(value) for value in lst] for lst in list(reader)
            ]
            solution = solve(
                copy.deepcopy(grid_input), *dims[str(len(grid_input))]
            )
        if args.hint:
            try:
                solution = hint(
                    int(args.hint),
                    solution,
                    grid_input,
                    *dims[str(len(grid_input))],
                )
            except TooManyHintsError:
                print(
                    f"Error, number of hints requested is greater than the number of zeroes present in grid"
                )
                sys.exit(1)
        if args.explain:
            changes = explain(grid_input, solution, False)
        else:
            changes = None
        to_file(args, solution, changes)
    else:
        original_grids = [
            (grid1, 2, 2),
            (grid2, 2, 2),
            (grid3, 2, 2),
            (grid4, 2, 2),
            (grid5, 2, 2),
            (grid6, 2, 3),
            (grid7, 3, 3),
            (grid8, 3, 3),
            (grid9, 2, 2),
            (grid10, 2, 2),
            (grid11, 2, 2),
            (grid12, 2, 2),
            (grid13, 2, 2),
            (grid14, 2, 3),
            (grid15, 2, 3),
            (grid16, 2, 3),
            (grid17, 3, 3),
            (grid18, 3, 3),
        ]
        grids = copy.deepcopy(original_grids)
        print("Running test script for coursework 1")
        print("====================================")
        for i, (grid, n_rows, n_cols) in enumerate(grids):
            print("Solving grid: %d" % (i + 1))
            start_time = time.time()
            solution = solve(grid, n_rows, n_cols)
            elapsed_time = time.time() - start_time
            print("Solved in: %f seconds" % elapsed_time)
            if args.hint:
                try:
                    solution = hint(
                        int(args.hint), solution, *original_grids[i]
                    )
                except TooManyHintsError:
                    print(
                        f"Error, number of hints requested is greater than the number of zeroes present in grid {i+1}"
                    )
                    sys.exit(1)
            for line in solution:
                print(line)
            if check_solution(solution, n_rows, n_cols):
                print("grid %d correct" % (i + 1))
                points = points + 10
                if args.explain:
                    explain(original_grids[i][0], solution, True)
            else:
                print("grid %d incorrect" % (i + 1))
        if args.profile:
            repeats = 10
            profiling_results = [
                profiling(grid, n_rows, n_cols, repeats)
                for _, (grid, n_rows, n_cols) in enumerate(original_grids)
            ]
            barplot(profiling_results, repeats)

        print("====================================")
        print("Test script complete, Total points: %d" % points)


if __name__ == "__main__":
    _main()
