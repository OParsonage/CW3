import copy
import random


def check_section(section: list[int], n: int) -> bool:
    """
    Checks the validity of a section of Sudoku.

    Args:
        section (list[int]): List containing the values for a section
        n (int): Dimension of Sudoku grid

    Returns
        bool: True if section is valid
    """
    return len(set(section)) == len(section) and sum(section) == 0.5 * n * (n + 1)


def get_squares(grid: list[list[int]], n_rows: int, n_cols: int) -> list[list[int]]:
    """
    Creates a nested list containing the squares for a Sudoku grid.

    Args:
        grid (list[list[int]]): Grid in its current state
        n_rows (int): Number of rows in grid
        n_cols (int): Number of columns in grid

    Returns:
        list[list[int]]: Nested list containing the squares of a Sudoku grid.
    """
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


def check_solution(grid: list[list[int]], n_rows: int, n_cols: int) -> bool:
    """
    Checks if a Sudoku grid has been correctly solved.

    Args:
        grid (list[list[int]]): Grid in its current state
        n_rows (int): Number of rows in grid
        n_cols (int): Number of columns in grid

    Returns:
        bool: True if solution is correct
    """
    n = n_rows * n_cols
    for row in grid:
        if not check_section(row, n):
            return False
    for i in range(n_rows):
        column = [row[i] for row in grid]  # Comprehension creates a list containing the values in a column
        if not check_section(column, n):
            return False
    squares = get_squares(grid, n_rows, n_cols)
    for square in squares:
        if not check_section(square, n):
            return False
    return True


# we check if the number is valid in the row, column and box
def valid(
    grid: list[list[int]],
    row_index: int,
    column_index: int,
    number: int,
    n_rows: int,
    n_cols: int,
) -> bool:
    """
    Tests if the number to be input is a valid input to the existing grid,
    i.e. does not clash with another value in the row, column, or box.

    Args:
            grid (list[list[int]]): Grid in its current state
            row_index (int): Index of row where value is to be input into
            column_index (int): Index of column where value is to be input into
            number (int): Value to be tested for validity
            n_rows (int): Number of rows in grid
            n_cols (int): Number of columns in grid
    Returns:
            bool: True if number is valid for location in current grid
    """
    if number in grid[row_index]:  # If the number is already in the row, it is not valid
        return False
    column = [grid[n][column_index] for n in range(0, len(grid))]  # Create a list of the numbers in the column
    if number in column:  # If the number is already in the column, it is not valid
        return False

    # Check sub grid
    # Find the coordinates of the top left corner of the sub grid
    grid_x = column_index // n_cols
    grid_y = row_index // n_rows

    # Check each cell in the sub grid to see if the number is already present
    for row_number in range(grid_y * n_rows, grid_y * n_rows + n_rows):
        for column_number in range(grid_x * n_cols, grid_x * n_cols + n_cols):
            if grid[row_number][column_number] == number and (
                row_number,
                column_number,
            ) != (row_index, column_index):
                return False

    return True  # If the number is not in the row, column or box, it is valid


def priority_length(
    priority_array: tuple[list[list[list[int] | int]]],
) -> int:
    """
    Key function to be used when sorting the priority array for the number of potential values for a location.

    Args:
        priority_array (tuple[list[list[list[int] | int]]]): Unsorted priority array

    Returns:
        int: Length of priority_array[2] to be used when sorting priority_array
    """
    return len(priority_array[2])


def create_priority(
    grid: list[list[int]],
    n_rows: int,
    n_cols: int,
    valid_array_old: list[list[list[int] | int]],
) -> tuple[list[list[list[int] | int]], list[list[list[int] | int]]]:
    """Creates an nested list of containing the number of valid values for each location
    and another nested list containing empty values replaced with an array of all valid values.

    Args:
        grid (list[list[int]]): Grid in its current state
        n_rows (int): Number of rows in grid
        n_cols (int): Number of columns in grid

    Returns:
        priority_array (tuple[list[list[list[int] | int]]): Nested list containing the coordinates of each point
        in the Sudoku grid and potential values - sorted against the number of potential values
        valid_array (list[list[list[int] | int]]]): A nested list containing the grid with zeroes replaced
        with all valid values for that location in the Sudoku grid
    """
    priority_array = []
    valid_array = copy.deepcopy(valid_array_old)
    n = n_rows * n_cols
    for row in range(0, n):
        for column in range(0, n):
            if grid[row][column] == 0:  # Check if cell is unfilled
                valid_array[row][column] = []
                priority_array.append([row, column, []])  # Append empty list containing the location of an unfilled value
                for value in valid_array_old[row][column]:
                    if valid(grid, row, column, value, n_rows, n_cols):  # Test if the value entered could be part of a valid solution
                        valid_array[row][column].append(value)  # Append valid value to corresponding list in valid_array
                        priority_array[-1][2].append(value)  # Append value to end of corresponding list in priority_array
    priority_array.sort(key=priority_length)  # Sort priority array in ascending order based on number of potential values for each grid location
    if (
        priority_array and type(priority_array[0][2]) == int
    ):  # Check if priority_array is empty and if only one potential value for location with least values
        priority_array[0][2] = [priority_array[0][2]]  # Create a list of length 1 containing the single valid value
    return priority_array, valid_array


def recursive_solve(
    grid: list[list[int]],
    n_rows: int,
    n_cols: int,
    priority_array: list[list[list[int]]],
) -> list[list[int]]:
    """A recursive Sudoku solver to enter values to grid and check if the current grid state is solved

    Args:
        grid (list[list[int]]): Unsolved grid
        n_rows (int): Number of rows in grid
        n_cols (int): Number of columns in grid
        priority_array (list[list[list[int]]]): Nested list containing the coordinates for unfilled grid locations and potential values for that grid location - sorted in ascending order based on the number of potential values

    Returns:
        grid (list[list[int]]): Sudoku grid after either a partial or full solution or the original grid if no values attempted were valid
    """
    if priority_array:  # Check if priority_array is empty
        row = priority_array[0][0]
        column = priority_array[0][1]
        for value in priority_array[0][2]:  # Try value from priority_array to corresponding grid location
            if valid(grid, row, column, value, n_rows, n_cols):  # Check if value is valid for the current state of the grid
                grid[row][column] = value  # Place value into grid
                recursive_solve(grid, n_rows, n_cols, priority_array[1:])  # Continue to next recursion level
                if check_solution(grid, n_rows, n_cols):  # Return grid if solution is complete
                    return grid
        grid[row][column] = 0  # Reset grid to original state if all values have been attempted and none are valid
    return grid  # Return the grid if it is solved


def wavefront_solve(
    grid: list[list[int]],
    n_rows: int,
    n_cols: int,
    valid_array: list[list[list[int] | int]],
    priority_array: list[list[int]],
) -> tuple[(list[list[int]] | bool), (list[list[int]] | bool)]:
    """
    A recursive Sudoku solver employing the 'wavefront' method. Checks if a valid solution has been reached if all grid locations are filled.

    Args:
        grid (list[list[int]]): Current state of grid
        n_rows (int): Number of rows in grid
        n_cols (int): Number of columns in grid
        valid_array (list[list[list[int] | int]]): Array containing the current state of the grid - zeroes are replaced with potential values for that grid location
        priority_array (list[list[list[int]]]): Nested list containing the coordinates for unfilled grid locations and potential values for that grid location - sorted in ascending order based on the number of potential values

    Returns:
        tuple(list[list[int]] | bool, list[list[int]] | bool): (Current grid state | False when invalid value is attempted, Updated priority_array | False when invalid value is attempted)
    """
    grid_update = copy.deepcopy(grid)  # Create a copy of the input grid
    priority_array_update = copy.deepcopy(priority_array)  # Create a copy of priority_array
    valid_array_update = copy.deepcopy(valid_array)  # Create a copy of valid_array
    grid_update_2 = None
    while priority_array_update:  # Repeat until priority_array_update is empty
        if len(priority_array_update[0][2]) == 1:  # Check if only one potential value for corresponding location in grid
            # Check if single value is valid for corresponding grid location
            if valid(grid_update, priority_array_update[0][0], priority_array_update[0][1], priority_array_update[0][2][0], n_rows, n_cols):
                grid_update[priority_array_update[0][0]][priority_array_update[0][1]] = priority_array_update[0][2][0]  # Update value in grid
                priority_array_update = priority_array_update[1:]  # Remove first element in priority_array_update
            else:
                return False, False
        else:
            # Randomly choose a potential number from those in priority_array for the corresponding location
            test_num = random.choice(priority_array_update[0][2])
            # Check if test_num is valid
            if valid(grid_update, priority_array_update[0][0], priority_array_update[0][1], test_num, n_rows, n_cols):
                grid_update_2 = copy.deepcopy(grid_update)  # Create a copy of the updated grid
                # Update location in grid with randomly chosen number from priority_array
                grid_update_2[priority_array_update[0][0]][priority_array_update[0][1]] = test_num
                # Update priority_array after inserting a random value
                (priority_array_update_2, valid_array_update_2) = create_priority(grid_update_2, n_rows, n_cols, valid_array_update)
                if check_solution(grid_update_2, n_rows, n_cols):  # Check if current state of grid is valid
                    return grid_update_2, False
                if not priority_array_update_2[0][2]:
                    # If value from next recursion layer is not valid, remove from current recursion level
                    priority_array_update[0][2].remove(test_num)
                    continue
                # Continue to next recursion level
                grid_update_2, priority_array_update_2 = wavefront_solve(grid_update_2, n_rows, n_cols, valid_array_update_2, priority_array_update_2)
                if not grid_update_2:  # Check if number was invalid from previous recursion level
                    priority_array_update[0][2].remove(test_num)  # Remove invalid value from current recursion level
            else:
                priority_array_update[0][2].remove(test_num)  # Remove invalid value from current recursion level
            if grid_update_2 is not None and grid_update_2:  # If grid_update_2 was created for this recursion level and is not False
                grid_check = grid_update_2  # Set grid_check to grid_update_2 from current recursion level
            elif grid_update:  # If grid_update was created for this recursion level and is not False
                grid_check = grid_update  # Set grid_check to grid_update from current recursion level
            if check_solution(grid_check, n_rows, n_cols):  # Check if solution is valid
                return grid_check, False  # Return solved grid
    # Return updated grid and priority array to previous recursion level
    return grid_update, priority_array


def solve(grid: list[list[int]], n_rows: int, n_cols: int, solver: str) -> list[list[int]]:
    """
    Function to execute setup code common for both the 'recursive' and 'wavefront' solvers and execute the chosen solver.

    Args:
        grid (list[list[int]]): Unsolved grid
        n_rows (int): Number of rows in grid
        n_cols (int): Number of columns in grid
        solver (str): Chosen solver

    Returns:
        result (list[list[int]]): Solved grid
    """
    possible_values = list(range(1, n_rows * n_cols + 1))
    valid_array_init = [
        [
            possible_values if grid[row][column] == 0 else grid[row][column] for column, _ in enumerate(grid)
        ]  # Creates a list of possible values for one grid location containing a '0' - otherwise the value at that grid location is used instead
        for row, _ in enumerate(grid)
    ]  # Creates a nested list containing all possible values for unfilled grid locations or a single value if that grid location was already filled
    priority_array, valid_array = create_priority(
        grid, n_rows, n_cols, valid_array_init
    )  # Creates initial priority_array based on initial valid_array
    if solver == "recursive":  # Execute the 'recursive' solver if selected
        result = recursive_solve(grid, n_rows, n_cols, priority_array)
    elif solver == "wavefront":  # Execute the 'wavefront' solver if selected
        result, empty_priority = wavefront_solve(grid, n_rows, n_cols, valid_array, priority_array)
    return result  # Return solved grid
