import copy
import random


def _check_section(section: list[int], n: int) -> bool:
    if len(set(section)) == len(section) and sum(section) == sum(
        [i for i in range(n + 1)]
    ):
        return True
    return False


def _get_squares(
    grid: list[list[int]], n_rows: int, n_cols: int
) -> list[list[int]]:
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


def _check_solution(grid: list[list[int]], n_rows: int, n_cols: int) -> bool:
    """
    This function is used to check whether a sudoku board has been correctly solved
    args: grid - representation of a suduko board as a nested list.
    returns: True (correct solution) or False (incorrect solution)
    """
    n = n_rows * n_cols
    for row in grid:
        if not _check_section(row, n):
            return False
    for i in range(n_rows):
        column = []
        for row in grid:
            column.append(row[i])
        if not _check_section(column, n):
            return False
    squares = _get_squares(grid, n_rows, n_cols)
    for square in squares:
        if not _check_section(square, n):
            return False
    return True


# we check if the number is valid in the row, column and box
def _valid(
    grid: list[list[int]],
    row_index: int,
    column_index: int,
    number: int,
    n_rows: int,
    n_cols: int,
) -> bool:
    """Tests if the number to be input is a valid input to the existing grid, i.e. does not clash with another value in the row, column, or box
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

    # Check sub grid
    # find the coordinates of the top left corner of the sub grid
    grid_x = column_index // n_cols
    grid_y = row_index // n_rows

    # check each cell in the sub grid to see if the number is already present
    for i in range(grid_y * n_rows, grid_y * n_rows + n_rows):
        for j in range(grid_x * n_cols, grid_x * n_cols + n_cols):
            if grid[i][j] == number and (i, j) != (row_index, column_index):
                return False

    return True  # if the number is not in the row, column or box, it is valid


def _priority_length(term: list) -> int:
    return len(term[2])


def _create_priority(
    grid: list[list[int]],
    n_rows: int,
    n_cols: int,
    valid_array_old: list[list[list[int] | int]],
) -> tuple[list[list[list[int] | int]], list[list[list[int] | int]]]:
    """Creates an array of the number of _valid values & a separate array with empty values replaced with an array of all _valid values
    Inputs:
        grid: current grid to be solved
        n_rows: number of boxes horizontally
        n_cols: number of boxes vertically
        Outputs:
        priority_array: array of coordinates in the format [row_no, col_no, possible_values_count]
            valid_array: array of original grid with all 0-values replaced with viable values
    """
    priority_array = []
    valid_array = copy.deepcopy(valid_array_old)
    n = n_rows * n_cols
    for row in range(0, n):  # i is the row
        for column in range(0, n):  # j is the column
            if grid[row][column] == 0:  # if the cell is empty
                valid_array[row][column] = []
                priority_array.append([row, column, []])
                for value in valid_array_old[row][column]:
                    if _valid(
                        grid, row, column, value, n_rows, n_cols
                    ):  # test that the value entered could be part of a _valid solution
                        valid_array[row][column].append(value)
                        priority_array[-1][2].append(value)
    priority_array.sort(key=_priority_length)
    if priority_array and type(priority_array[0][2]) == int:
        priority_array[0][2] = [priority_array[0][2]]
    return priority_array, valid_array


# function to solve the sudoku board
def recursive_solve(
    grid: list[list[int]],
    n_rows: int,
    n_cols: int,
    priority_array: list[list[list[int]]],
) -> list[list[int]]:
    """A recursive function to both enter and test possible values in the grid
    Inputs:
    grid: initial grid to solve
    n_rows: number of boxes horizontally
    n_cols: number of boxes vertically"""
    # N is the maximum integer considered in this board
    if priority_array:
        row = priority_array[0][0]
        column = priority_array[0][1]
        for value in priority_array[0][
            2
        ]:  # k is the number we are trying to put in the cell
            if _valid(
                grid, row, column, value, n_rows, n_cols
            ):  # test that the value entered could be part of a _valid solution
                grid[row][column] = value  # we put k in the cell
                recursive_solve(
                    grid, n_rows, n_cols, priority_array[1:]
                )  # we call the function recursively
                if _check_solution(
                    grid, n_rows, n_cols
                ):  # if the grid is correct, we return it
                    return grid
        grid[row][
            column
        ] = 0  # if we have tried all the numbers and none of them work, we return the grid to its original state
    return grid
    # we return the grid if it is already solved


def wavefront_solve(
    grid: list[list[int]],
    n_rows: int,
    n_cols: int,
    valid_array: list[list[list[int] | int]],
    priority_array: list[list[int]],
):
    grid_update = copy.deepcopy(grid)
    priority_array_update = copy.deepcopy(priority_array)
    valid_array_update = copy.deepcopy(valid_array)
    while priority_array_update:
        if len(priority_array_update[0][2]) == 1:
            if _valid(
                grid_update,
                priority_array_update[0][0],
                priority_array_update[0][1],
                priority_array_update[0][2][0],
                n_rows,
                n_cols,
            ):
                grid_update[priority_array_update[0][0]][
                    priority_array_update[0][1]
                ] = priority_array_update[0][2][0]
                priority_array_update = priority_array_update[1:]
            else:
                return False, False
        else:
            test_num = random.choice(priority_array_update[0][2])
            if _valid(
                grid_update,
                priority_array_update[0][0],
                priority_array_update[0][1],
                test_num,
                n_rows,
                n_cols,
            ):
                grid_update_2 = copy.deepcopy(grid_update)
                grid_update_2[priority_array_update[0][0]][
                    priority_array_update[0][1]
                ] = test_num
                (
                    priority_array_update_2,
                    valid_array_update_2,
                ) = _create_priority(
                    grid_update_2, n_rows, n_cols, valid_array_update
                )
                if _check_solution(grid_update_2, n_rows, n_cols):
                    return grid_update_2, False
                if not priority_array_update_2[0][2]:
                    priority_array_update[0][2].remove(test_num)
                    continue
                grid_update_2, priority_array_update_2 = wavefront_solve(
                    grid_update_2,
                    n_rows,
                    n_cols,
                    valid_array_update_2,
                    priority_array_update_2,
                )
                if not grid_update_2:
                    priority_array_update[0][2].remove(test_num)
            else:
                priority_array_update[0][2].remove(test_num)
            if "grid_update_2" in locals() and grid_update_2:
                grid_check = grid_update_2
            elif "grid_update" in locals() and grid_update:
                grid_check = grid_update
            if _check_solution(grid_check, n_rows, n_cols):
                return grid_check, False
    return grid_update, priority_array


def solve(
    grid: list[list[int]], n_rows: int, n_cols: int, solver: str
) -> list[list[int]]:
    """
    Solve function for Sudoku coursework.
    Comment out one of the lines below to either use the random or recursive solver
    """
    valid_array_init = []
    possible_values = list(range(1, n_rows * n_cols + 1))
    for row in range(0, len(grid)):  # i is the row
        valid_array_init.append([])
        for column in range(0, len(grid)):  # j is the column
            if grid[row][column] == 0:  # if the cell is empty
                # append list of all possible values
                valid_array_init[row].append(possible_values)
            else:
                valid_array_init[row].append(grid[row][column])
    priority_array, valid_array = _create_priority(
        grid, n_rows, n_cols, valid_array_init
    )
    if solver == "recursive":
        result = recursive_solve(grid, n_rows, n_cols, priority_array)
    elif solver == "wavefront":
        result, empty_priority = wavefront_solve(
            grid, n_rows, n_cols, valid_array, priority_array
        )
    return result
