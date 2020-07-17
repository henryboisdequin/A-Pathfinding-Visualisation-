import pygame
from queue import PriorityQueue
import math

# Setup
WIDTH = 800
win = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("A* Visualizer")

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)


class Node:
    """
    Keeps track of all nodes in grid and is responsible for coloring the grid.
    """

    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.total_rows = total_rows
        self.color = WHITE
        self.neighbours = []
        self.width = width

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == RED

    def is_open(self):
        return self.color == GREEN

    def is_wall(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == ORANGE

    def is_end(self):
        return self.color == TURQUOISE

    def reset(self):
        return self.color == WHITE

    def make_closed(self):
        self.color = RED

    def make_open(self):
        self.color = GREEN

    def make_wall(self):
        self.color = BLACK

    def make_start(self):
        self.color = ORANGE

    def make_end(self):
        self.color = TURQUOISE

    def make_path(self):
        self.color = PURPLE

    def draw(self, window):
        pygame.draw.rect(window, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbours(self, grid):
        """
        Updates neighbours around node.
        :return: None
        """
        self.neighbours = []
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_wall():  # down a row
            self.neighbours.append(grid[self.row + 1][self.col])

        if self.row > 0 and not grid[self.row - 1][self.col].is_wall():  # up a row
            self.neighbours.append(grid[self.row - 1][self.col])

        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_wall():  # right 1 node
            self.neighbours.append(grid[self.row][self.col + 1])

        if self.col > 0 and not grid[self.row][self.col - 1].is_wall():  # left 1 node
            self.neighbours.append(grid[self.row][self.col - 1])

    def __lt__(self, other):
        """
        Compare two nodes together.
        :param other: node object
        :return: bool
        """
        return False


def heuristic(p1, p2):
    """
    Heuristic for a* algorithm.
    :param p1: tuple
    :param p2: tuple
    :return: int
    """
    x1, y1 = p1
    x2, y2 = p2
    return math.fabs(x1 - x2) + math.fabs(y1 - y2)


def make_grid(rows, width):
    """
    Makes grid.
    :param rows: int
    :param width: int
    :return: list
    """
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            node = Node(i, j, gap, rows)
            grid[i].append(node)

    return grid


def draw_grid(window, rows, width):
    """
    Draws grid on the screen.
    :param window: pygame surface
    :param rows: int
    :param width: int
    :return:
    """
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(window, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(window, GREY, (j * gap, 0), (j * gap, width))


def draw(window, grid, rows, width):
    """
    Draws everything onto the screen.
    :param window: pygame surface
    :param grid: list
    :param rows: int
    :param width: int
    :return:
    """
    window.fill(WHITE)  # fills screen with white

    for row in grid:  # draws node, row
        for node in row:
            node.draw(window)

    draw_grid(window, rows, width)  # draws grid

    pygame.display.update()


def get_clicked_pos(pos, rows, width):
    """
    Gets node clicked on.
    :param pos: tuple
    :param rows: int
    :param width: int
    :return: int, int
    """
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap

    return row, col


def recontruct_path(came_from, current, draw_):
    while current in came_from:
        current = came_from[current]
        current.make_path()

        draw_()


def a_star(draw_, grid, start, end):
    """
    Runs the A* algorithm.
    :param end: int
    :param start: int
    :param grid: list
    :param draw_: function
    :return: bool
    """
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    # Calculates g score
    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0
    # Calculates f score
    f_score = {node: float("inf") for row in grid for node in row}
    f_score[start] = heuristic(start.get_pos(), end.get_pos())

    open_set_hash = {start}

    while not open_set.empty():
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            recontruct_path(came_from, end, draw_)
            end.make_end()
            return True

        for neighbour in current.neighbours:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbour]:
                came_from[neighbour] = current
                g_score[neighbour] = temp_g_score
                f_score[neighbour] = temp_g_score + heuristic(neighbour.get_pos(), end.get_pos())

                if neighbour not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbour], count, neighbour))
                    open_set_hash.add(neighbour)
                    neighbour.make_open()

        draw_()

        if current != start:
            current.make_closed()

    return False


def main(window, width):
    """
    Mainloop of the game.
    :param window: pyagme surface
    :param width: int
    :return: None
    """
    ROWS = 20
    grid = make_grid(ROWS, width)

    start = None
    end = None
    running = True
    started = False

    while running:
        draw(window, grid, ROWS, width)

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            if started:
                continue

            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, WIDTH)
                node = grid[row][col]
                if not start and node != end:
                    start = node
                    start.make_start()

                elif not end and node != start:
                    end = node
                    end.make_end()

                elif node != end and node != start:
                    node.make_wall()

            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, WIDTH)
                node = grid[row][col]
                node.reset()

                if node == start:
                    start = None
                elif node == end:
                    end = None

            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE and not started:
                    for row in grid:
                        for node in row:
                            node.update_neighbours(grid)

                    a_star(lambda: draw(window, grid, ROWS, width), grid, start, end)

                if ev.key == pygame.K_r:
                    start = None
                    end = None
                    grid = make_grid(ROWS, width)

    pygame.quit()


if __name__ == '__main__':
    main(win, WIDTH)
