# Snake, and an AI that learns to play it, written from zero.
# Run it with:  python snake.py
# It needs pygame for the window:  pip install pygame

import random
import time

import pygame

BOARD_SIZE = 10  # the board is 10 by 10 squares
SQUARE = 60  # each square is 60 pixels wide on screen

# How one move changes (row, column) for each direction.
DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
# Which way the snake faces after turning left or right.
LEFT_OF = {"up": "left", "left": "down", "down": "right", "right": "up"}
RIGHT_OF = {"up": "right", "right": "down", "down": "left", "left": "up"}

# Colours as (red, green, blue), each 0 to 255.
BACKGROUND = (13, 15, 22)
SNAKE_COLOUR = (74, 222, 128)
HEAD_COLOUR = (21, 128, 61)
APPLE_COLOUR = (244, 63, 94)


def new_game():
    # A square on the board is (row, column). Row 0 is the top, column 0 the left.
    middle = BOARD_SIZE // 2
    game = {
        # The snake is a list of squares, head first.
        "snake": [(middle, middle), (middle, middle - 1), (middle, middle - 2)],
        "direction": "right",
        "apple": None,
        "score": 0,
        "alive": True,
        "hunger": 0,  # moves since the last apple
    }
    place_apple(game)
    return game


def place_apple(game):
    # The apple goes on a random square that the snake isn't on.
    empty_squares = []
    for row in range(BOARD_SIZE):
        for column in range(BOARD_SIZE):
            if (row, column) not in game["snake"]:
                empty_squares.append((row, column))
    game["apple"] = random.choice(empty_squares)


def print_board(game):
    for row in range(BOARD_SIZE):
        line = ""
        for column in range(BOARD_SIZE):
            square = (row, column)
            if square == game["snake"][0]:
                line += "H "  # the head
            elif square in game["snake"]:
                line += "o "  # the body
            elif square == game["apple"]:
                line += "@ "  # the apple
            else:
                line += ". "  # empty
        print(line)


def next_square(game, direction):
    head_row, head_column = game["snake"][0]
    row_change, column_change = DIRECTIONS[direction]
    return (head_row + row_change, head_column + column_change)


def is_deadly(game, square):
    row, column = square
    hit_wall = row < 0 or row >= BOARD_SIZE or column < 0 or column >= BOARD_SIZE
    # The last piece of tail moves out of the way this turn, so it doesn't count.
    hit_body = square in game["snake"][:-1]
    return hit_wall or hit_body


# step() is the whole game: one move, no drawing, no keyboard.
# It returns the reward: +1 for an apple, -1 for dying, 0 for anything else.
def step(game, direction):
    # The snake can't turn back into its own neck: ignore that and keep going.
    if direction != OPPOSITE[game["direction"]]:
        game["direction"] = direction
    new_head = next_square(game, game["direction"])
    if is_deadly(game, new_head):
        game["alive"] = False
        return -1

    # Moving = a new head in front, and the last piece of tail goes away.
    game["snake"].insert(0, new_head)
    if new_head == game["apple"]:
        game["score"] += 1
        game["hunger"] = 0
        place_apple(game)  # and the tail stays: the snake grows by one
        return 1
    game["snake"].pop()

    # A snake that never eats can circle forever, and the game never ends.
    # So it starves: 100 moves without an apple (plus 2 per piece of snake,
    # because a long snake needs a long way round). Starving counts as dying.
    game["hunger"] += 1
    if game["hunger"] > 100 + 2 * len(game["snake"]):
        game["alive"] = False
        return -1
    return 0


def draw_square(screen, colour, square):
    row, column = square
    # Leave a 2 pixel gap so you can see the separate pieces of the snake.
    pygame.draw.rect(screen, colour, (column * SQUARE + 2, row * SQUARE + 2, SQUARE - 4, SQUARE - 4))


def draw(screen, game):
    screen.fill(BACKGROUND)
    draw_square(screen, APPLE_COLOUR, game["apple"])
    for square in game["snake"]:
        draw_square(screen, SNAKE_COLOUR, square)
    draw_square(screen, HEAD_COLOUR, game["snake"][0])
    pygame.display.flip()  # show what we just drew


def play():
    pygame.init()
    screen = pygame.display.set_mode((BOARD_SIZE * SQUARE, BOARD_SIZE * SQUARE))
    clock = pygame.time.Clock()
    keys = {pygame.K_UP: "up", pygame.K_DOWN: "down", pygame.K_LEFT: "left", pygame.K_RIGHT: "right"}
    game = new_game()
    wanted = game["direction"]
    # The game loop: check for input, move, draw, wait. Over and over.
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # the window's close button
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN and event.key in keys:
                wanted = keys[event.key]
        step(game, wanted)
        if not game["alive"]:
            pygame.display.set_caption(f"Game over! Score: {game['score']}")
            pygame.time.wait(1500)
            game = new_game()
            wanted = game["direction"]
        pygame.display.set_caption(f"Score: {game['score']}")
        draw(screen, game)
        clock.tick(8)  # 8 moves a second


def distance_to_apple(square, game):
    # Squares to walk if there were nothing in the way (no diagonals).
    return abs(square[0] - game["apple"][0]) + abs(square[1] - game["apple"][1])


# What the AI gets to see: six yes/no answers, from the snake's own point of view.
# It never sees the whole board.
def observe(game):
    facing = game["direction"]
    head = game["snake"][0]
    now = distance_to_apple(head, game)
    looks = []
    for direction in [facing, LEFT_OF[facing], RIGHT_OF[facing]]:  # straight, left, right
        square = next_square(game, direction)
        looks.append(is_deadly(game, square))  # would I die going this way?
        looks.append(distance_to_apple(square, game) < now)  # would I get closer to the apple?
    return tuple(looks)


# The AI doesn't press arrow keys: it picks one of three turns.
TURNS = ["straight", "left", "right"]


def turn(game, choice):
    facing = game["direction"]
    if choice == "left":
        return LEFT_OF[facing]
    if choice == "right":
        return RIGHT_OF[facing]
    return facing


def random_bot(game):
    return random.choice(["up", "down", "left", "right"])


def apple_chaser(game):
    # A rule written by hand: head straight for the apple, never look for danger.
    head_row, head_column = game["snake"][0]
    apple_row, apple_column = game["apple"]
    if apple_row < head_row:
        return "up"
    if apple_row > head_row:
        return "down"
    if apple_column < head_column:
        return "left"
    return "right"


def test_bot(name, bot, number_of_games=1000):
    # No window, no waiting: the game runs as fast as the computer can go.
    start = time.time()
    total_score = 0
    for game_number in range(number_of_games):
        game = new_game()
        while game["alive"]:
            step(game, bot(game))
        total_score += game["score"]
    seconds = time.time() - start
    print(f"{name}: average score {total_score / number_of_games:.2f} ({number_of_games} games in {seconds:.1f} s)")


# The AI's memory: for everything it can see, a score for each of the three turns
# ("how good is going straight / left / right from here?"). It starts empty.
table = {}

LEARNING_RATE = 0.1  # how far one experience moves a score
FUTURE = 0.9  # how much an apple later counts, compared to an apple now
EXPLORE = 0.1  # how often it tries a random turn, just to see what happens


def scores_for(seen):
    if seen not in table:
        table[seen] = [0.0, 0.0, 0.0]  # never seen this before: no opinion yet
    return table[seen]


def pick_turn(seen, explore):
    if random.random() < explore:
        return random.randrange(3)  # 0, 1 or 2: try something
    scores = scores_for(seen)
    return scores.index(max(scores))  # the turn with the best score so far


def train(number_of_games):
    total_score = 0
    for game_number in range(1, number_of_games + 1):
        game = new_game()
        seen = observe(game)
        while game["alive"]:
            choice = pick_turn(seen, EXPLORE)
            reward = step(game, turn(game, TURNS[choice]))
            new_seen = observe(game)
            # What that move turned out to be worth: the reward now, plus the best
            # it expects from where it ended up (nothing, if it died).
            if game["alive"]:
                worth = reward + FUTURE * max(scores_for(new_seen))
            else:
                worth = reward
            # Nudge the old score a little towards what actually happened.
            scores = scores_for(seen)
            scores[choice] += LEARNING_RATE * (worth - scores[choice])
            seen = new_seen
        total_score += game["score"]
        if game_number % 100 == 0:
            print(f"games {game_number - 99}-{game_number}: average score {total_score / 100:.2f}")
            total_score = 0
    print(f"the table now has {len(table)} rows (things it has seen)")


def ai_bot(game):
    return turn(game, TURNS[pick_turn(observe(game), explore=0)])


def watch(bot, title, number_of_games=3):
    pygame.init()
    screen = pygame.display.set_mode((BOARD_SIZE * SQUARE, BOARD_SIZE * SQUARE))
    clock = pygame.time.Clock()
    for game_number in range(number_of_games):
        game = new_game()
        while game["alive"]:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            step(game, bot(game))
            pygame.display.set_caption(f"{title}: score {game['score']}")
            draw(screen, game)
            clock.tick(15)
        pygame.time.wait(1000)
    pygame.quit()


# Pick what to run: put a # in front of the lines you don't want.
# play()
# test_bot("random bot", random_bot)
# test_bot("apple chaser", apple_chaser)
watch(ai_bot, "Before learning")  # an empty table: it always goes straight
train(100)
watch(ai_bot, "After 100 games")
train(900)
test_bot("trained AI", ai_bot)
watch(ai_bot, "After 1,000 games")
