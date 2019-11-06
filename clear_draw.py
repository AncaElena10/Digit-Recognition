import pygame
import primary_function as pf
import random
import timeit
from pathlib import Path


random.seed(0)
start = timeit.default_timer()


# initialize colors
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)

# set the pen properties
pen_color = (0, 0, 0)
lineWeight = 17

# set the image properties
width = 350
height = 350

# set image result position
x_image_result = (width * 1.4)
y_image_result = (height * 1 / 2.5)
x_image_text = (width * 1.1)
y_image_text = (height * 1 / 4)

# initialize the screen
screen = pygame.display.set_mode((width * 2, height))
screen.fill(white)
pygame.font.init()

# display text
text = pygame.image.load("./text.png")
screen.blit(text, (x_image_text, y_image_text))


def display_predicted_number(filename):
    extension = '.png'
    full_filename = str(filename) + extension
    predicted_number = pygame.image.load(full_filename)

    # display image result
    screen.blit(predicted_number, (x_image_result, y_image_result))


# save the image
def save_image(img_):
    object_ = pygame.Surface((width - 5, height - 5))
    object_.blit(img_, (0, 0), (0, 0, width - 5, height - 5))

    return object_


def check_if_out_exists():
    my_file = Path("./out.png")
    if my_file.is_file():
        predicted_number = pf.predict_my_data()
        print("My prediction is: " + str(predicted_number))
        if predicted_number == 0:
            display_predicted_number(0)
        if predicted_number == 1:
            display_predicted_number(1)
        if predicted_number == 2:
            display_predicted_number(2)
        if predicted_number == 3:
            display_predicted_number(3)
        if predicted_number == 4:
            display_predicted_number(4)
        if predicted_number == 5:
            display_predicted_number(5)
        if predicted_number == 6:
            display_predicted_number(6)
        if predicted_number == 7:
            display_predicted_number(7)
        if predicted_number == 8:
            display_predicted_number(8)
        if predicted_number == 9:
            display_predicted_number(9)
    else:
        print("Sorry, that file doesn't exist.")


# draw lines without interruptions
def draw_without_interruption(srf, color_, start_, end_, weight=1):
    dx = end_[0] - start_[0]
    dy = end_[1] - start_[1]
    distance = max(abs(dx), abs(dy))

    for i in range(distance):
        x = int(start_[0] + float(i) / distance * dx)
        y = int(start_[1] + float(i) / distance * dy)
        pygame.draw.circle(srf, color_, (x, y), weight)


# start to draw
def start_draw():
    pygame.draw.line(screen, black, [width, 0], [width, height], 8)


# mouse functions
def draw_functions():
    lets_draw = False
    previous_pos = (0, 0)

    while True:
        # all the events
        event_ = pygame.event.wait()
        start_draw()

        # on left click
        if event_.type == pygame.MOUSEBUTTONDOWN and event_.button != 3:
            lets_draw = True

        # start drawing
        if event_.type == pygame.MOUSEMOTION:
            if lets_draw:
                draw_without_interruption(screen, pen_color, event_.pos, previous_pos, lineWeight)
            previous_pos = event_.pos

        # empty screen on right click
        if event_.type == pygame.MOUSEBUTTONDOWN and event_.button == 3:
            # screen.fill(white)
            screen.fill(white, (0, 0, screen.get_width()/ 2, screen.get_height()))

        # save after left click is off
        if event_.type == pygame.MOUSEBUTTONUP and event_.button != 3:
            lets_draw = False
            file_name = "out.png"
            img = save_image(screen)
            screen.blit(img, (0, 0))
            pygame.display.flip()
            pygame.image.save(img, file_name)

            # when the file is generated, predict the number
            check_if_out_exists()

        # quit
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                quit()
        pygame.display.update()


if __name__ == "__main__":
    draw_functions()


stop = timeit.default_timer()
print("\n")
print("Time: " + str(stop - start))
