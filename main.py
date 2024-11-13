import pygame
import aiLib as ai
import numpy as np
from keras.datasets import mnist # type: ignore

(train_X, train_y), (test_X, test_y) = mnist.load_data()
# print(train_X.shape, train_y.shape)
train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1]*train_X.shape[2]))

# print(train_X.shape, train_y.shape)
num_model = ai.Model()
print(num_model.train(train_X, train_y))

# Example file showing a basic pygame "game loop"
# pygame setup
# pygame.init()
# screen = pygame.display.set_mode((1280, 720))
# clock = pygame.time.Clock()
# running = True

# while running:
#     # poll for events
#     # pygame.QUIT event means the user clicked X to close your window
#     for event in pygame.event.get():
#         print(event)
#         if event.type == pygame.QUIT:
#             running = False

#     # fill the screen with a color to wipe away anything from last frame
#     screen.fill("red")

#     # RENDER YOUR GAME HERE

#     # flip() the display to put your work on screen
#     pygame.display.flip()

#     clock.tick(60)  # limits FPS to 60

# pygame.quit()


def main():
    """adfasdfa

    Args:
        a (integer): variable that does stuff
        b (heyyy): yayyy
    """

    

if __name__ == '__main__':
    main()