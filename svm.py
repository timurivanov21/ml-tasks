import pygame
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np


def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def draw_line(xx, yy, b):
    pygame.draw.line(screen, (0, 255, 255), (xx[0], yy[0]), (xx[len(xx) - 1], yy[len(yy) - 1]), 2)
    pygame.draw.line(screen, (0, 255, 255), (xx[0] - b, yy[0] - b), (xx[len(xx) - 1] - b, yy[len(yy) - 1] - b),
                     2)
    pygame.draw.line(screen, (0, 255, 255), (xx[0] + b, yy[0] + b), (xx[len(xx) - 1] + b, yy[len(yy) - 1] + b),
                     2)


def show_plot(points, cluster_number, xx, yy, b):
    graph = plt.subplot(111)

    graph.axis([0.0, 600.0, -300.0, 0.0])
    for i in range(0, len(points)):
        if cluster_number[i] == 0:
            graph.scatter(points[i][0], -points[i][1], c='b')
        else:
            graph.scatter(points[i][0], -points[i][1], c='y')
    graph.plot(xx, -yy, c='r')
    graph.plot(xx + b, -(yy + b), c='r')
    graph.plot(xx - b, -(yy - b), c='r')


if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((600, 400))
    screen.fill((255, 255, 255))
    clock = pygame.time.Clock()
    FPS = 60

    points = []
    cluster_number = []
    play = True

    algorithm = SVC(C=1.0, kernel='linear')
    r_was_pressed = False

    while play:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                play = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if r_was_pressed:
                    x, y = event.pos
                    new_point = np.array([[x, y]])
                    prediction = algorithm.predict(new_point)
                    color = (255, 255, 0) if prediction == 0 else (255, 0, 0)
                    pygame.draw.circle(screen, color, (x, y), 8)
                    pygame.display.flip()
                else:
                    if event.button == 1:
                        pygame.draw.circle(screen, (255, 255, 0), event.pos, 8)
                        points.append(event.pos)
                        cluster_number.append(0)
                    if event.button == 3:
                        pygame.draw.circle(screen, (255, 0, 0), event.pos, 8)
                        points.append(event.pos)
                        cluster_number.append(1)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    r_was_pressed = True
                    algorithm.fit(points, cluster_number)

                    w = algorithm.coef_[0]

                    a = -w[0] / w[1]
                    xx = np.linspace(100, 500, 600)
                    yy = (a * xx - algorithm.intercept_[0] / w[1])
                    b = 10000
                    point = []

                    for j in range(0, len(points)):
                        for i in range(0, len(yy)):
                            if (b > dist(points[j][0], points[j][1], xx[i], yy[i])):
                                b = dist(points[j][0], points[j][1], xx[i], yy[i])
                                point = [j, i]

                    draw_line(xx, yy, b)
                    show_plot(points, cluster_number, xx, yy, b)

        clock.tick(FPS)
        pygame.display.update()
