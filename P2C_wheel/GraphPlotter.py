# @Author: Joey Teng <Toujour>
# @Date:   19-Nov-2017
# @Email:  joey.teng.dev@gmail.com
# @Filename: GraphPlotter.py
# @Last modified by:   Toujour
# @Last modified time: 19-Nov-2017


import json
from sys import argv

import numpy
from PIL import Image, ImageDraw

DefaultColor = (0, 0, 0, 0)
BackgroundColor = (255, 255, 255, 100)


def Plot(draw, graph):
    for y in range(len(graph)):
        for x in range(len(graph[y])):
            # draw.point((x, y), fill=tuple(graph[y][x]))
            draw.ellipse((x, y, x + 5, y + 5),
                         fill=tuple(graph[y][x]), outline=tuple(graph[y][x]))

    return draw


def main(path, data, color=DefaultColor):
    # data: list: [tuple:(x, y)]

    x_max = 0
    y_max = 0
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = int(data[i][j])

    for x, y in data:
        x_max = max(x_max, x)
        y_max = max(y_max, y)

    graph = numpy.ones(
        (x_max + 1, y_max + 1, len(BackgroundColor)), dtype='int64') * BackgroundColor
    for x, y in data:
        graph[x][y] = DefaultColor

    image = Image.new('RGBA', (x_max + 1, y_max + 1), BackgroundColor)
    draw = ImageDraw.Draw(image)
    Plot(draw, graph)
    image.save(path, 'GIF', transparency=0)

    return image


def ReadJSON(path):
    return json.load(open(path, 'r'))


if __name__ == '__main__':
    main(argv[2], ReadJSON(argv[1]))
