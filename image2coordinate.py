# @Author: Joey Teng <Toujour>
# @Date:   18-Nov-2017
# @Email:  joey.teng.dev@gmail.com
# @Filename: image2coordinate.py
# @Last modified by:   Toujour
# @Last modified time: 18-Nov-2017

import queue
import sys
import warnings

import matplotlib.image
import numpy

BLACK = [0., 0., 0., 1]  # The color that need to be displayed


def load_image(filename):
    # Return an np.array
    return matplotlib.image.imread(filename)


def cal_angle(a, b, c):
    # a, b, c: dict ['position': (x, y)]
    # return
    a_p = numpy.array(a['position'])
    b_p = numpy.array(b['position'])
    c_p = numpy.array(c['position'])
    ab = b_p - a_p
    bc = c_p - b_p

    # sin(abc)
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            cross = numpy.cross(ab, bc)
            ab_norm = numpy.linalg.norm(ab)
            bc_norm = numpy.linalg.norm(bc)
            return (cross / ab_norm / bc_norm)
        except Warning:
            print('ab: {0}'.format(ab), flush=True)
            print('bc: {0}'.format(bc), flush=True)
            assert(False)


def is_black(pixel):
    result = (pixel == BLACK)

    for boolean in result:
        if not boolean:
            return False

    return True  # if the pixel is black. i.e. should this pixel be displayed


def at_edge(point, image):
    # point: tuple(x, y)
    # image: numpy.array(2D), pixels
    # return: Bool. At edge & should be

    if (point[0] < 0 or point[1] < 0 or point[0] >= image.shape[0] or point[1] >= image.shape[1]):
        return False
    if (is_black(image[point[0]][point[1]]) and (point[0] == 0 or point[1] == 0 or point[0] == image.shape[0] - 1 or point[1] == image.shape[1] - 1)):
        return True

    if (is_black(image[point[0]][point[1]])):
        for inc in [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]:
            new_point = ((point[0] + inc[0]), (point[1] + inc[1]))

            try:
                if (not is_black(image[new_point[0]][new_point[1]])):
                    return True
            except IndexError:
                assert(False)
                return True

    return False


def next_point(point, image, visited):
    # point: {'position': tuple(x, y)}
    # image: numpy.array(2D), pixels
    # visited: dict of tuple(s) (x, y), denote the points that has been visited.

    Queue = queue.Queue()
    Queue.put(point['position'])

    test_point = point['position']
    while (not Queue.empty()):
        point = Queue.get()
        for inc in [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]:
            new_point = ((point[0] + inc[0]), (point[1] + inc[1]))

            if (new_point not in visited):
                if (at_edge(new_point, image)):
                    # if (numpy.sqrt((test_point[0] - new_point[0])**2 + (test_point[1] - new_point[1])**2 > 2)):
                    #    print("Warning: distance > 2! Origin: {0}; point: {3} Current: {1}; inc: {2}".format(
                    #        test_point, new_point, inc, point), flush=True)
                    #    for x in (-1, 0, 1):
                    #        print("{0} {1} {2}".format(is_black(image[point[0] + x][point[1] - 1]), is_black(
                    #            image[point[0] + x][point[1]]), is_black(image[point[0] + x][point[1] + 1])))

                    visited[new_point] = True
                    return {'position': new_point}
                else:
                    if (point[0] < 0 or point[1] < 0 or point[0] >= image.shape[0] or point[1] >= image.shape[1]):
                        visited[new_point] = True
                        Queue.put(new_point)
            else:
                pass
        # print('INFO: Pop: {0}'.format(point), flush=True)

    return None


def find_angle(image):
    # Return a list of dict(s) with
    #     ['position': (x, y), 'angle':]

    # dict of tuple(s) (x, y), denote the points that has been visited.
    visited = {}
    # Find the first point
    point = None
    y = 0
    for row in image:
        x = 0
        for pixel in row:
            # if (is_black(pixel)):
            # if (is_black(pixel) and not at_edge((x, y), image)):
            #    print("Warning: pixel not at edge! ({0}, {1})".format(
            #        x, y), flush=True)
            #    for xx in (-1, 0, 1):
            #        print("{0} {1} {2}".format(is_black(image[x + xx][y - 1]), is_black(
            #            image[x + xx][y]), is_black(image[x + xx][y + 1])))
            if (at_edge((x, y), image) and (x, y) not in visited):  # if $pixel need to be drawn
                point = {'position': (x, y)}
                visited[point['position']] = True
                break
            x += 1
        y += 1
        if (point is not None):
            break

    print(point, flush=True)

    angle = []
    angle.append(point)

    point = next_point(point, image, visited)

    if (point is not None):
        angle[-1]['next'] = point
        angle.append(point)
        point = next_point(point, image, visited)

    while (point is not None):
        try:
            point['angle'] = cal_angle(angle[-2], angle[-1], point)
        except AssertionError:
            print('angle[-2]: {0}\nangle[-1]: {1}\npoint: {2}'.format(
                angle[-2], angle[-1], point), flush=True)
            assert(False)

        angle[-1]['next'] = point
        angle.append(point)
        point = next_point(point, image, visited)

        i = 1
        Traced = False
        while (point is None and i < len(angle)):
            Traced = True
            point = next_point(angle[-i], image, visited)
            i += 1
        if (Traced):
            print("INFO: Trace back {0} steps".format(i), flush=True)

        if len(angle) % 100 == 0:
            # print("INFO: {0}".format(angle[-1]), flush=True)
            print("INFO: {0} points has been identified".format(
                len(angle)), flush=True)
            # for i in angle:
            #    print(i['position'], flush=True)

    angle[-1]['next'] = angle[0]
    if (len(angle) > 2):
        angle[0]['angle'] = cal_angle(angle[-2], angle[-1], angle[0])
        angle[1]['angle'] = cal_angle(angle[-1], angle[0], angle[1])

    import matplotlib.pyplot as plt
    plt.figure(figsize=(256, 256), dpi=32)
    xx = []
    yy = []
    for x, y in visited.keys():
        xx.append(x)
        yy.append(y)
    plt.axis([0, image.shape[1], 0, image.shape[0]])
    plt.scatter(yy, xx, s=32)
    plt.savefig('visited.png')
    plt.cla()

    return angle


def line_to(start, end, image):
    if (start[0] == end[0]):
        if (start[1] > end[1]):
            start, end = end, start

        for i in range(start[1], end[1] + 1):
            image[start[0]][i] = BLACK
    else:
        if (start[0] < end[0]):
            start, end = end, start

        inc = (start[1] - end[1]) / (start[0] - end[0])

        for i in range(start[0] - end[0] + 1):
            y = int(start[1] - inc * i)
            if (y >= image.shape[1]):
                y = image.shape[1] - 1
            try:
                image[start[0] - i][y] = BLACK
            except IndexError:
                assert(False)


def fit_image(vertices, shape):
    image = numpy.ones(shape)
    for i in range(-1, len(vertices), 1):
        line_to(vertices[i], vertices[(i + 1) % len(vertices)], image)

    return image


def similarity(vertices, image):
    # vertices: list of tuple(s)(x, y) in order.

    image_fitted = fit_image(vertices, image.shape)

    import matplotlib.pyplot as plt
    xx = []
    yy = []
    for x in range(image_fitted.shape[0]):
        for y in range(image_fitted.shape[1]):
            if (is_black(image_fitted[x][y])):
                xx.append(x)
                yy.append(y)
    plt.axis([0, image_fitted.shape[1], 0, image_fitted.shape[0]])
    plt.scatter(yy, xx, s=32)
    plt.savefig('fitted.png')
    plt.cla()

    correct = 0
    incorrect = 0
    for x in range(image_fitted.shape[0]):
        for y in range(image_fitted.shape[1]):
            f = is_black(image_fitted[x][y])
            if (at_edge((x, y), image)):  # or f):
                i = is_black(image[x][y])
                if (f or i):
                    if (f == i):
                        correct += 1
                    else:
                        incorrect += 1

    # print(incorrect + correct, flush=True)
    return (correct / (incorrect + correct))


def processing(angle, image):
    # import matplotlib.pyplot as plt
    # img = numpy.ones(image.shape)
    # for i in range(len(angle)):
    #    img[angle[i]['position'][0]][angle[i]['position'][1]] = BLACK
    # plt.imshow(img)
    # plt.axis('off')
    # plt.savefig('image.png')
    # plt.cla()
    import matplotlib.pyplot as plt
    xx = []
    yy = []
    for a in angle:
        x, y = a['position']
        xx.append(x)
        yy.append(y)
    plt.axis([0, image.shape[1], 0, image.shape[0]])
    plt.scatter(yy, xx, s=32)
    plt.savefig('image.png')
    plt.cla()

    threshold = 0.4  # Assumption

    max_len = len(angle)
    min_len = 0

    vertices = []
    while (max_len > min_len):
        backup = [element for element in angle]

        test_len = (max_len + min_len) >> 1
        vertices.clear()

        i = 0
        j = 0
        while i < len(angle):
            if (((j * (test_len)) % (max_len)) >= test_len):
                angle[i - 1]['next'] = angle[(i + 1) % len(angle)]
                angle.pop(i)
                i -= 1
            else:
                vertices.append(angle[i]['position'])
            i += 1
            j += 1

        if (len(angle) == len(backup)):
            return vertices

        sim = similarity(vertices, image)
        if (sim >= threshold):
            max_len = len(vertices) - 1
        else:
            angle = backup
            if (min_len == len(vertices)):
                break
            min_len = len(vertices)

        print("INFO: One iteration completed.", flush=True)
        print("INFO: Max points: {0}; Min points: {1}; Similarity: {2}; size of vertices: {3}".format(
            max_len, min_len, sim, len(vertices)), flush=True)

    return vertices


if __name__ == '__main__':
    print("INFO: Task Start", flush=True)

    image = load_image(sys.argv[1])
    print("INFO: Image Loaded", flush=True)

    angle = find_angle(image)
    print("INFO: Points at edges identified", flush=True)

    vertices = processing(angle, image)

    print(vertices)
    import matplotlib.pyplot as plt
    xx = []
    yy = []
    for y, x in vertices:
        xx.append(x)
        yy.append(y)
    plt.axis([0, image.shape[1], 0, image.shape[0]])
    plt.scatter(xx, yy, s=32)
    plt.savefig('vertices.png')
    plt.cla()

    print("INFO: Task Complete", flush=True)
