import numpy as np

from metalpy.utils.rand import check_random_state


def extract_vec(pool, random_state):
    n_edges = len(pool)
    pool = np.sort(pool)
    vec = []

    xmin, xmax = pool[0], pool[-1]

    last_left, last_right = xmin, xmin
    for i in range(1, n_edges - 1):
        x = pool[i]

        if random_state.rand() > 0.5:
            vec.append(x - last_left)
            last_left = x
        else:
            vec.append(last_right - x)
            last_right = x

    vec.append(xmax - last_left)
    vec.append(last_right - xmax)

    return vec


def gen_random_convex_polygon(n_edges, size=None, random_state=None):
    """
    https://kingins.cn/2022/02/18/%E9%9A%8F%E6%9C%BA%E5%87%B8%E5%A4%9A%E8%BE%B9%E5%BD%A2%E7%94%9F%E6%88%90%E7%AE%97%E6%B3%95/
    Valtr P . Probability thatnrandom points are in convex position[J]. 1995, 13(1):637-643.

    :param size: 随机多边形的最大尺寸
    :param n_edges: 边数
    :param random_state: 随机状态
    :return: 随机多边形的各顶点
    """

    if size is None:
        size = [1, 1]
    size = np.asarray(size)

    # np.random.Generator
    random_state = check_random_state(random_state)

    # generate two lists of random X and Y coordinates
    x_pool = random_state.rand(n_edges)
    y_pool = random_state.rand(n_edges)

    # sort
    x_pool = np.sort(x_pool)
    y_pool = np.sort(y_pool)

    # isolate the extreme points
    xmin, xmax = x_pool[0], x_pool[-1]
    ymin, ymax = y_pool[0], y_pool[-1]

    # divide the interior points into two chains and extract the vector components
    x_vec = extract_vec(x_pool, random_state)
    y_vec = extract_vec(y_pool, random_state)

    # randomly pair up the X- and Y- components
    random_state.shuffle(y_vec)

    # combine the paired up components into vectors
    vectors = [np.asarray((x, y)) for x, y in zip(x_vec, y_vec)]
    vectors.sort(key=lambda a: np.arctan2(*a))

    # lay them end to end
    point = np.asarray((0, 0))
    minPolygonX = 0.0
    minPolygonY = 0.0
    points = []

    for vector in vectors:
        points.append(point)
        point = point + vector
        minPolygonX = min(minPolygonX, point[0])
        minPolygonY = min(minPolygonY, point[1])

    points = np.vstack(points)
    # translate the polygon to the original min and max coordinates
    translation = np.asarray((xmin, ymin)) - np.asarray((minPolygonX, minPolygonY))

    result = (points + translation) * size

    return result