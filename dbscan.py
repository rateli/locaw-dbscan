#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# This script contains the project work for Location Awareness course. It is an implementation of
# DBSCAN algorithm to perform spatial clustering. The algorithm uses a k-dimensional for approximating
# the nearest neighbors. Input data is provided in a .csv file and the results of the clustering are
# written into an another .csv file.


from timeit import default_timer as timer
import math
import argparse
import os
from collections import namedtuple
from operator import itemgetter
from heapq import heappop, heappush


Args = namedtuple('Args', [
    'verbose', 'input', 'output', 'latitude', 'longitude', 'epsilon', 'min_pts', 'knn_limit'
])
"""A named tuple containing arguments provided to the program."""


Point = namedtuple('Point', ['lat', 'lon'])
"""A named tuple representing a point in the map."""


TreeNode = namedtuple('TreeNode', ['point', 'axis', 'left', 'right'])
"""A named tuple representing a node in the k-d tree."""


HeapNode = namedtuple('HeapNode', ['distance', 'point'])
"""A named tuple representing a heap node for the knn search."""


clusters = []
"""Contains the clusters as arrays of points."""


clustered = {}
"""Contains points that have already been assigned to a cluster."""


visited = {}
"""Contains points that have already been visited."""


noise = {}
"""Contains points that have been flagged as noise."""


def create_kdtree(points, depth=0):
    """ Create a k-dimensional tree from given points.
            Args:
                    points ([]): Array containing all the points in data
                    depth (int): Tree depth
    """
    if len(points) == 0:
        return None
    k = len(points[0])
    axis = depth % k
    points.sort(key=itemgetter(axis))
    median = len(points) // 2
    return TreeNode(
        point=points[median],
        axis=axis,
        left=create_kdtree(points[:median], depth + 1),
        right=create_kdtree(points[median + 1:], depth + 1)
    )


def dbscan(dataset, eps, min_pts, knn_limit):
    """ Perform DBSCAN algorithm to find clusters in thee given dataset.
            Args:
                    dataset ([]): A dataset that contains the points
                    eps (int): Maximum range of neighbors from core point
                    min_pts (int): Minimum number of points it takes to make a cluster
                    knn_limit (int):  Number of approximate neighbors returned from knn (should be bigger than MIN_PTS)
    """
    cluster = 0
    for point in dataset:
        if point in visited:
            continue
        visited[point] = True
        neighbors = region_query(point, eps, knn_limit)
        if len(neighbors) < min_pts:
            noise[point] = True
        else:
            clusters.append([])
            expand_cluster(point, neighbors, cluster, eps, min_pts, knn_limit)
            cluster += 1


def region_query(point, eps, knn_limit):
    """ Find the approximate neighbors inside eps range from point.
            Args:
                    point (Point): Target point
                    eps (int): Max. distance from target point
                    knn_limit (int): Number of approximate neighbors returned from knn
            Returns:
                    []: Points in the eps-neighborhood
    """
    neighbors = []
    in_range = []
    heappush(neighbors, HeapNode(distance=float('-inf'), point=None)) # use negative distances because heapq only implements min heap
    neighbors = knn_search(tree, point, knn_limit, neighbors)
    for n in neighbors:
        if n.point is not None and haversine_distance(point, n.point) < eps:
            in_range.append(n.point)
    return in_range


def haversine_distance(a, b):
    """ Use the haversine formula to calculate the distance between two points in meters.
            Args:
                    a (Point): Point A
                    b (Point): Point B
            Returns:
                    int
    """
    R = 6371 * 1000
    dLat = math.radians(b.lat - a.lat)
    dLon = math.radians(b.lon - a.lon)
    lat_a = math.radians(a.lat)
    lat_b = math.radians(b.lat)

    a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.sin(dLon / 2) * \
        math.sin(dLon / 2) * math.cos(lat_a) * math.cos(lat_b)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return d


def squared_distance(a, b):
    """ Calculate the squared euclidean distance between two points.
            Args:
                    a (Point): Point A
                    b (Point): Point B
            Returns:
                    int
    """
    return (a.lat - b.lat)**2 + (a.lon - b.lon)**2


def expand_cluster(point, neighbors, cluster, eps, min_pts, knn_limit):
    """ Expand the current cluster by merging new neighbors into it.
            Args:
                    point (Point): Core point of the neighbors
                    neighbors ([]): List of neighbors in point's eps-neighborhood
                    cluster ([]): The cluster being processed
                    eps (int): Maximum range of neighbors from core point
                    min_pts (int): Minimum number of points it takes to make a cluster
                    knn_limit (int): Number of approximate neighbors returned from knn (should be bigger than MIN_PTS)

    """
    clustered[point] = True
    clusters[cluster].append(point)

    for n in neighbors:
        if n not in visited:
            visited[n] = True
            new_neighbors = region_query(n, eps, knn_limit)
            if len(new_neighbors) > min_pts:
                neighbors += new_neighbors
        if n not in clustered:
            clustered[n] = True
            clusters[cluster].append(n)


def knn_search(n, target, k, queue):
    """ Perform a k nearest neighbors -search for given point using a k-d tree.
            Args:
                    n (TreeNode): Current node in the tree
                    target (Point): The target point of the query
                    k (int): Number of nearest neighbors searched for
                    queue ([]): A priority queue that of current k nearest points and their distances.
                                    The point with greatest distance is on top of the heap.
            Returns:
                    []: A priority queue containing the k nearest points and their distances
                            as (distance, point) tuples
    """
    if n is None:
        return queue
    sq_dist = squared_distance(n.point, target) * (-1) # negate the distance for heap
    if sq_dist >= queue[0].distance:
        heappush(queue, HeapNode(distance=sq_dist, point=n.point))
        if(len(queue) >= k):
            heappop(queue)
    diff_on_axis = (target.lat - n.point.lat) if n.axis == 0 else (target.lon - n.point.lon)
    if diff_on_axis <= 0:
        first, second = n.left, n.right
    else:
        first, second = n.right, n.left
    queue = knn_search(first, target, k, queue)
    if -(diff_on_axis**2) > queue[0].distance: # also negate axis diff before comparing to current max distance
        queue = knn_search(second, target, k, queue)
    return queue


def write_output(outfile, clusters):
    """ Write clustered points into a result csv.
            Args:
                    outfile (str): Filename for the resulting .csv file
                    clusters ([]): List containing the clustered points
    """
    with open(outfile, 'w') as wf:
        wf.write('Latitude,Longitude\n')
        for i in range(len(clusters)):
            for p in clusters[i]:
                wf.write('%s,%s\n' % (p.lat, p.lon))


def parse_csv(infile, lat_col, lon_col):
    """ Parse CSV file containing latitude and longitude values into points.
            Args:
                    infile (str): Path to .csv file containing the latitude and longitude values
                    lat_col (int): Index of column containing latitude values
                    lon_col (int): Index of column containing longitude values
            Returns:
                    []: Array containing the latitudes and longitudes as Point named tuples
    """
    if not os.path.isfile(infile):
        print('Specified file "' + infile + '" does not exist.')
        exit(1)
    data = []
    with open(infile, 'r') as f:
        f.readline()  # skip header row
        for line in f:
            line = line.strip().split(',')
            try:
                data.append(Point(
                    lat=float(line[lat_col]),
                    lon=float(line[lon_col])
                ))
            except Exception as e:
                print(e)
                exit(1)
    return data


def initialize():
    """ Parse input arguments and initialize the variables used by the algorithm.
            Returns:
                    Args: namedtuple containing the arguments.
    """
    parser = argparse.ArgumentParser(
        description='Performs spatial clustering for given data sheet using DBSCAN algorithm.')
    parser.add_argument('-v', '--verbose', action="store_true",
                        help='increase output verbosity')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='set .csv input file location')
    parser.add_argument('-o', '--output', type=str,
                        help='set .csv output file location (defaults to output.csv)')
    parser.add_argument('-y', '--latitude', type=int,
                        help='set (0-based) index of latitude column in the .csv file (defaults to 0)')
    parser.add_argument('-x', '--longitude', type=int,
                        help='set (0-based) index of longitude column in the .csv file (defaults to 1)')
    parser.add_argument('-e', '--epsilon', type=int, required=True,
                        help='set cluster range in meters')
    parser.add_argument('-m', '--min-pts', type=int, required=True,
                        help='set minimum number of points considered as a cluster')
    parser.add_argument('-k', '--knn-limit', type=int, required=True,
                        help='set approximate number of neighbors returned from knn-search (should be higher than MIN_PTS)')
    args = parser.parse_args()
    return Args(
        verbose=args.verbose,
        input=args.input,
        output=args.output if args.output is not None else 'output.csv',
        latitude=args.latitude if args.latitude is not None else 0,
        longitude=args.longitude if args.longitude is not None else 0,
        epsilon=args.epsilon,
        min_pts=args.min_pts,
        knn_limit=args.knn_limit,
    )


if __name__ == "__main__":
    start = timer()
    args = initialize()
    data = parse_csv(args.input, args.latitude, args.longitude)

    tree = create_kdtree(data)
    dbscan(data, args.epsilon, args.min_pts, args.knn_limit)
    write_output(args.output, clusters)

    if(args.verbose):
        print('Clusters:', len(clusters))
        print('Points clustered:', len(clustered))
        print('Noise:', len(noise))
    print("Output written to file", args.output)
    end = timer()
    print(timer() - start, 'seconds')
