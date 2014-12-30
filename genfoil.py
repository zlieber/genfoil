#!/usr/bin/python

import argparse
import re
import sys
import numpy
import scipy.optimize

A1 = 0.2969
A2 = -0.1260
A3 = -0.3516
A4 = 0.2843
A5 = -0.1015

def calcy(x):
    """Calculate a single airfoil point.

    Uses NACA formula.

    Args:
      x: X coordinate

    Returns:
      The calculated point (x, y)"""

    x1 = x / C_PARAM;
    return (T_PARAM / 0.2 * C_PARAM) * (
        A1 * x1 ** 0.5 +
        A2 * x1 +
        A3 * x1 ** 2 +
        A4 * x1 ** 3 +
        A5 * x1 ** 4)

def printpt(name, x, y):
    print "%s: %f, %f" % (name, x, y)

def padcurve(x1, y1, x2, y2, r):
    # midpoint between two points
#    printpt("pt1", x1, y1);
#    printpt("pt2", x2, y2);
    xmid = (x1 + x2) / 2
    ymid = (y1 + y2) / 2
#    printpt("Mid", xmid, ymid)

    if abs(x2 - x1) < 0.00001:
        normA = 0
    else:
        a = (y2 - y1) / (x2 - x1)
        b = ymid - a*xmid

        # If the derivative is 0, normal is infinity, but the result
        # is much simpler to compute: x stays the same, just add r to y
        if abs(a) < 0.00001:
            return (xmid, ymid + r)

            # now we know: y = a*x + b
        normA = -1/a

    normB = ymid - normA*xmid

    # now we know the normal: y = normA*x + normB
    # Solving quadratic equation
    # r^2 = (x - xmid)^2 + (y - ymid)^2
    # Substituting y = normA*x + normB
    # r^2 = (x - xmid)^2 + (normA*x + normB - ymid)^2
    # r^2 = x^2 - 2*xmid*x + xmid^2 + normA^2*x^2 + 2(normB-ymid)*normA*x + (normB - ymid)^2
    # (1 + normA^2)x^2 + (2*(normB-ymid)*normA - 2*xmid)x + (xmid^2 + (normB - ymid)^2 - r^2) = 0
    # qA*x^2 + qA*x + qC = 0
    qA = 1 + normA**2
    qB = 2*(normB-ymid)*normA - 2*xmid
    qC = xmid**2 + (normB - ymid)**2 - r**2
    D = qB**2 - 4*qA*qC
    if D < 0:
        raise "No solutions for quadratic equation"
    root1X = (-qB + D**0.5) / (2*qA)
    root2X = (-qB - D**0.5) / (2*qA)
    root1Y = normA*root1X + normB
    root2Y = normA*root2X + normB

    # Pick the solution that results in higher Y
    if root1Y > ymid:
        return (root1X, root1Y)
    else:
        return (root2X, root2Y)
        
def distance(x1, x2):
    """Computes distance between two points on the airfoil.

    Uses pythagoras theorem.

    Args:
      x1: first point
      x2: second point

    Returns:
      Distance in mm"""

    y1 = calcy(x1)
    y2 = calcy(x2)
    return ( (x1 - x2)**2 + (y1 - y2)**2 ) ** 0.5

def findnext(x, step):
    """Finds next point (x, y) to add to the curve.

    Uses binary search to determine the x value so that the
    total distance on the curve is within the resolution.

    Args:
      x: starting point
      step: resolution

    Returns
      Next point on the curve (x, y)."""
 
    prevdelta = 0
    delta = 0.1
    curr = x
    done = False

    while not done:
        d = distance(x, x + delta)
        if abs((d - step) / step) < 0.0001:
            done = True
        else:
            if d < step:
                newdelta = delta * 2
                prevdelta = delta
                delta = newdelta
            else:
                delta = (delta + prevdelta) / 2
    return ((x + delta), calcy((x + delta)))
        

def curve(step):
    """Generates the curve with given step.

    Iterable for points (x, y)."""

    x = 0
    y = calcy(x)
    yield (x, y)

    while x < C_PARAM:
        (x, y) = findnext(x, step)
        if (x > C_PARAM):
            break
        yield (x, y)
    yield (C_PARAM, calcy(C_PARAM))

def unit_normal(v1, v2, v3):
    """Calculates unit normal vector for the plane
       defined by the 3 points.
    """
    vec1 = numpy.subtract(v2, v1)
    vec2 = numpy.subtract(v3, v1)
    normal = numpy.cross(vec1, vec2).tolist()
    normal_len = numpy.linalg.norm(normal)
    if (abs(normal_len) < 0.000001):
        return None
    return numpy.multiply(normal, 1.0/normal_len);

def triangle(pt1, pt2, pt3):
    """Generates a single STL triangle based on 3 3D poitns.
    """

    un = unit_normal(pt1, pt2, pt3)
    if (un is None):
        return

    out.write("facet normal %f %f %f\n" % (un[0], un[1], un[2]))
    out.write("    outer loop\n")
    out.write("        vertex %f %f %f\n" % pt1)
    out.write("        vertex %f %f %f\n" % pt2)
    out.write("        vertex %f %f %f\n" % pt3)
    out.write("    endloop\n")
    out.write("endfacet\n")

def quad(pt1, pt2, pt3, pt4):
    """Generates a rectangle based on 4 3D points.

    Does this by generating 2 triangles."""

    triangle(pt1, pt2, pt3)
    triangle(pt1, pt3, pt4)

def triangles(prevx, prevy, x, y):
    """Generates a set of triangles for the curve.

    Args:
      prevx, prevy: previous point on the curve.
      x, y: current point on the curve."""

    # working surface
    quad((prevx, 0, prevy),
         (prevx, THICKNESS, prevy),
         (x, THICKNESS, y),
         (x, 0, y))

    # top side
    quad((prevx, THICKNESS, prevy),
         (prevx, THICKNESS, HEIGHT),
         (x, THICKNESS, HEIGHT),
         (x, THICKNESS, y))

    # bottom side
    quad((x, 0, y),
         (x, 0, HEIGHT),
         (prevx, 0, HEIGHT),
         (prevx, 0, prevy))


def stl_template(step, inv):
    """Generates the STL shape.

    Args:
      inv: True if the shape should be inverted."""

    max_thickness = T_PARAM * C_PARAM

    out.write("solid airfoil\n")

    if inv:
        z_start = max_thickness
    else:
        z_start = 0

    # Left wall
    quad((0, 0, HEIGHT),
         (0, THICKNESS, HEIGHT),
         (0, THICKNESS, z_start),
         (0, 0, z_start))

    # Right wall
    quad((C_PARAM, 0, z_start),
         (C_PARAM, THICKNESS, z_start),
         (C_PARAM, THICKNESS, HEIGHT),
         (C_PARAM, 0, HEIGHT))

    # Back wall
    quad((0, 0, HEIGHT),
         (C_PARAM, 0, HEIGHT),
         (C_PARAM, THICKNESS, HEIGHT),
         (0, THICKNESS, HEIGHT))

    prevx = -1
    prevy = -1
    for (x, y) in curve(step):
        if inv:
            y =  max_thickness - y
        if prevx == -1:
            (prevx, prevy) = (x, y)
            continue
        triangles(prevx, prevy, x, y)
        (prevx, prevy) = (x, y)

    out.write("endsolid airfoil\n")

def getthickness(args):
    """Computes the thickness parameter.

    Based on various args."""

    if (args.naca4 is not None and
        args.thick is not None):
       print 'Do not specify both thickness and NACA4.'
       sys.exit(1)

    if args.thick:
        t = float(args.thick) / float(args.chord) * 100
    elif args.naca4:
        if not args.naca4.startswith('00'):
            print ('Invalid curve: "%s". Only curves starting with 00 are supported at this point.' %
                   args.naca4)
            sys.exit(1)
        t = float(re.sub(r'^00', '', args.naca4))
    else:
        print 'Provide either thickness or NACA4 spec'
        sys.exit(1)

    return t / 100

def printcurve(args):
    prevx = -10
    prevy = 0.0
    for (x, y) in curve(step):
        if args.padding:
            if prevx != -10:
                newx, newy = padcurve(prevx, prevy, x, y, args.padding)
                out.write("%f %f\n" % (newx, newy))
            prevx = x
            prevy = y
        else:
            newx, newy = (x, y)
            out.write("%f %f\n" % (newx, newy))

def printstl_template(args):
    global HEIGHT, THICKNESS

    HEIGHT = args.height
    THICKNESS = args.thickness

    stl_template(step, args.inv)

def foil_shape_xx(y):
    min_func = lambda x: (foil_shape_func(x) - y)**2

    result = scipy.optimize.minimize(min_func, [10.e-10], bounds=[(1.0e-25, MIN_X)], method='L-BFGS-B')
    if not result.success:
        print 'Cannot optimize left, y = %f' % y
        return None
    left_x = result.x[0]

    result = scipy.optimize.minimize(min_func, [MIN_X+0.1], bounds=[(MIN_X, None)], method='L-BFGS-B')
    if not result.success:
        print 'Cannot optimize right, y = %f' % y
        return None
    right_x = result.x[0]
    return (left_x, right_x)

def func_to_foil_y(y):
    """Given y in func coords, return distance from tip r.
    """
    return (y - MIN_Y) / (MAX_Y - MIN_Y) * LENGTH

def foil_to_func_y(r):
    """Given distance from tip r, return y in func coords.
    """
    return float(r) / LENGTH * (MAX_Y - MIN_Y) + MIN_Y

def func_to_foil_x(x):
    """Given x in func coords, return distance from edge x.
    """
    return (x - MAX_X1) / (MAX_X2 - MAX_X1) * WIDTH

def foil_to_func_x(x):
    """Given distance from edge x, return x in func coords.
    """
    return x / WIDTH * (MAX_X2 - MAX_X1) + MAX_X1

def foil_shape_func(x):
    y = 1.0/x + 0.03 * x ** 4
    return y
# y = 1/x + 0.03 x^4
# dy / dx = log x + 0.12 x^3

def get_shape_from_to(r):
    """Returns the pair (x1, x2) of coordinates for foil shape.
    Arguments:
        r: Distance in mm from foil tip.
    """
    y = foil_to_func_y(r)
    (left, right) = foil_shape_xx(y)
    return (func_to_foil_x(left), func_to_foil_x(right))

def gen_shape(length, step):
    for r in range(1, length, step):
        yield get_shape_from_to(r, length)

def stl_foil(step):
    pass

def point(y):
    (left, right) = get_shape_from_to(y)
    print '%d %f %f' % (y, left, right)

def curve_point_list_for_distance(r):
    global T_PARAM, C_PARAM
    (left, right) = get_shape_from_to(r)
    C_PARAM = right - left
    local_thickness = float(THICKNESS) * C_PARAM / float(WIDTH)
    T_PARAM = float(local_thickness) / float(C_PARAM)
    my_curve = [ (x + left, y) for (x, y) in curve(step) ]
    return my_curve

def connect_curves(prev_curve, prev_distance, next_curve, next_distance):
    """Output triangles connecting two curves into a surface.
    """

    total_points_prev = len(prev_curve)
    total_points_next = len(next_curve)
    total_points_ratio = float(total_points_prev) / total_points_next
    running_ratio = 1.0
    done = False
    last_point_prev = 0
    last_point_next = 0
    while (last_point_prev + 1 < len(prev_curve) or
           last_point_next + 1 < len(next_curve)):

        take_next = running_ratio > total_points_ratio
        if (last_point_next + 1 >= len(next_curve)):
            take_next = False
        if (last_point_prev + 1 >= len(prev_curve)):
            take_next = True

        # Choose the curve from which to take the next point
        if (take_next):
            # Take point from "next" curve
            triangle(
                next_curve[last_point_next+1] + (next_distance,),
                prev_curve[last_point_prev] + (prev_distance,),
                next_curve[last_point_next] + (next_distance,))
            last_point_next += 1
        else:
            # Take point from "prev" curve
            triangle(
                prev_curve[last_point_prev+1] + (prev_distance,),
                prev_curve[last_point_prev] + (prev_distance,),
                next_curve[last_point_next] + (next_distance,))
            last_point_prev += 1
        running_ratio = float(last_point_prev+1) / (last_point_next+1)

def printstl_airfoil(args):
    global LENGTH, WIDTH, THICKNESS
    global MIN_X, MIN_Y
    global MAX_X1, MAX_X2, MAX_Y

    LENGTH = args.length
    WIDTH = args.width
    THICKNESS = args.thickness

    result = scipy.optimize.minimize(foil_shape_func, [0.1])
    if not result.success:
        print 'Cannot optimize airfoil shape!'
        return None

    MIN_X = result.x
    MIN_Y = foil_shape_func(MIN_X)
    MAX_Y = 35
    (MAX_X1, MAX_X2) = foil_shape_xx(MAX_Y)

    out.write("solid airfoil\n")
    prev_curve = None
    i = 0.0
    count = 0
    while count < 2:
        next_curve = curve_point_list_for_distance(i)
        next_curve.append((next_curve[len(next_curve)-1][0], 0.0))
        next_curve.insert(0, (next_curve[0][0], 0.0))
        if prev_curve is None:
            prev_curve = next_curve
            my_step = numpy.amax([ y for (x, y) in prev_curve])
            i += my_step
            continue
        connect_curves(prev_curve, i - my_step, next_curve, i)
        quad(
            prev_curve[len(prev_curve)-1] + (i - my_step,),
            next_curve[len(next_curve)-1] + (i,),
            next_curve[0] + (i,),
            prev_curve[0] + (i - my_step,))
        prev_curve = next_curve
        my_step = float(numpy.amax([ y for (x, y) in prev_curve]))
        i += my_step
        if i > LENGTH:
            i = LENGTH
            count += 1

    if args.box_length != 0:
        top_curve = [ (x, args.box_thickness) for (x, y) in next_curve ]
        top_curve.insert(0, (0, args.box_thickness))
        top_curve.insert(0, (0, 0))
        top_curve.append((args.box_width, args.box_thickness))
        top_curve.append((args.box_width, 0))
        connect_curves(next_curve, LENGTH, top_curve, LENGTH)
        quad(
            (0, 0, LENGTH + args.box_length),
            (0, args.box_thickness, LENGTH + args.box_length),
            (0, args.box_thickness, LENGTH),
            (0, 0, LENGTH))
        quad(
            (0, args.box_thickness, LENGTH + args.box_length),
            (args.box_width, args.box_thickness, LENGTH + args.box_length),
            (args.box_width, args.box_thickness, LENGTH),
            (0, args.box_thickness, LENGTH))
        quad(
            (args.box_width, 0, LENGTH),
            (args.box_width, args.box_thickness, LENGTH),
            (args.box_width, args.box_thickness, LENGTH + args.box_length),
            (args.box_width, 0, LENGTH + args.box_length))
        quad(
            (0, 0, LENGTH),
            (args.box_width, 0, LENGTH),
            (args.box_width, 0, LENGTH + args.box_length),
            (0, 0, LENGTH + args.box_length))
        quad(
            (args.box_width, 0, LENGTH + args.box_length),
            (args.box_width, args.box_thickness, LENGTH + args.box_length),
            (0, args.box_thickness, LENGTH + args.box_length),
            (0, 0, LENGTH + args.box_length))
    else:
        # Back wall
        bottom_straight = [ (x, 0) for (x, y) in next_curve ]
        connect_curves(next_curve, LENGTH, bottom_straight, LENGTH)
    out.write("endsolid airfoil\n")

parser = argparse.ArgumentParser(description='Generate airfoil data for plotting or 3D printing.')

sp = parser.add_subparsers()

sp_start = sp.add_parser('airfoil', help='Produce a 3D STL shape of airfoil.')
sp_start.add_argument('--length', type=float, help='Length of airfoil in mm, default is 1000', default=1000)
sp_start.add_argument('--width', type=float, help='Width of airfoil in mm, default is 300', default=300)
sp_start.add_argument('--thickness', type=float, help='Thickness of airfoil in mm, default is 25', default=25)
sp_start.add_argument('--box_length', type=int, help='Leave unprocessed material of this length at the beginning', default=0)
sp_start.add_argument('--box_width', type=int, help='Width of unprocessed material', default=300)
sp_start.add_argument('--box_thickness', type=int, help='Thickness of unprocessed material', default=13)

sp_start.set_defaults(func=printstl_airfoil)

sp_start = sp.add_parser('template', help='Produce a 3D STL shape used for 3D-printing of templates.')
sp_start.add_argument('--thickness', type=float, default=10, help='Thickness of the template in mm, default is 10.')
sp_start.add_argument('--height', type=float, default=100, help='Height of the template in mm, default is 100.')
sp_start.add_argument('--inv', action='store_true', default=False, help='Produce inverted template. Use when hand-sanding and checking. The non-inverted parts can be used for guiding the router.') 

sp_start.set_defaults(func=printstl_template)

sp_start = sp.add_parser('curve', help='Produce a set of x-y points for the curve, suitable for gnuplot. Use for printing the curve on paper and transferring to plywood.')
sp_start.set_defaults(func=printcurve)

parser.add_argument('--naca4', metavar='curvespec', default='0012', type=str,
                    help='NACA 4-digit curve spec. Only 00xx currently supported. Not required if thickness is specified. Default is 0012.')

parser.add_argument('--chord', '-c', metavar='len', default=100, type=int,
                    help='Chord length of the airfoil in mm. Default is 100')

parser.add_argument('--thick', '-t', metavar='t', type=int,
                    help='Max thickness of the airfoil in mm. Not required if NACA spec is given.')

parser.add_argument('--padding', '-p', metavar='p', type=int,
                    help='Padding of the airfoil in mm.')

parser.add_argument('--out', metavar='file', type=str,
                    help='Output file to write. If not specified, stdout is used.')

parser.add_argument('--resolution', metavar='density', type=float,
                    default=1, help='Curve resolution in points/mm. Default is 1.')

args = parser.parse_args()

if args.chord is None:
    print ('Chord length must be specified')
    sys.exit(1)

T_PARAM = getthickness(args)
C_PARAM = args.chord

step = 1.0 / args.resolution

if args.out is not None:
    out = open(args.out, 'w')
else:
    out = sys.stdout

args.func(args)

if args.out:
    out.close()
