#!/usr/bin/python

import argparse
import re
import sys
import numpy
import scipy.optimize

class StlWriter:
    def __init__(self, stream):
        self.out = stream
        self.write_header()

    def write_header(self):
        self.out.write("solid airfoil\n")

    def close(self):
        out.write("endsolid airfoil\n")

    def unit_normal(self, v1, v2, v3):
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

    def triangle(self, pt1, pt2, pt3):
        """Generates a single STL triangle based on 3 3D poitns.
        """

        un = self.unit_normal(pt1, pt2, pt3)
        if (un is None):
            return

        self.out.write("facet normal %f %f %f\n" % (un[0], un[1], un[2]))
        self.out.write("    outer loop\n")
        self.out.write("        vertex %f %f %f\n" % pt1)
        self.out.write("        vertex %f %f %f\n" % pt2)
        self.out.write("        vertex %f %f %f\n" % pt3)
        self.out.write("    endloop\n")
        self.out.write("endfacet\n")

    def quad(self, pt1, pt2, pt3, pt4):
        """Generates a rectangle based on 4 3D points.

        Does this by generating 2 triangles."""

        self.triangle(pt1, pt2, pt3)
        self.triangle(pt1, pt3, pt4)

    def connect_curves(self, prev_curve, prev_distance, next_curve, next_distance):
        """Output triangles connecting two curves into a surface.
        """

        total_points_prev = len(prev_curve)
        total_points_next = len(next_curve)
        total_points_ratio = float(total_points_prev) / total_points_next
        running_ratio = 1.0
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
                self.triangle(
                    next_curve[last_point_next+1] + (next_distance,),
                    prev_curve[last_point_prev] + (prev_distance,),
                    next_curve[last_point_next] + (next_distance,))
                last_point_next += 1
            else:
                # Take point from "prev" curve
                self.triangle(
                    prev_curve[last_point_prev+1] + (prev_distance,),
                    prev_curve[last_point_prev] + (prev_distance,),
                    next_curve[last_point_next] + (next_distance,))
                last_point_prev += 1
            running_ratio = float(last_point_prev+1) / (last_point_next+1)

class NacaCurve:
    def __init__(self, chord, naca_spec=None, thickness=None, tail_height=0.0):
        if (naca_spec is None) == (thickness is None):
            raise "Either spec or thickness must be given"
        self.chord = chord
        self.t_param = self.getthickness(naca_spec, thickness, chord)
        self.c_param = chord

        # NACA curve does not converge to zero at the trailing edge.
        # We introduce a minor correction - convergence_param to
        # force that to happen.
        self.convergence_param = self.calcy(chord) - tail_height

    def getthickness(self, naca4, thick, chord):
        """Computes the thickness parameter.

        Based on various args."""

        if (naca4 is not None and
            thick is not None):
           raise 'Do not specify both thickness and NACA4.'

        if thick is not None:
            t = float(thick) / float(chord) * 100
        elif naca4 is not None:
            if not naca4.startswith('00'):
                raise ('Invalid curve: "%s". Only curves starting with 00 are supported at this point.' %
                       naca4)
            t = float(re.sub(r'^00', '', naca4))
        else:
            raise 'Provide either thickness or NACA4 spec'

        return t / 100

    def calcy(self, x):
        """Calculate a single airfoil point.

        Uses NACA formula.

        Args:
          x: X coordinate

        Returns:
          The calculated point (x, y)"""

        A1 = 0.2969
        A2 = -0.1260
        A3 = -0.3516
        A4 = 0.2843
        A5 = -0.1015

        x1 = x / self.c_param;
        return (self.t_param / 0.2 * self.c_param) * (
            A1 * x1 ** 0.5 +
            A2 * x1 +
            A3 * x1 ** 2 +
            A4 * x1 ** 3 +
            A5 * x1 ** 4)

    def calcy_converged(self, x):
        y = self.calcy(x)
        y -= self.convergence_param * x / self.c_param
        return y

    def padcurve(self, x1, y1, x2, y2, r):
        # midpoint between two points
        xmid = (x1 + x2) / 2
        ymid = (y1 + y2) / 2

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

    def distance(self, x1, x2):
        """Computes distance between two points on the airfoil.

        Uses pythagoras theorem.

        Args:
          x1: first point
          x2: second point

        Returns:
          Distance in mm"""

        y1 = self.calcy_converged(x1)
        y2 = self.calcy_converged(x2)
        return ( (x1 - x2)**2 + (y1 - y2)**2 ) ** 0.5

    def findnext(self, x, step):
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
            d = self.distance(x, x + delta)
            if abs((d - step) / step) < 0.0001:
                done = True
            else:
                if d < step:
                    newdelta = delta * 2
                    prevdelta = delta
                    delta = newdelta
                else:
                    delta = (delta + prevdelta) / 2
        return ((x + delta), self.calcy_converged((x + delta)))

    def curve(self, step):
        """Generates the curve with given step.

        Iterable for points (x, y)."""

        x = 0
        y = self.calcy_converged(x)
        yield (x, y)

        while x < self.c_param:
            (x, y) = self.findnext(x, step)
            if (x > self.c_param):
                break
            yield (x, y)
        yield (self.c_param, self.calcy_converged(self.c_param))

class FoilGenerator:

    def __init__(self, stl_writer, length, width, thickness):
        self.writer = stl_writer
        self.length = length
        self.width = width
        self.thickness = thickness
        self.box_length = 0
        self.curvespec = None
        self.curve_res = 1
        self.planform_res = 1
        self.init_planform_params()

    def init_planform_params(self):
        result = scipy.optimize.minimize(self.planform_func, [0.1])
        if not result.success:
            # TODO: proper errors
            raise 'Cannot optimize airfoil shape!'

        self.planform_min_x = result.x
        self.planform_min_y = self.planform_func(self.planform_min_x)
        self.planform_max_y = 35
        (self.planform_max_x1, self.planform_max_x2) = self.foil_shape_xx(
            self.planform_max_y)

    def set_raw_box(self, length, thickness):
        self.box_length = length
        self.box_thickness = thickness

    def set_naca_curve(self, curvespec):
        self.curvespec = curvespec

    def set_precision(self, curve_res, planform_res):
        self.curve_res = curve_res
        self.planform_res = planform_res

    def planform_func(self, x):
        y = 1.0/x + 0.03 * x ** 4
        return y

    def foil_shape_xx(self, y):
        min_func = lambda x: (self.planform_func(x) - y)**2

        result = scipy.optimize.minimize(min_func, [10.e-10],
                                         bounds=[(1.0e-25, self.planform_min_x)],
                                         method='L-BFGS-B')
        if not result.success:
            print 'Cannot optimize left, y = %f' % y
            return None
        left_x = result.x[0]

        result = scipy.optimize.minimize(min_func,
                                         [self.planform_min_x + 0.1],
                                         bounds=[(self.planform_min_x, None)],
                                         method='L-BFGS-B')
        if not result.success:
            print 'Cannot optimize right, y = %f' % y
            return None
        right_x = result.x[0]
        return (left_x, right_x)

    def foil_to_func_y(self, r):
        """Given distance from tip r, return y in func coords.
        """
        return (float(r) / self.length *
                (self.planform_max_y - self.planform_min_y) +
                self.planform_min_y)

    def func_to_foil_x(self, x):
        """Given x in func coords, return distance from edge x.
        """
        return ((x - self.planform_max_x1) /
                (self.planform_max_x2 - self.planform_max_x1) *
                self.width)

    def get_shape_from_to(self, r):
        """Returns the pair (x1, x2) of coordinates for foil shape.
        Arguments:
            r: Distance in mm from foil tip.
        """
        y = self.foil_to_func_y(r)
        (left, right) = self.foil_shape_xx(y)
        return (self.func_to_foil_x(left), self.func_to_foil_x(right))

    def curve_point_list_for_distance(self, r):
        (left, right) = self.get_shape_from_to(r)
        chord = right - left
        naca = NacaCurve(chord, None, self.thickness * chord / self.width, tail_height=1.0)
        my_curve = [ (x + left, y) for (x, y) in naca.curve(step) ]
        return my_curve

    def generate_airfoil(self):
        prev_curve = None
        prev_z = 0

        # TODO(zlieber): parametrize this
        i = 1.0
        count = 0
        while count < 2:
            next_curve = self.curve_point_list_for_distance(i)
            print "Curve: (%f, %f) - (%f, %f) (%d points)" % (next_curve[0] + next_curve[-1] + (len(next_curve),))
            if prev_curve is None:
                prev_curve = next_curve
                # Front wall
                bottom_curve = [
                    (prev_curve[0][0], 0),
                    (prev_curve[-1][0], 0)]
                self.writer.connect_curves(bottom_curve, i, prev_curve, i);
                prev_z = i
                my_step = numpy.amax([ y for (x, y) in prev_curve])
                i += my_step
                continue
            self.writer.connect_curves(prev_curve, prev_z, next_curve, i)
            self.writer.quad(
                prev_curve[-1] + (prev_z,),
                next_curve[-1] + (i,),
                (next_curve[-1][0], 0.0, i),
                (prev_curve[-1][0], 0.0, prev_z))
            self.writer.quad(
                (prev_curve[-1][0], 0.0, prev_z),
                (next_curve[-1][0], 0.0, i),
                (next_curve[0][0], 0.0, i),
                (prev_curve[0][0], 0.0, prev_z))
            prev_curve = next_curve
            prev_z = i
            my_step = float(numpy.amax([ y for (x, y) in prev_curve]))
            i += my_step
            if i > self.length:
                i = self.length
                count += 1

        left = next_curve[0][0]
        right = next_curve[-1][0]
        back = prev_z

        if self.box_length != 0:
            top_curve = [ (x, self.box_thickness) for (x, y) in next_curve ]
            self.writer.connect_curves(next_curve, back, top_curve, back)
            back_curve = [ (left, self.box_thickness),
                           (right, self.box_thickness) ]
            self.writer.connect_curves(top_curve, back, back_curve, back + self.box_length)
            self.writer.quad(
                (left, 0, back + self.box_length),
                (left, self.box_thickness, back + self.box_length),
                (left, self.box_thickness, back),
                (left, 0, back))

            back_vertical = [
                (right, self.box_thickness),
                (right, 0) ]
            front_vertical = [
                (right, self.box_thickness),
                (right, 1.0),
                (right, 0.0) ]
            self.writer.connect_curves(front_vertical, back, back_vertical, back + self.box_length)

            self.writer.quad(
                (left, 0, back),
                (right, 0, back),
                (right, 0, back + self.box_length),
                (left, 0, back + self.box_length))
            self.writer.quad(
                (right, 0, back + args.box_length),
                (right, args.box_thickness, back + args.box_length),
                (left, args.box_thickness, back + args.box_length),
                (left, 0, back + args.box_length))
        else:
            # Back wall
            bottom_straight = [ 
                (next_curve[0][0], 0),
                (next_curve[-1][0], 0) ]
            self.writer.connect_curves(next_curve, prev_z, bottom_straight, prev_z)


def triangles(writer, prevx, prevy, x, y):
    """Generates a set of triangles for the curve.

    Args:
      writer: stl writer to use.
      prevx, prevy: previous point on the curve.
      x, y: current point on the curve."""

    # working surface
    writer.quad((prevx, 0, prevy),
         (prevx, THICKNESS, prevy),
         (x, THICKNESS, y),
         (x, 0, y))

    # top side
    writer.quad((prevx, THICKNESS, prevy),
         (prevx, THICKNESS, HEIGHT),
         (x, THICKNESS, HEIGHT),
         (x, THICKNESS, y))

    # bottom side
    writer.quad((x, 0, y),
         (x, 0, HEIGHT),
         (prevx, 0, HEIGHT),
         (prevx, 0, prevy))

def stl_template(step, inv, naca):
    """Generates the STL shape.

    Args:
      inv: True if the shape should be inverted."""

    max_thickness = naca.t_param * naca.c_param

    writer = StlWriter(out)

    if inv:
        z_start = max_thickness
    else:
        z_start = 0

    # Left wall
    writer.quad((0, 0, HEIGHT),
                (0, THICKNESS, HEIGHT),
                (0, THICKNESS, z_start),
                (0, 0, z_start))

    # Right wall
    writer.quad((naca.c_param, 0, z_start),
                (naca.c_param, THICKNESS, z_start),
                (naca.c_param, THICKNESS, HEIGHT),
                (naca.c_param, 0, HEIGHT))

    # Back wall
    writer.quad((0, 0, HEIGHT),
                (naca.c_param, 0, HEIGHT),
                (naca.c_param, THICKNESS, HEIGHT),
                (0, THICKNESS, HEIGHT))

    prevx = -1
    prevy = -1
    for (x, y) in naca.curve(step):
        if inv:
            y =  max_thickness - y
        if prevx == -1:
            (prevx, prevy) = (x, y)
            continue
        triangles(writer, prevx, prevy, x, y)
        (prevx, prevy) = (x, y)

    writer.close()

def printcurve(args):
    naca = NacaCurve(args.chord, args.naca4, args.thick)

    prevx = -10
    prevy = 0.0
    for (x, y) in naca.curve(step):
        if args.padding:
            if prevx != -10:
                newx, newy = naca.padcurve(prevx, prevy, x, y, args.padding)
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
    naca = NacaCurve(args.chord, args.naca4, args.thick)

    stl_template(step, args.inv, naca)

def printstl_airfoil(args):
    writer = StlWriter(out)
    gen = FoilGenerator(writer,
                        args.length,
                        args.width,
                        args.thickness)
    if args.box_length != 0:
        gen.set_raw_box(args.box_length,
                        args.box_thickness)

    gen.generate_airfoil()
    writer.close()

parser = argparse.ArgumentParser(description='Generate airfoil data for plotting or 3D printing.')

sp = parser.add_subparsers()

sp_start = sp.add_parser('airfoil', help='Produce a 3D STL shape of airfoil.')
sp_start.add_argument('--length', type=float, help='Length of airfoil in mm, default is 1000', default=1000)
sp_start.add_argument('--width', type=float, help='Width of airfoil in mm, default is 300', default=300)
sp_start.add_argument('--thickness', type=float, help='Thickness of airfoil in mm, default is 25', default=25)
sp_start.add_argument('--box_length', type=int, help='Leave unprocessed material of this length at the beginning', default=0)
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

step = 1.0 / args.resolution

if args.out is not None:
    out = open(args.out, 'w')
else:
    out = sys.stdout

args.func(args)

if args.out:
    out.close()
