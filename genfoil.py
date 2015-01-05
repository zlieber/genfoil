#!/usr/bin/python

import argparse
import re
import sys
import math
import numpy
import scipy.optimize
import textwrap

def derivative(func, x):
    epsilon = 1e-9
    diffx = 0
    try:
        before = func(x - epsilon)
        if (math.isnan(before)):
            raise "isNaN"
        diffx += epsilon
    except:
        before = func(x)

    try:
        after = func(x + epsilon)
        if (math.isnan(after)):
            raise "isNaN"
        diffx += epsilon
    except:
        after = func(x)
    result = (after - before) / diffx
    return result

def angular_distance(func, x1, x2):
    """Computes distance between two points on the airfoil.

    Uses pythagoras theorem.

    Args:
      x1: first point
      x2: second point

    Returns:
      Distance in mm"""

    curr_angle = math.degrees(math.atan(derivative(func, x1)))
    new_angle = math.degrees(math.atan(derivative(func, x2)))
    result = abs(curr_angle - new_angle)
    return result

def findnext(func, x, step, start_delta, minx=None, maxx=None):
    """Finds next point (x, y) to add to the curve.

    Uses binary search to determine the x value so that the
    change in derivative is less than the given # of degrees.

    Args:
      func: function definiing the curve
      x: starting point
      step: target angular change
      start_delta: distance from x for initial guess. May be negative
      minx: minimum value of x (lower bound).

    Returns
      Next point on the curve (x, y)."""

    prevdelta = 0
    delta = start_delta
    curr = x
    done = False

    while not done:
        d = angular_distance(func, x, x + delta)
        if abs((d - step) / step) < 0.0001:
            done = True
        else:
            if d < step:
                if minx is None:
                    newdelta = delta * 2
                else:
                    newdelta = (delta + minx - x) / 2
                    if abs(newdelta - delta) < 1.0e-10:
                        newdelta = minx - x
                        done = True
                prevdelta = delta
                delta = newdelta
            else:
                delta = (delta + prevdelta) / 2
    return ((x + delta), func((x + delta)))

class StlWriter:
    MM_INCH = 25.4

    def __init__(self, stream, units):
        self.out = stream
        self.write_header()
        self.transform_triangles = False
        self.units = units

    def write_header(self):
        self.out.write("solid airfoil\n")

    def convert(self, pt):
        if self.units == 'mm':
            return pt
        if self.units == 'in':
            return tuple(x / self.MM_INCH for x in pt)
        raise "Unsupported unit: %s" % self.units

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
        return numpy.multiply(normal, 1.0/normal_len)

    def setTransformTriangles(self):
        self.transform_triangles = True

    def triangle(self, pt1, pt2, pt3):
        """Generates a single STL triangle based on 3 3D poitns.
        """

        un = self.unit_normal(pt1, pt2, pt3)
        if (un is None):
            return

        # Convert axis
        if (self.transform_triangles):
            pt1 = (pt1[2], pt1[0], pt1[1])
            pt2 = (pt2[2], pt2[0], pt2[1])
            pt3 = (pt3[2], pt3[0], pt3[1])
            un = (un[2], un[0], un[1])

        # Convert units - but not for the normal vector.
        # Its size is "1" in whatever units.
        pt1 = self.convert(pt1)
        pt2 = self.convert(pt2)
        pt3 = self.convert(pt3)

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
    def __init__(self, chord, curvespec, tail_height=0.0, resolution=1.0):
        self.chord = chord
        self.calc_thickness(curvespec, chord)
        self.c_param = chord
        self.resolution = resolution

        # NACA curve does not converge to zero at the trailing edge.
        # We introduce a minor correction - convergence_param to
        # force that to happen.
        self.convergence_param = self.calcy(chord) - tail_height

    def calc_thickness(self, curvespec, chord):
        """Computes the thickness parameter.

        Based on various args."""

        if curvespec.startswith('00'):
            t = float(re.sub(r'^00', '', curvespec))
        else:
            t = float(curvespec) / float(chord) * 100 * 2

        self.t_param = t / 100
        self.thickness = self.chord * self.t_param

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

        x1 = x / self.c_param
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

    def findnext(self, x, step):
        return findnext(self.calcy_converged, x, step, 0.1)

    def curve(self):
        """Generates the curve with given step.

        Iterable for points (x, y)."""

        x = 0
        y = self.calcy_converged(x)
        yield (x, y)

        while x < self.c_param:
            (x, y) = self.findnext(x, self.resolution)
            if (x > self.c_param):
                break
            yield (x, y)
        yield (self.c_param, self.calcy_converged(self.c_param))

class FoilGenerator:
    def __init__(self, stl_writer, length,
                 width, curvespec, planform,
                 resolution=1.0, trail_width=1.0):
        self.writer = stl_writer
        self.resolution = resolution
        self.length = length
        self.width = width
        self.box_length = 0
        self.curvespec = curvespec
        self.trail_width = trail_width
        if self.curvespec.startswith('00'):
            self.thickness = float(re.sub(r'^00', '', curvespec)) / 100 * self.width
        else:
            self.thickness = float(curvespec)
        self.planform = planform

    def set_raw_box(self, length, thickness):
        self.box_length = length
        self.box_thickness = thickness

    def set_naca_curve(self, curvespec):
        self.curvespec = curvespec

    def foil_to_func_y(self, r):
        """Given distance from tip r, return y in func coords.
        """
        return (float(r) / self.length *
                (self.planform.top_y - self.planform.apex_y) +
                self.planform.apex_y)

    def func_to_foil_x(self, x):
        """Given x in func coords, return distance from edge x.
        """
        return ((x - self.planform.top_x_left) /
                (self.planform.top_x_right - self.planform.top_x_left) *
                self.width)

    def func_to_foil_y(self, y):
        """Given y in func coords, return distance from tip r.
        """
        return ((y - self.planform.apex_y) /
                (self.planform.top_y - self.planform.apex_y) * self.length)

    def get_shape_from_to(self, r):
        """Returns the pair (x1, x2) of coordinates for foil shape.
        Arguments:
            r: Distance in mm from foil tip.
        """
        y = self.foil_to_func_y(r)
        (left, right) = self.planform.foil_shape_xx(y)
        return (self.func_to_foil_x(left), self.func_to_foil_x(right))

    def get_next_curve(self, prev_curve):
        y = self.foil_to_func_y(prev_curve)
        next_y = self.planform.get_next_curve(y, self.resolution)
        result = self.func_to_foil_y(next_y)
        return result

    def curve_point_list_for_distance(self, r):
        (left, right) = self.get_shape_from_to(r)
        chord = right - left
        naca = NacaCurve(chord,
                         str(self.thickness * chord / self.width),
                         tail_height=self.trail_width, resolution = self.resolution)
        my_curve = [ (x + left, y) for (x, y) in naca.curve() ]
        return my_curve

    def generate_airfoil(self):
        prev_curve = None
        prev_z = 0

        z = self.func_to_foil_y(self.planform.get_start_curve(self.resolution))
        while prev_z + 0.001 < z or prev_curve is None:
            next_curve = self.curve_point_list_for_distance(z)
            print "Curve: (%f, %f, %f) - (%f, %f, %f) (%d points)" % (next_curve[0] + (z,) + next_curve[-1] + (z,) + (len(next_curve),))
            if prev_curve is None:
                prev_curve = next_curve
                # Front wall
                for idx in range(1, len(prev_curve)):
                    self.writer.triangle(
                        prev_curve[idx-1] + (z,),
                        prev_curve[idx] + (z,),
                        (prev_curve[-1][0], 0.0, z))
                prev_z = z
                z = self.get_next_curve(z)
                continue
            self.writer.connect_curves(prev_curve, prev_z, next_curve, z)
            self.writer.quad(
                prev_curve[-1] + (prev_z,),
                next_curve[-1] + (z,),
                (next_curve[-1][0], 0.0, z),
                (prev_curve[-1][0], 0.0, prev_z))
            self.writer.quad(
                (prev_curve[-1][0], 0.0, prev_z),
                (next_curve[-1][0], 0.0, z),
                (next_curve[0][0], 0.0, z),
                (prev_curve[0][0], 0.0, prev_z))
            prev_curve = next_curve
            prev_z = z
            z = self.get_next_curve(z)

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
                (right, self.trail_width),
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

class TemplateGenerator:
    def __init__(self, stl_writer, naca, height, thickness):
        self.writer = stl_writer
        self.naca = naca
        self.height = height
        self.thickness = thickness

    def generate_template(self, inv):
        """Generates the STL shape.

        Args:
          inv: True if the shape should be inverted."""

        max_thickness = self.naca.t_param * self.naca.c_param
        if (inv):
            top_curve = [ (x, y + self.height) for (x, y) in self.naca.curve() ]
            top_y = self.height
        else:
            flat = 1
            top_curve = [ (x + flat, self.height + max_thickness - y) for (x, y) in self.naca.curve() ]
            top_y = self.height + max_thickness
            top_curve.insert(0, (0, top_y))

        self.writer.connect_curves(top_curve, 0, top_curve, self.thickness)
        bottom_curve = [
            (0, 0),
            (top_curve[-1][0], 0) ]
        self.writer.connect_curves(bottom_curve, 0, top_curve, 0)
        self.writer.connect_curves(top_curve, self.thickness, bottom_curve, self.thickness)
        self.writer.quad(
            (top_curve[0][0], top_y, self.thickness),
            (top_curve[0][0], top_y, 0),
            (top_curve[0][0], 0, 0),
            (top_curve[0][0], 0, self.thickness))
        self.writer.quad(
            (top_curve[-1][0], top_y, 0),
            (top_curve[-1][0], top_y, self.thickness),
            (top_curve[-1][0], 0, self.thickness),
            (top_curve[-1][0], 0, 0))
        self.writer.quad(
            (top_curve[0][0], 0, 0),
            (top_curve[-1][0], 0, 0),
            (top_curve[-1][0], 0, self.thickness),
            (top_curve[0][0], 0, self.thickness))

class Planform:
    def __init__(self, name, description,
                 func_left, top_y, range_x,
                 func_right=None):
        self.name = name
        self.description = description
        self.func_left = func_left
        self.func_right = func_right
        self.top_y = top_y
        self.range_x = range_x
        if func_right is None:
            (self.apex_x, self.apex_y) = self.calc_apex()
            self.func_right = func_left
            self.dual = False
        else:
            self.bottom_y = 0
            self.apex_y = 0
            self.apex_x = self.calc_apex_dual()
            self.dual = True
        (self.top_x_left, self.top_x_right) = self.foil_shape_xx(self.top_y)

    def find_root(self, func):
        min_func = lambda x: func(x)**2
        result = scipy.optimize.minimize(min_func, [0.1])
        if not result.success:
            # TODO: proper errors
            raise "Cannot find root"
        return result.x[0]

    def calc_apex_dual(self):
        left_root = self.find_root(self.func_left)
        right_root = self.find_root(self.func_right)
        return (left_root + right_root) / 2

    def calc_apex(self):
        result = scipy.optimize.minimize(self.func_left, [0.1])
        if not result.success:
            # TODO: proper errors
            raise 'Cannot optimize airfoil shape!'
        return (result.x[0], self.func_left(result.x[0]))

    def get_start_curve(self, resolution):
        if self.dual:
            return 0
        else:
            return self.get_next_curve(self.apex_y, resolution)

    def foil_shape_xx(self, y):
        min_func = lambda x: (self.func_left(x) - y)**2

        result = scipy.optimize.minimize_scalar(min_func,
                                                bounds=[self.range_x[0], self.apex_x],
                                                method='bounded')
        if not result.success:
            print 'Cannot optimize left, y = %f' % y
            return None
        left_x = result.x

        min_func = lambda x: (self.func_right(x) - y)**2
        result = scipy.optimize.minimize_scalar(min_func,
                                                bounds=[self.apex_x, self.range_x[1]],
                                                method='bounded')
        if not result.success:
            print 'Cannot optimize right, y = %f' % y
            return None
        right_x = result.x
        return (left_x, right_x)

    def get_next_curve(self, prev_curve, resolution):
        if prev_curve == self.apex_y:
            (left, right) = (self.apex_x, self.apex_x)
        else:
            (left, right) = self.foil_shape_xx(prev_curve)
        (x1, y1) = findnext(self.func_left, left, resolution, -1e-9, self.top_x_left)
        (x2, y2) = findnext(self.func_right, right, resolution, 1e-9, self.top_x_right)
        if (y1 > y2):
            (resx, resy) = (x2, y2)
        else:
            (resx, resy) = (x1, y1)
        return resy

def printcurve(args):
    naca = NacaCurve(args.chord, args.curvespec,
                     resolution=args.resolution)

    prevx = -10
    prevy = 0.0
    for (x, y) in naca.curve():
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
    writer = StlWriter(out, args.stl_units)
    naca = NacaCurve(args.chord, args.curvespec,
                     resolution=args.resolution)
    gen = TemplateGenerator(writer, naca, args.height, args.thickness)
    gen.generate_template(args.inv)
    writer.close()

def printstl_airfoil(args):
    writer = StlWriter(out, args.stl_units)
    if args.axis != 'y' and args.axis != 'z':
        raise "Unsupported axis. Only y and z are supported."

    if args.axis == 'z':
        writer.setTransformTriangles()

    planform = PLANFORMS[args.planform]

    gen = FoilGenerator(writer,
                        args.length,
                        args.width,
                        args.curvespec,
                        planform,
                        trail_width=args.trail_width,
                        resolution=args.resolution)
    if args.box_length != 0:
        gen.set_raw_box(args.box_length,
                        args.box_thickness)

    gen.generate_airfoil()
    writer.close()

def print_planforms(args):
    print """
Planform is the shape of a rudder when laid flat on
the ground and looked at from above. These planforms
are currently supported:

"""
    for (name, planform) in PLANFORMS.items():
        print name.ljust(10).rjust(15)
        lines = textwrap.wrap(planform.description, 40)
        for l in lines:
            print l.lstrip().ljust(60).rjust(80)

numpy.seterr(all='raise')

PLANFORMS = {}

# Init planforms
planform = Planform(
    "ellipse",
    """
Roughly elliptical-shaped tip. Leading edge nearly vertical,
trailing edge somewhat at an angle. Represented by function
1/x + 0.03 * x^4, until maximum value of y=35.
""",
    lambda x: 1.0/x + 0.03 * x ** 4,
    35,
    (1.0e-25, 6))
PLANFORMS[planform.name] = planform

planform = Planform(
    "straight",
    """
Vertical planform, no rounded tip.
""",
    lambda x: -1000*(x-1), 1,
    (0, 2.1),
    func_right=lambda x: 1e3*(x-2))
PLANFORMS[planform.name] = planform

planform = Planform(
    "sweepback",
    """
Straight edges, swept back planform, slightly tapered.
""",
    lambda x: -5*(x-1), 5,
    (0, 2.1),
    func_right=lambda x: -10*(x-2))
PLANFORMS[planform.name] = planform

planform = Planform(
    "sine",
    """
Shaped as a stretched sine function.
""",
    lambda x: math.cos(x) + 1, 0.5,
    (0, 2*math.pi))
PLANFORMS[planform.name] = planform

parser = argparse.ArgumentParser(
    description='Generate airfoil data for plotting or 3D printing.')

sp = parser.add_subparsers()

sp_start = sp.add_parser('airfoil', help="""
Produce a 3D STL shape of airfoil. Only top half is generated.
Airfoil includes planform, taper and optionally a box of
unprocessed material on one end.
""")

sp_start.add_argument('--length', type=float,
                      help='Length of airfoil in mm, default is 1000',
                      default=1000)
sp_start.add_argument('--width', type=float,
                      help='Width of airfoil in mm, default is 300',
                      default=300)
sp_start.add_argument('--planform', type=str, default='ellipse',
                      help="""
Specifies the planform to use. Run "genfoil.py planform" to see the
complete list of supported planforms. Default is 'ellipse'.
""")
sp_start.add_argument('--resolution', type=float,
                      default=1,
                      help="""
Curve angular resolution in degrees. This is maximum curve
direction change between two points. The value is used
for both foil shape and planform shape. Default is 1.
""")

sp_start.add_argument('--curvespec', type=str, help="""
If starts with "00", this is the NACA4 specification of the airfoil.
If starts with a number [1-9], this is the maximum thickness of the
airfoil in mm. Default is 12
""", default='12')

sp_start.add_argument('--axis', type=str, default='y', help="""
Specify axis that will contain the thickness of the airfoil. Default is y.
""")
sp_start.add_argument('--stl_units', type=str, default='mm', help="""
Units to use when generating STL. Supported values are mm and in. Default is mm.
""")

sp_start.add_argument('--trail_width', type=float, help="""
Width of the trailing edge in mm. Trailing edge cannot have zero
thickness, and all cross-section curves are adjusted to converge
to this value. Default is 1mm.
""", default=1)

sp_start.add_argument('--box_length', type=int, help="""
Leave unprocessed material of this length at the beginning
""", default=0)

sp_start.add_argument('--box_thickness', type=float,
                      help='Thickness of unprocessed material', default=13)

sp_start.set_defaults(func=printstl_airfoil)

sp_start = sp.add_parser('template', help="""
Produce a 3D STL shape used for 3D-printing of guides or templates.
Templates are shaped so that they can be used to verify foil shape,
or guide an instrument such as a router.
""")

sp_start.add_argument('--curvespec', type=str, help="""
If starts with "00", this is the NACA4 specification of the airfoil.
If starts with a number [1-9], this is the maximum thickness of the
airfoil in mm. Default is 12
""", default='12')

sp_start.add_argument('--thickness', type=float, default=10,
                      help='Thickness of the template in mm, default is 10.')
sp_start.add_argument('--height', type=float, default=100,
                      help='Height of the template in mm, default is 100.')
sp_start.add_argument('--inv', action='store_true', default=False, help="""
Produce inverted template. Use when hand-sanding and checking.
The non-inverted parts can be used for guiding the router.
""") 
sp_start.add_argument('--stl_units', type=str, default='mm', help="""
Units to use when generating STL. Supported values are mm and in. Default is mm.
""")

sp_start.add_argument('--resolution', type=float,
                      default=1,
                      help="""
Curve angular resolution in degrees. This is maximum curve
direction change between two points. Default is 1.
""")
sp_start.add_argument('--chord', '-c', metavar='len', default=100, type=int,
                      help='Chord length of the airfoil in mm. Default is 100')

sp_start.set_defaults(func=printstl_template)

sp_start = sp.add_parser('curve', help="""
Produce a set of x-y points for the curve, suitable for gnuplot, exel
or any plotting program. Use for printing the curve on paper and
transferring to plywood.
""")

sp_start.set_defaults(func=printcurve)

sp_start.add_argument('--chord', '-c', metavar='len', default=100, type=int,
                      help='Chord length of the airfoil in mm. Default is 100')
sp_start.add_argument('--curvespec', type=str, help="""
If starts with "00", this is the NACA4 specification of the airfoil.
If starts with a number [1-9], this is the maximum thickness of the
airfoil in mm. Default is 12
""", default='12')
sp_start.add_argument('--resolution', type=float,
                      default=1,
                      help="""
Curve angular resolution in degrees. This is maximum curve
direction change between two points. Default is 1.
""")
sp_start.add_argument('--padding', '-p', metavar='p', type=int, help="""
Padding of the airfoil in mm. This is useful if you guide your
instrument some distance from the surface. That distance is added
to foil shape.
""")

sp_start = sp.add_parser('planform', help="""
Show supported planforms.
""")
sp_start.set_defaults(func=print_planforms)

parser.add_argument('--out', '-o', metavar='file', type=str, help="""
Output file to write. If not specified, stdout is used.
""")

args = parser.parse_args()

if args.out is not None:
    out = open(args.out, 'w')
else:
    out = sys.stdout

args.func(args)

if args.out:
    out.close()
