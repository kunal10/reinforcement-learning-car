import random
import math
import numpy as np

import pygame
from pygame.color import THECOLORS

import pymunk
from pymunk.vec2d import Vec2d
from pymunk.pygame_util import draw

# PyGame init
width = 1000
height = 700
pygame.init()
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# Turn off alpha since we don't use it.
screen.set_alpha(None)

# Showing sensors and redrawing slows things down.
show_sensors = True
draw_screen = True

use_obstacles = False

# Whether red team agent should be added
use_red_team = True
# Whether red team agent is random or trained
trained_red_team = True

class GameState:
    def __init__(self):
        # Physics stuff.
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0., 0.)

        # Record steps.
        self.num_steps = 0

        # Global-ish.
        self.car_crashed = False
        # Create the car.
        self.create_car(100, 100, 0.5)

        # Create a cat.
        if use_red_team:
            self.cat_crashed = False
            self.create_cat()

        # Create walls.
        static = [
            pymunk.Segment(
                self.space.static_body,
                (0, 1), (0, height), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, height), (width, height), 1),
            pymunk.Segment(
                self.space.static_body,
                (width-1, height), (width-1, 1), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, 1), (width, 1), 1)
        ]
        for s in static:
            s.friction = 1.
            s.group = 1
            s.collision_type = 1
            s.color = THECOLORS['red']
        self.space.add(static)

        # Create some obstacles, semi-randomly.
        # We'll create three and they'll move around to prevent over-fitting.
        self.obstacles = []
        if use_obstacles:
            self.obstacles.append(self.create_obstacle(200, 350, 100))
            self.obstacles.append(self.create_obstacle(700, 200, 125))
            self.obstacles.append(self.create_obstacle(600, 600, 35))

    def create_obstacle(self, x, y, r):
        c_body = pymunk.Body(pymunk.inf, pymunk.inf)
        c_shape = pymunk.Circle(c_body, r)
        c_shape.elasticity = 1.0
        c_body.position = x, y
        c_shape.color = THECOLORS["blue"]
        self.space.add(c_body, c_shape)
        return c_body

    def create_cat(self):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.cat_body = pymunk.Body(1, inertia)
        self.cat_body.position = 50, height - 100
        self.cat_shape = pymunk.Circle(self.cat_body, 25)
        self.cat_shape.color = THECOLORS["red"]
        self.cat_shape.elasticity = 1.0
        self.cat_shape.angle = 0.5
        driving_direction = Vec2d(1, 0).rotated(self.cat_body.angle)
        # TODO : Do we need to apply impulse here ?
        self.cat_body.apply_impulse(driving_direction)
        self.space.add(self.cat_body, self.cat_shape)

    def create_car(self, x, y, r):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.car_body = pymunk.Body(1, inertia)
        self.car_body.position = x, y
        self.car_shape = pymunk.Circle(self.car_body, 25)
        self.car_shape.color = THECOLORS["green"]
        self.car_shape.elasticity = 1.0
        self.car_body.angle = r
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.apply_impulse(driving_direction)
        self.space.add(self.car_body, self.car_shape)

    def frame_step(self, car_action, cat_action=None):
        # Move obstacles.
        self.move_obstacles()

        # Move cat.
        cat_driving_direction = self.move_cat(cat_action)

        # Move car
        car_driving_direction = self.move_car(car_action)

        # Update the screen and stuff.
        screen.fill(THECOLORS["black"])
        draw(screen, self.space)
        self.space.step(1./10)
        if draw_screen:
            pygame.display.flip()
        clock.tick()

        # Get the current cat location and the readings there.
        cat_state = None
        if use_red_team and trained_red_team:
            cat_x, cat_y = self.cat_body.position
            cat_readings = self.get_sonar_readings(cat_x, cat_y, self.cat_body.angle)
            cat_state = np.array([cat_readings])
            if self.is_crashed(cat_readings):
                self.cat_crashed = True
                self.recover_cat_crash(cat_driving_direction)

        # Get the current car location and the readings there.
        car_x, car_y = self.car_body.position
        car_readings = self.get_sonar_readings(car_x, car_y, self.car_body.angle)
        car_state = np.array([car_readings])

        # Set the reward.
        # Car crashed when any reading == 1
        if self.is_crashed(car_readings):
            self.car_crashed = True
            reward = -500
            self.recover_car_crash(car_driving_direction)
        else:
            # Higher readings are better, so return the sum.
            reward = -5 + int(sum(car_readings) / 10)

        self.num_steps += 1
        return reward, car_state, cat_state

    def move_obstacles(self):
        if self.num_steps % 100 == 0:
            # Randomly move obstacles around.
            for obstacle in self.obstacles:
                speed = random.randint(1, 5)
                direction = Vec2d(1, 0).rotated(self.car_body.angle + random.randint(-2, 2))
                obstacle.velocity = speed * direction

    def move_car(self, action):
        return self.move_body(self.car_body, action)

    def move_cat(self, action=None):
        if not use_red_team:
            return
        if trained_red_team:
            assert action is not None
            return self.move_body(self.cat_body, action)
        else:
            if self.num_steps % 5 == 0:
                speed = random.randint(20, 200)
                self.cat_body.angle -= random.randint(-1, 1)
                direction = Vec2d(1, 0).rotated(self.cat_body.angle)
                self.cat_body.velocity = speed * direction
                return

    @staticmethod
    def move_body(body, action):
        if action == 0:  # Turn left.
            body.angle -= .2
        elif action == 1:  # Turn right.
            body.angle += .2
        driving_direction = Vec2d(1, 0).rotated(body.angle)
        body.velocity = 100 * driving_direction
        return driving_direction

    @staticmethod
    # TODO : Fix this to consider collisions from back as well.
    def is_crashed(readings):
        return readings[0] == 1 or readings[1] == 1 or readings[2] == 1

    def recover_car_crash(self, car_driving_direction):
        """
        We hit something, so recover.
        """
        while self.car_crashed:
            # Go backwards.
            self.car_body.velocity = -100 * car_driving_direction
            self.car_crashed = False
            for i in range(10):
                self.car_body.angle += .2  # Turn a little.
                screen.fill(THECOLORS["yellow"])
                draw(screen, self.space)
                # Set velocity of other agents to 0 before retracting
                cat_velocity = self.cat_body.velocity
                self.cat_body.velocity = pymunk.Vec2d(0., 0.)
                self.space.step(1./10)
                if draw_screen:
                    pygame.display.flip()
                # Fill the screen black to avoid false detection by other agents
                screen.fill(THECOLORS["black"])
                self.cat_body.velocity = cat_velocity
                draw(screen, self.space)
                clock.tick()

    def recover_cat_crash(self, cat_driving_direction):
        """
        We hit something, so recover.
        """
        while self.cat_crashed:
            # Go backwards.
            self.cat_body.velocity = -100 * cat_driving_direction
            self.cat_crashed = False
            for i in range(10):
                self.cat_body.angle += .2  # Turn a little.
                screen.fill(THECOLORS["orange"])  # Red is scary!
                draw(screen, self.space)
                # Set velocity of other agents to 0 before retracting
                car_velocity = self.car_body.velocity
                self.car_body.velocity = pymunk.Vec2d(0., 0.)
                self.space.step(1./10)
                if draw_screen:
                    pygame.display.flip()
                # Fill the screen black to avoid false detection by other agents
                screen.fill(THECOLORS["black"])  # Red is scary!
                self.car_body.velocity = car_velocity
                draw(screen, self.space)
                clock.tick()

    def get_sonar_readings(self, x, y, angle):
        readings = []
        """
        Instead of using a grid of boolean(ish) sensors, sonar readings
        simply return N "distance" readings, one for each sonar
        we're simulating. The distance is a count of the first non-zero
        reading starting at the object. For instance, if the fifth sensor
        in a sonar "arm" is non-zero, then that arm returns a distance of 5.
        """
        # Make our arms.
        arm_left = self.make_sonar_arm(x, y)
        arm_middle = arm_left
        arm_right = arm_left

        # Rotate them and get readings.
        readings.append(self.get_arm_distance(arm_left, x, y, angle, 0.75))
        readings.append(self.get_arm_distance(arm_middle, x, y, angle, 0))
        readings.append(self.get_arm_distance(arm_right, x, y, angle, -0.75))

        if show_sensors:
            pygame.display.update()

        return readings

    def get_arm_distance(self, arm, x, y, angle, offset):
        # Used to count the distance.
        i = 0

        # Look at each point and see if we've hit something.
        for point in arm:
            i += 1

            # Move the point to the right spot.
            rotated_p = self.get_rotated_point(
                x, y, point[0], point[1], angle + offset
            )

            # Check if we've hit something. Return the current i (distance)
            # if we did.
            if rotated_p[0] <= 0 or rotated_p[1] <= 0 \
                    or rotated_p[0] >= width or rotated_p[1] >= height:
                return i  # Sensor is off the screen.
            else:
                obs = screen.get_at(rotated_p)
                if self.get_track_or_not(obs) != 0:
                    return i

            if show_sensors:
                pygame.draw.circle(screen, (255, 255, 255), (rotated_p), 2)

        # Return the distance for the arm.
        return i

    def make_sonar_arm(self, x, y):
        spread = 10  # Default spread.
        distance = 20  # Gap before first sensor.
        arm_points = []
        # Make an arm. We build it flat because we'll rotate it about the
        # center later.
        for i in range(1, 40):
            arm_points.append((distance + x + (spread * i), y))

        return arm_points

    def get_rotated_point(self, x_1, y_1, x_2, y_2, radians):
        # Rotate x_2, y_2 around x_1, y_1 by angle.
        x_change = (x_2 - x_1) * math.cos(radians) + \
            (y_2 - y_1) * math.sin(radians)
        y_change = (y_1 - y_2) * math.cos(radians) - \
            (x_1 - x_2) * math.sin(radians)
        new_x = x_change + x_1
        new_y = height - (y_change + y_1)
        return int(new_x), int(new_y)

    def get_track_or_not(self, reading):
        if reading == THECOLORS['black']:
            return 0
        else:
            return 1

if __name__ == "__main__":
    game_state = GameState()
    while True:
        game_state.frame_step((random.randint(0, 2)), (random.randint(0, 2)))
