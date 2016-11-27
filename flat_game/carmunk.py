import random
import math
import numpy as np

import pygame
from pygame.color import THECOLORS

import pymunk
from pymunk.vec2d import Vec2d
from pymunk.pygame_util import draw

# Showing sensors and redrawing slows things down.
show_sensors = True
draw_screen = True
use_obstacles = True
# Whether red team agent should be added
use_red_team = True
# Whether red team agent is random or trained
trained_red_team = True
# Senor parameters
num_arms = 6
num_arm_points = 40
sensor_spread = 10  # Distance between consecutive sensors on a arm
first_sensor_gap = 35  # Gap before first sensor.
arm_angle = math.pi/3

# Colors for obstacle detections
wall_color = 'purple'
car_color = 'green'
car_crash_color = 'yellow'
cat_color = 'red'
cat_crash_color = 'orange'
obstacle_color = 'blue'
car_collision_color_set = set([wall_color, cat_color, obstacle_color])
cat_collision_color_set = set([wall_color, obstacle_color])

# Rewards and penalties
car_crash_penalty = -100
cat_crash_penalty = -100
cat_success_reward = 500

# PyGame init
width = 1000
height = 700
frame_step = 0.06
pygame.init()
screen = pygame.display.set_mode((width, height))
if not draw_screen:
    pygame.display.iconify()
clock = pygame.time.Clock()
# Turn off alpha since we don't use it.
screen.set_alpha(None)

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
            s.color = THECOLORS[wall_color]
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
        c_shape.color = THECOLORS[obstacle_color]
        self.space.add(c_body, c_shape)
        return c_body

    def create_cat(self):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.cat_body = pymunk.Body(1, inertia)
        self.cat_body.position = 50, height - 100
        self.cat_shape = pymunk.Circle(self.cat_body, 25)
        self.cat_shape.color = THECOLORS[cat_color]
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
        self.car_shape.color = THECOLORS[car_color]
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
        self.space.step(frame_step)
        if draw_screen:
            pygame.display.flip()
            # Required after update to MacOS Sierra
            pygame.event.clear()
        clock.tick()

        # Get the current cat location and the readings there.
        cat_state, car_reward = None, None
        if use_red_team and trained_red_team:
            cat_x, cat_y = self.cat_body.position
            cat_readings = self.get_sonar_readings(cat_x, cat_y, self.cat_body.angle,
                                                   [cat_collision_color_set, set([car_color])])
            cat_state = np.array([cat_readings])
            cat_reward = self.get_cat_reward(cat_readings, cat_driving_direction)

        # Get the current car location and the readings there.
        car_x, car_y = self.car_body.position
        car_readings = self.get_sonar_readings(car_x, car_y, self.car_body.angle,
                                               [car_collision_color_set])
        car_state = np.array([car_readings])
        car_reward = self.get_car_reward(car_readings, car_driving_direction)

        self.num_steps += 1
        return car_state, car_reward, cat_state, cat_reward

    def get_car_reward(self, car_readings, car_driving_direction):
        if self.is_crashed(car_readings):
            self.car_crashed = True
            reward = car_crash_penalty
            self.recover_car_crash(car_driving_direction)
        else:
            # Higher readings are better, so return the sum.
            reward = -5 + int(sum(car_readings) / 10)
        return reward

    def get_cat_reward(self, cat_readings, cat_driving_direction):
        if self.is_cat_successful(cat_readings):
            self.cat_crashed = True
            self.recover_cat_crash(cat_driving_direction)
            reward = cat_success_reward
        if self.is_crashed(cat_readings):
            self.cat_crashed = True
            reward = cat_crash_penalty
            self.recover_cat_crash(cat_driving_direction)
        else:
            # Higher readings are better, so return the sum.
            reward = -5 + int(sum(cat_readings) / 10)
        return reward

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
        else:
            # Don't modify angle
            pass
        driving_direction = Vec2d(1, 0).rotated(body.angle)
        body.velocity = 100 * driving_direction
        return driving_direction

    # This assumes that first num_arm readings is for collision.
    def is_crashed(self, readings):
        for arm_index in range(num_arms):
            if readings[arm_index] == 1:
                print('Crash detected: ', readings)
                return True
        return False

    # This assumes that first num_arm readings is for collision.
    def is_cat_successful(self, readings):
        for arm_index in range(num_arms):
            if readings[num_arms + arm_index] == 1:
                print('Cat collided with car: ', readings)
                return True
        return False

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
                screen.fill(THECOLORS[car_crash_color])
                draw(screen, self.space)
                if use_red_team:
                    # Set velocity of other agents to 0 before retracting
                    cat_velocity = self.cat_body.velocity
                    self.cat_body.velocity = pymunk.Vec2d(0., 0.)
                self.space.step(frame_step)
                if draw_screen:
                    pygame.display.flip()
                    # Required after update to MacOS Sierra
                    pygame.event.clear()
                clock.tick()

                if use_red_team:
                    # Fill the screen black to avoid false detection by other agents
                    screen.fill(THECOLORS["black"])
                    self.cat_body.velocity = cat_velocity
                    # draw(screen, self.space)

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
                screen.fill(THECOLORS[cat_crash_color])
                draw(screen, self.space)
                # Set velocity of other agents to 0 before retracting
                car_velocity = self.car_body.velocity
                self.car_body.velocity = pymunk.Vec2d(0., 0.)
                self.space.step(frame_step)
                if draw_screen:
                    pygame.display.flip()
                    # Required after update to MacOS Sierra
                    pygame.event.clear()
                clock.tick()
                # Fill the screen black to avoid false detection by other agents
                screen.fill(THECOLORS["black"])
                self.car_body.velocity = car_velocity
                # draw(screen, self.space)

    def get_sonar_readings(self, x, y, angle, color_sets):
        readings = []
        """
        Instead of using a grid of boolean(ish) sensors, sonar readings
        simply return N "distance" readings, one for each sonar
        we're simulating. The distance is a count of the first non-zero
        reading starting at the object. For instance, if the fifth sensor
        in a sonar "arm" is non-zero, then that arm returns a distance of 5.
        """
        # Make sensor arm.
        sensor_arm = self.make_sonar_arm(x, y)
        # Get reading for passed colors.
        for color_set in color_sets:
            # Rotate arm in all directions and get readings.
            for arm_index in range(num_arms):
                readings.append(self.get_arm_distance(
                    sensor_arm, x, y, angle, arm_index * arm_angle, color_set))

        if draw_screen and show_sensors:
            pygame.display.update()
            # Required after update to MacOS Sierra
            pygame.event.clear()

        return readings

    def get_arm_distance(self, arm, x, y, angle, offset, color_set):
        """ If color is set get distance for obstacles of that color
        Else return distance for any obstacles (i.e. 1st non black pixel).
        """
        # Used to count the distance.
        i = 0

        # Look at each point and see if we've hit something.
        for point in arm:
            i += 1

            # Move the point to the right spot.
            rotated_p = self.get_rotated_point(
                x, y, point[0], point[1], angle + offset
            )

            # Check if sensor is off the screen
            if rotated_p[0] <= 0 or rotated_p[1] <= 0 \
                    or rotated_p[0] >= width or rotated_p[1] >= height:
                if wall_color in color_set:
                    return i
                else:
                    return num_arm_points

            # Check for colors in color set
            obs = screen.get_at(rotated_p)
            for color in color_set:
                if obs == THECOLORS[color]:
                    return i

            if show_sensors:
                pygame.draw.circle(screen, (255, 255, 255), (rotated_p), 2)

        # Return the distance for the arm.
        return i

    def make_sonar_arm(self, x, y):
        arm_points = []
        # Make an arm. We build it flat because we'll rotate it about the
        # center later.
        for i in range(0, num_arm_points):
            arm_points.append((x + first_sensor_gap + (sensor_spread * i), y))

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

if __name__ == "__main__":
    game_state = GameState()
    i = 0
    while True:
        i += 1
        game_state.frame_step((random.randint(0, 2)), (random.randint(0, 2)))
