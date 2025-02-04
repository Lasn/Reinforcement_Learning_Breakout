import pygame
import random
import numpy as np
from PIL import Image


class Brick(pygame.sprite.Sprite):
    def __init__(self, color, x, y, width, height):
        super().__init__()
        self.image = pygame.Surface((width, height))
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y


class Ball(pygame.sprite.Sprite):
    def __init__(self, x, y, size):
        super().__init__()
        self.image = pygame.Surface((size, size))
        self.image.fill((255, 255, 255))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y


class Paddle(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height):
        super().__init__()
        self.image = pygame.Surface((width, height))
        self.image.fill((255, 255, 255))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y


class BreakoutGame:
    def __init__(self):
        pygame.init()
        self.screen_width = 800
        self.screen_height = 600
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.red = (255, 0, 0)
        self.green = (0, 255, 0)
        self.blue = (0, 0, 255)
        self.yellow = (255, 255, 0)
        self.orange = (255, 165, 0)
        self.colors = [
            self.red,
            self.orange,
            self.yellow,
            self.green,
            self.blue,
            self.white,
        ]
        self.paddle_width = 100
        self.paddle_height = 10
        self.paddle_speed = 6
        self.ball_size = 10
        self.ball_speed_x = 4
        self.ball_speed_y = -4
        self.brick_rows = 6
        self.brick_cols = 10
        self.brick_width = (self.screen_width - 20) // self.brick_cols
        self.brick_height = 20
        self.brick_padding = 2
        self.brick_offset_top = 50
        self.lives = 5
        self.score = 0
        self.font = pygame.font.Font(None, 36)
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Breakout")
        self.clock = pygame.time.Clock()
        self.draw_with_score = False
        self.running = True
        self.all_sprites = pygame.sprite.Group()
        self.bricks = pygame.sprite.Group()
        self.create_objects()
        self.draw_game()

    def create_objects(self):
        self.paddle = Paddle(
            (self.screen_width - self.paddle_width) // 2,
            self.screen_height - self.paddle_height - 10,
            self.paddle_width,
            self.paddle_height,
        )
        self.ball = Ball(
            self.screen_width // 2 - self.ball_size // 2,
            self.screen_height // 2 - self.ball_size // 2,
            self.ball_size,
        )
        self.all_sprites.add(self.paddle)
        self.all_sprites.add(self.ball)

        color_index = 0  # Index to track colors from self.colors list
        for row in range(self.brick_rows):
            color = self.colors[color_index]
            for col in range(self.brick_cols):
                brick = Brick(
                    color,
                    10 + col * self.brick_width,
                    self.brick_offset_top + row * self.brick_height,
                    self.brick_width - self.brick_padding,
                    self.brick_height - self.brick_padding,
                )
                self.bricks.add(brick)
                self.all_sprites.add(brick)
            color_index = (color_index + 1) % len(self.colors)

    def update_paddle(self, action=None):
        if action is not None:
            if action == 0:
                self.paddle.rect.x -= self.paddle_speed
            if action == 1:
                self.paddle.rect.x += self.paddle_speed
        else:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                self.paddle.rect.x -= self.paddle_speed
            if keys[pygame.K_RIGHT]:
                self.paddle.rect.x += self.paddle_speed
        if self.paddle.rect.x < 0:
            self.paddle.rect.x = 0
        if self.paddle.rect.x > self.screen_width - self.paddle_width:
            self.paddle.rect.x = self.screen_width - self.paddle_width

    def reset(self):
        self.lives = 5
        self.score = 0
        self.ball.rect.x = self.screen_width // 2 - self.ball_size // 2
        self.ball.rect.y = self.screen_height // 2 - self.ball_size // 2
        self.ball_speed_x = 4
        self.ball_speed_y = -4
        self.all_sprites.empty()
        self.bricks.empty()
        self.create_objects()
        self.draw_game()
        return self.get_state()

    def get_state(self):
        screen = pygame.surfarray.array3d(pygame.display.get_surface())
        screen = Image.fromarray(screen).convert("L")
        screen = screen.resize((84, 84))
        screen = screen.transpose(Image.FLIP_LEFT_RIGHT)
        screen = screen.rotate(90)
        # screen.show()
        return np.array(screen)

    def update_ball(self):
        paddle_collision = False
        live_lost = False
        self.ball.rect.x += self.ball_speed_x
        self.ball.rect.y += self.ball_speed_y
        if (
            self.ball.rect.x <= 0
            or self.ball.rect.x >= self.screen_width - self.ball_size
        ):
            self.ball_speed_x = -self.ball_speed_x
        if self.ball.rect.y <= 0:
            self.ball_speed_y = -self.ball_speed_y
        if self.ball.rect.colliderect(self.paddle.rect):
            self.ball.rect.y = self.paddle.rect.y - self.ball_size
            self.ball_speed_y = -self.ball_speed_y
            paddle_collision = True
        if self.ball.rect.y > self.screen_height:
            self.lives -= 1
            self.ball.rect.x = self.screen_width // 2 - self.ball_size // 2
            self.ball.rect.y = self.screen_height // 2 - self.ball_size // 2
            self.ball_speed_y = -self.ball_speed_y
            live_lost = True
        return paddle_collision, live_lost

    def draw_game(self):
        self.screen.fill(self.black)
        self.all_sprites.draw(self.screen)
        if self.draw_with_score:
            self.draw_score_lives()
        pygame.display.flip()

    def step(self, action, draw_game=True, normal_speed=False):
        reward = 0
        self.update_paddle(action)
        paddle_collision, live_lost = self.update_ball()
        if paddle_collision:
            reward += 1
        if live_lost:
            reward -= 10
        if draw_game:
            self.draw_game()

        brick_collisions = pygame.sprite.spritecollide(
            self.ball, self.bricks, dokill=True
        )
        if brick_collisions:
            self.ball_speed_y = -self.ball_speed_y
            self.score += 10 * len(brick_collisions)
            reward = 10

        done = self.lives <= 0
        if done:
            reward = -100

        if normal_speed:
            self.clock.tick(60)

        return self.get_state(), reward, done, self.score

    def run(self):
        self.draw_with_score = True
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            self.update_paddle()
            self.update_ball()
            self.draw_game()

            brick_collisions = pygame.sprite.spritecollide(
                self.ball, self.bricks, dokill=True
            )
            if brick_collisions:
                self.ball_speed_y = -self.ball_speed_y
                self.score += 10 * len(brick_collisions)

            if self.lives <= 0:
                self.running = False
            self.clock.tick(60)
        pygame.quit()

    def draw_score_lives(self):
        score_text = self.font.render(f"Score: {self.score}", True, self.white)
        lives_text = self.font.render(f"Lives: {self.lives}", True, self.white)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(lives_text, (self.screen_width - 100, 10))


if __name__ == "__main__":
    game = BreakoutGame()
    game.run()
