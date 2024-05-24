import pygame
import neat
import os
import sys
from bird import Bird
from pipe import Pipe
from base import Base

WIN_WIDTH = 576
WIN_HEIGHT = 900
BASE_HEIGHT = 775
BIRD_X, BIRD_SPAWN_Y = round(WIN_WIDTH / 4), round(WIN_HEIGHT / 3)

GEN = 0

BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join('images', 'bg.png')))

pygame.font.init()
STAT_FONT = pygame.font.SysFont('comicsans', 50)
INPUT_FONT = pygame.font.SysFont('comicsans', 30)


def draw_window(window, birds, pipes, base, score):
    window.blit(BG_IMG, (0, 0))
    for bird in birds:
        bird.draw(window)
    score_text = STAT_FONT.render('Score: ' + str(score), 1, (255, 255, 255))
    window.blit(score_text, (WIN_WIDTH - 10 - score_text.get_width(), 10))

    gen_text = STAT_FONT.render('Gen: ' + str(GEN), 1, (255, 255, 255))
    window.blit(gen_text, (10, 10))

    for pipe in pipes:
        pipe.draw(window)

    base.draw(window)

    pygame.display.update()


def main(genomes_para, config):
    global GEN
    GEN += 1
    nets = []
    genomes = []
    birds = []

    for _, genome in genomes_para:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(BIRD_X, BIRD_SPAWN_Y))
        genome.fitness = 0
        genomes.append(genome)

    pipes = [Pipe(WIN_WIDTH)]
    base = Base(BASE_HEIGHT)
    window = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()

    score = 0

    run = True
    while run:
        spacebar_pressed = False
        clock.tick(45)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not spacebar_pressed:
                    spacebar_pressed = True
                    for bird in birds:
                        bird.jump()

        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and BIRD_X > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1
        else:
            break

        for x, bird in enumerate(birds):
            bird.move()
            genomes[x].fitness += 0.1

            output = nets[x].activate(
                (bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))

            if output[0] > 0.5:
                bird.jump()

        add_pipe = False
        rem = []
        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    genomes[x].fitness -= 2
                    birds.pop(x)
                    nets.pop(x)
                    genomes.pop(x)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)
            pipe.move()

        if add_pipe:
            for g in genomes:
                g.fitness += 5
            score += 1
            pipes.append(Pipe(WIN_WIDTH))

        for r in rem:
            pipes.remove(r)

        for x, bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= BASE_HEIGHT or bird.y < 0:
                genomes[x].fitness -= 5
                birds.pop(x)
                nets.pop(x)
                genomes.pop(x)

        base.move()
        draw_window(window, birds, pipes, base, score)

        if score > 50:
            break


def run(config_path, pop_size):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    p = neat.Population(config)

    # Update population size
    p.config.pop_size = pop_size

    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())

    winner = p.run(main, 50)


def main_menu():
    window = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    input_box = pygame.Rect(WIN_WIDTH // 2 - 100, WIN_HEIGHT // 2 - 50, 200, 50)
    color_inactive = pygame.Color('lightskyblue3')
    color_active = pygame.Color('dodgerblue2')
    color = color_inactive
    active = False
    text = '50'  # Default value
    start_button = pygame.Rect(WIN_WIDTH // 2 - 50, WIN_HEIGHT // 2 + 20, 100, 50)
    button_color = (0, 255, 0)
    button_text_color = (0, 0, 0)

    base = Base(BASE_HEIGHT)
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if start_button.collidepoint(event.pos) and text.isdigit():
                    run(config_path, int(text))
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and text.isdigit():
                    run(config_path, int(text))
                elif event.key == pygame.K_BACKSPACE:
                    text = text[:-1]
                elif event.unicode.isdigit():
                    text += event.unicode

        window.blit(BG_IMG, (0, 0))
        base.move()
        base.draw(window)

        txt_surface = INPUT_FONT.render(text, True, color)
        width = max(200, txt_surface.get_width() + 10)
        input_box.w = width
        window.blit(txt_surface, (input_box.x + 5, input_box.y + 5))
        pygame.draw.rect(window, color, input_box, 2)

        pygame.draw.rect(window, button_color, start_button)
        start_text = INPUT_FONT.render('Start', True, button_text_color)
        window.blit(start_text, (start_button.x + 25, start_button.y + 10))

        pygame.display.flip()
        clock.tick(30)


config_path = os.path.join(os.path.dirname(__file__), 'config-neat.txt')

if __name__ == '__main__':
    main_menu()
