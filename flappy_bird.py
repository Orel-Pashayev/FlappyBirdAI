import pygame
import neat
import os
from bird import Bird
from pipe import Pipe
from base import Base

WIN_WIDTH = 576
WIN_HEIGHT = 900
BASE_HEIGHT = 775
BIRD_X, BIRD_SPAWN_Y = round(WIN_WIDTH/4), round(WIN_HEIGHT/3)

GEN = 0

BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join('images', 'bg.png')))

pygame.font.init()
STAT_FONT = pygame.font.SysFont('comicsans', 50)


def draw_window(window, birds, pipes, base, score):
   window.blit(BG_IMG, (0, 0))
   #bird.draw(window)
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
            quit()
         elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and not spacebar_pressed:
               spacebar_pressed = True
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
         
         output = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))
         
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
   

def run(config_path):
   config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                               neat.DefaultSpeciesSet, neat.DefaultStagnation,
                               config_path)
   
   p = neat.Population(config)
   
   p.add_reporter(neat.StdOutReporter(True))
   p.add_reporter(neat.StatisticsReporter())
   
   winner = p.run(main, 50)
   

if __name__ == '__main__':
   local_dir = os.path.dirname(__file__)
   config_path = os.path.join(local_dir, 'config-neat.txt')
   run(config_path)