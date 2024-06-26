import pygame
import os

BIRD_IMGS = [
   pygame.transform.scale2x(pygame.image.load(os.path.join('images', 'bird1.png'))),
   pygame.transform.scale2x(pygame.image.load(os.path.join('images', 'bird2.png'))),
   pygame.transform.scale2x(pygame.image.load(os.path.join('images', 'bird3.png')))
]

class Bird:
   IMGS = BIRD_IMGS
   MAX_ROTATION = 25
   ROT_VEL = 15
   ANIMATION_TIME = 5
   VEL = 1
   
   def __init__(self, x, y):
      self.x = x
      self.y = y
      self.tilt = 0
      self.tick_count = 0
      self.vel = 0
      self.height = self.y
      self.img_count = 0
      self.img = self.IMGS[0]
      
   def jump(self):
      self.vel = -10.5
      self.tick_count = 0
      self.height = self.y
      
   def move(self):
      self.tick_count += 1
      d = self.vel*self.tick_count + 1.5*self.tick_count**2
      
      if d >= 16:
         d = 16
         
      if d < 0:
         d -= 2
         
      self.y += d
      
      if d < 0 or self.y < self.height + 50:
         if self.tilt < self.MAX_ROTATION:
            self.tilt = self.MAX_ROTATION
      else:
         if self.tilt > -45:
            self.tilt -= self.ROT_VEL
            
   def draw(self, win):
      self.img_count += 1
      self.img_count %= 15
      if self.ANIMATION_TIME*3 <= self.img_count < self.ANIMATION_TIME * 4:
         self.img = self.IMGS[1]
      else:   
         self.img = self.IMGS[int(self.img_count/5)]
         
      if self.tilt <= -40:
         self.img = self.IMGS[1]
         self.img_count = self.ANIMATION_TIME*2
         
      rotated_image = pygame.transform.rotate(self.img, self.tilt)
      new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft = (self.x, self.y)).center)
      win.blit(rotated_image, new_rect.topleft)
      
   def get_mask(self):
      return pygame.mask.from_surface(self.img)