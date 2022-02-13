#Snake Tutorial Python

import math
import random
import pygame
import tkinter as tk
from tkinter import messagebox

class cube(object):
    rows = 20
    width = 550
    height = 720
    def __init__(self,start,dirnx=1,dirny=0,color=(255,0,0)):
        self.pos = start
        self.dirnx = 1
        self.dirny = 0
        self.color = color

        
    def move(self, dirnx, dirny):
        self.dirnx = dirnx
        self.dirny = dirny
        self.pos = (self.pos[0] + self.dirnx, self.pos[1] + self.dirny)

    def draw(self, surface, eyes=False):
        dis = self.width // self.rows
        i = self.pos[0]
        j = self.pos[1]

        pygame.draw.rect(surface, self.color, (i*dis+1,j*dis+1, dis-2, dis-2))
        if eyes:
            centre = dis//2
            radius = 3
            circleMiddle = (i*dis+centre-radius,j*dis+8)
            circleMiddle2 = (i*dis + dis -radius*2, j*dis+8)
            pygame.draw.circle(surface, (0,0,0), circleMiddle, radius)
            pygame.draw.circle(surface, (0,0,0), circleMiddle2, radius)

def drawGrid(w, h, rows, surface):
    #sizeBtwn = w // rows
    sizeBtwn = h // rows
    x = 0
    y = 0
    for l in range(rows):
        x = x + sizeBtwn
        y = y + sizeBtwn

        pygame.draw.line(surface, (255,255,255), (x,0),(x,h))
        pygame.draw.line(surface, (255,255,255), (0,y),(w,y))
        
def main():
    pygame.init()
    width = 550
    height = 720
    rows = 20
    screen = pygame.display.set_mode((width, height))
    background = pygame.image.load('field.png')
    running = True
    while running:
        #screen.fill((255,0,0))
        screen.blit(background, (0,0))
        drawGrid(width, height, rows, screen)
        c = cube((15,15), color=(0,255,0))
        c.draw(screen)
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

main()