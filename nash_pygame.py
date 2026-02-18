import pygame
import numpy as np

WIDTH, HEIGHT = 1000, 400
CELL_SIZE = 8
L = WIDTH // CELL_SIZE   
LANES = 1                

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
GRAY = (128, 128, 128)

class TrafficSimulation:
    def __init__(self, density=0.2, v_max=5, p=0.3):
        self.L = L
        self.v_max = v_max
        self.p = p
        self.density = density
        self.reset()
        
    def reset(self):
        """Reinicia a simulação com nova configuração aleatória."""
        self.road = np.random.choice([-1, 0], size=self.L, p=[1-self.density, self.density])
        self.step_count = 0
        
    def update(self):
        """Avança um passo temporal."""
        vehicles = np.where(self.road >= 0)[0]
        speeds = self.road[vehicles].copy()
        
        speeds = np.minimum(speeds + 1, self.v_max)
        
        for i, pos in enumerate(vehicles):
            next_idx = (i + 1) % len(vehicles)
            next_pos = vehicles[next_idx]
            if next_pos <= pos:
                dist = (next_pos + self.L) - pos
            else:
                dist = next_pos - pos
            dist -= 1
            speeds[i] = min(speeds[i], dist)
        
        mask = np.random.random(len(vehicles)) < self.p
        speeds[mask] = np.maximum(speeds[mask] - 1, 0)
        
        self.road[vehicles] = -1
        new_positions = (vehicles + speeds) % self.L
        order = np.argsort(new_positions)
        new_positions = new_positions[order]
        speeds = speeds[order]
        self.road[new_positions] = speeds
        
        self.step_count += 1
    
    def draw(self, screen):
        """Desenha a rodovia e os veículos na tela."""
        screen.fill(BLACK)
        road_y = HEIGHT // 2
        pygame.draw.line(screen, GRAY, (0, road_y), (WIDTH, road_y), 2)
        
        for i in range(self.L):
            if self.road[i] >= 0:
                speed = self.road[i]
                if speed >= self.v_max * 0.7:
                    color = GREEN
                elif speed >= self.v_max * 0.3:
                    color = YELLOW
                else:
                    color = RED
                x = i * CELL_SIZE
                y = road_y - CELL_SIZE // 2
                pygame.draw.rect(screen, color, (x, y, CELL_SIZE-1, CELL_SIZE-1))
        
        font = pygame.font.Font(None, 24)
        info = f"Step: {self.step_count}  v_max: {self.v_max}  p: {self.p:.2f}  density: {self.density:.2f}"
        text = font.render(info, True, WHITE)
        screen.blit(text, (10, 10))
        
        text2 = font.render("Espaço: pausar | R: reiniciar | +/ -: densidade", True, WHITE)
        screen.blit(text2, (10, HEIGHT - 30))

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Simulação de Tráfego - Modelo Nagel-Schreckenberg")
    clock = pygame.time.Clock()
    
    sim = TrafficSimulation(density=0.2, v_max=5, p=0.3)
    running = True
    paused = False
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    sim.reset()
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    sim.density = min(0.8, sim.density + 0.05)
                    sim.reset()
                elif event.key == pygame.K_MINUS:
                    sim.density = max(0.05, sim.density - 0.05)
                    sim.reset()
        
        if not paused:
            sim.update()
        
        sim.draw(screen)
        pygame.display.flip()
        clock.tick(30)
    
    pygame.quit()

if __name__ == "__main__":
    main()