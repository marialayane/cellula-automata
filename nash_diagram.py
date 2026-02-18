import numpy as np
import matplotlib.pyplot as plt

def nash_simulation(L=200, density=0.2, v_max=5, p=0.3, steps=500):
    """
    Simula o modelo de Nagel-Schreckenberg e retorna um array 2D (tempo x espaço)
    com 1 onde há veículo e 0 onde está vazio.
    
    Parâmetros:
        L (int): Número de células na rodovia.
        density (float): Fração inicial de células ocupadas.
        v_max (int): Velocidade máxima.
        p (float): Probabilidade de desaceleração aleatória.
        steps (int): Número de passos de tempo.
    
    Retorna:
        np.ndarray: Matriz binária de shape (steps, L).
    """
    road = np.random.choice([-1, 0], size=L, p=[1-density, density])
    spacetime = np.zeros((steps, L), dtype=int)
    spacetime[0] = (road >= 0).astype(int)
    
    for t in range(1, steps):
        vehicles = np.where(road >= 0)[0]
        speeds = road[vehicles].copy()
        
        speeds = np.minimum(speeds + 1, v_max)
        
        for i, pos in enumerate(vehicles):
            next_idx = (i + 1) % len(vehicles)
            next_pos = vehicles[next_idx]
            if next_pos <= pos:
                dist = (next_pos + L) - pos
            else:
                dist = next_pos - pos
            speeds[i] = min(speeds[i], dist)
        
        mask = np.random.random(len(vehicles)) < p
        speeds[mask] = np.maximum(speeds[mask] - 1, 0)
        
        road[vehicles] = -1  
        new_positions = (vehicles + speeds) % L
        order = np.argsort(new_positions)
        new_positions = new_positions[order]
        speeds = speeds[order]
        road[new_positions] = speeds
        
        spacetime[t] = (road >= 0).astype(int)
    
    return spacetime

def plot_diagram(spacetime, title=None):
    """Exibe o diagrama espaço-temporal."""
    plt.figure(figsize=(10, 8))
    plt.imshow(spacetime, cmap='binary', aspect='auto', interpolation='nearest')
    plt.xlabel('Posição na rodovia')
    plt.ylabel('Tempo')
    if title:
        plt.title(title)
    plt.colorbar(label='Ocupação (1 = veículo)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    L = 200
    steps = 300
    
    densities = [0.1, 0.2, 0.4]
    for i, rho in enumerate(densities):
        st = nash_simulation(L=L, density=rho, v_max=5, p=0.3, steps=steps)
        plot_diagram(st, title=f'Diagrama espaço-temporal (ρ = {rho})')