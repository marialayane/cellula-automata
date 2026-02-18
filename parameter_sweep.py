import numpy as np
import matplotlib.pyplot as plt

def compute_flow(L=200, density=0.2, v_max=5, p=0.3, steps=500, transient=200):
    """
    Retorna o fluxo médio (veículos por unidade de tempo) após o período transiente.
    """
    road = np.random.choice([-1, 0], size=L, p=[1-density, density])
    flow_count = 0
    for t in range(steps):
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
            dist -= 1
            speeds[i] = min(speeds[i], dist)
        
        mask = np.random.random(len(vehicles)) < p
        speeds[mask] = np.maximum(speeds[mask] - 1, 0)
        
        road[vehicles] = -1
        new_positions = (vehicles + speeds) % L
        order = np.argsort(new_positions)
        new_positions = new_positions[order]
        speeds = speeds[order]
        road[new_positions] = speeds
        
        if t >= transient:
            mean_speed = np.mean(speeds) if len(speeds) > 0 else 0
            flow_count += density * mean_speed
    
    return flow_count / (steps - transient)

def parameter_sweep():
    L = 200
    steps = 500
    transient = 200
    v_max = 5
    densities = np.linspace(0.05, 0.6, 20)
    p_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    results = {p: [] for p in p_values}
    
    for p in p_values:
        print(f"Processando p = {p}")
        for rho in densities:
            flow = compute_flow(L=L, density=rho, v_max=v_max, p=p, steps=steps, transient=transient)
            results[p].append(flow)
    
    plt.figure(figsize=(8, 6))
    for p in p_values:
        plt.plot(densities, results[p], marker='o', linestyle='-', label=f'p = {p}')
    plt.xlabel('Densidade (ρ)')
    plt.ylabel('Fluxo médio (veículos/passo)')
    plt.title('Fluxo vs. Densidade para diferentes probabilidades de desaceleração')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fluxo_vs_densidade.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    parameter_sweep()