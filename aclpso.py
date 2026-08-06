import numpy as np
import random
#AECLPSO实现
class AECLPSO:
    def __init__(self, pop_size=20, max_iter=30, w_max=0.9, w_min=0.4, c_max=2.5, c_min=1.5):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.w_max = w_max
        self.w_min = w_min
        self.c_max = c_max
        self.c_min = c_min
        self.diversity_history = []
        self.success_history = []
        self.stagnation_count = np.zeros(pop_size)
        self.improvement_rate = np.zeros(pop_size)
        self.stagnation_threshold = 5
        self.base_learning_probs = self._calculate_learning_probabilities()
        self.learning_probs = self.base_learning_probs.copy()
        self.elite_archive = []
        self.archive_size = 10
        self.convergence_state = "exploring"
        self.topology = "global"
        
    def _calculate_learning_probabilities(self):
        probs = []
        for i in range(self.pop_size):
            pc = 0.05 + 0.45 * (np.exp(10 * i / (self.pop_size - 1)) - 1) / (np.exp(10) - 1)
            probs.append(pc)
        return np.array(probs)
    def _adaptive_learning_probability(self, particle_idx, iteration):
        base_pc = self.base_learning_probs[particle_idx]
        if self.improvement_rate[particle_idx] < 0.1:  
            pc = min(0.9, base_pc * 1.5)
        elif self.improvement_rate[particle_idx] > 0.5: 
            pc = max(0.1, base_pc * 0.7)
        else:
            pc = base_pc
        progress = iteration / self.max_iter
        if progress > 0.8:  
            pc = min(0.9, pc * 1.2)
        if self.stagnation_count[particle_idx] > self.stagnation_threshold:
            pc = min(0.95, pc * 1.3)
        return pc
    def _calculate_diversity(self, population):
        if len(population) == 0:
            return 0.0
        positions = np.array([ind[:] for ind in population])
        mean_pos = np.mean(positions, axis=0)
        diversity = np.mean([np.sqrt(np.sum((pos - mean_pos)**2)) for pos in positions])
        return diversity
    def _determine_convergence_state(self, diversity_ratio, success_rate, iteration):
        progress = iteration / self.max_iter
        
        if progress < 0.3:
            return "exploring"
        elif diversity_ratio > 0.5 and success_rate < 0.3:
            return "exploring"
        elif diversity_ratio < 0.2 or success_rate > 0.6:
            return "exploiting"
        else:
            return "converging"
    def _adaptive_weight_coefficients(self, convergence_state):
        if convergence_state == "exploring":
            return {"iter": 0.2, "diversity": 0.5, "success": 0.3}
        elif convergence_state == "converging":
            return {"iter": 0.4, "diversity": 0.3, "success": 0.3}
        else:  # exploiting
            return {"iter": 0.5, "diversity": 0.2, "success": 0.3}
    
    def _adaptive_inertia_weight(self, iteration, diversity_ratio, success_rate):
        coeffs = self._adaptive_weight_coefficients(self.convergence_state)
        w_iter = self.w_max - (self.w_max - self.w_min) * iteration / self.max_iter
        w_diversity = self.w_min + (self.w_max - self.w_min) * diversity_ratio
        w_success = self.w_min + (self.w_max - self.w_min) * (1 - success_rate)
        w = (coeffs["iter"] * w_iter + 
             coeffs["diversity"] * w_diversity + 
             coeffs["success"] * w_success)
        
        return np.clip(w, self.w_min, self.w_max)
    
    def _adaptive_learning_coefficients(self, diversity_ratio, success_rate):
        c = self.c_min + (self.c_max - self.c_min) * diversity_ratio
        if success_rate < 0.2:
            c *= 1.1
        elif success_rate > 0.8:
            c *= 0.9
            
        return c
    
    def _update_elite_archive(self, population):
        sorted_pop = sorted(population, key=lambda x: x.fitness.values[0])
        self.elite_archive.extend(sorted_pop[:2])
        
        if len(self.elite_archive) > self.archive_size:
            self.elite_archive = self._diversity_selection(
                self.elite_archive, self.archive_size
            )
    
    def _diversity_selection(self, archive, target_size):
        if len(archive) <= target_size:
            return archive
        
        sorted_archive = sorted(archive, key=lambda x: x.fitness.values[0])
        selected = [sorted_archive[0]]
        candidates = sorted_archive[1:]
        
        while len(selected) < target_size and candidates:
            max_min_dist = -1
            best_candidate = None
            
            for candidate in candidates:
                min_dist = float('inf')
                for selected_ind in selected:
                    dist = np.sqrt(np.sum((np.array(candidate[:]) - np.array(selected_ind[:]))**2))
                    min_dist = min(min_dist, dist)
                
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                candidates.remove(best_candidate)
        
        return selected
    
    def _select_exemplar(self, particle_idx, dimension, pbest_list, learning_prob):
        if random.random() < learning_prob:
            return pbest_list[particle_idx][dimension]
        else:
            if random.random() < 0.2 and self.elite_archive:
                elite = random.choice(self.elite_archive)
                return elite[dimension]
            else:
                if self.convergence_state == "exploring":
                    tournament_size = max(2, int(0.15 * self.pop_size))
                else:
                    tournament_size = max(2, int(0.1 * self.pop_size))
                
                candidates = random.sample(range(len(pbest_list)), tournament_size)
                best_candidate = min(candidates, key=lambda x: pbest_list[x].fitness.values[0])
                return pbest_list[best_candidate][dimension]
    
    def _handle_stagnation(self, particle_idx, particle, velocity):
        if self.stagnation_count[particle_idx] > self.stagnation_threshold:
            strategy = random.random()
            
            if strategy < 0.3:
                for d in range(len(velocity)):
                    velocity[d] = random.uniform(-1, 1) * (particle[d] * 0.1)
            elif strategy < 0.6:
                for d in range(len(particle)):
                    if random.random() < 0.3: 
                        perturbation = np.random.normal(0, 0.1) * (particle[d] + 1e-10)
                        particle[d] += perturbation
            else:
                if self.elite_archive:
                    elite = random.choice(self.elite_archive)
                    for d in range(len(particle)):
                        if random.random() < 0.5:
                            particle[d] = 0.5 * (particle[d] + elite[d])
            
            self.stagnation_count[particle_idx] = 0
            
        return particle, velocity
    
    def _adaptive_topology(self, iteration, diversity):
        progress = iteration / self.max_iter
        
        if diversity > 0.5:
            self.topology = "global"
        elif diversity < 0.2:
            self.topology = "ring"
        else:
            self.topology = "dynamic"