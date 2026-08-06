#**data processing**
import sys
sys.path.append('./model-code/code')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import BiTCN,TCN,KAN,LSTM,Informer,NBEATSx
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from loss import RGM
from aclpso import AECLPSO
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import uniform, randint, loguniform
from properscoring import crps_ensemble
from deap import base, creator, tools
import random
import math
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')
def load_and_preprocess_data():
    data = pd.read_csv("./data.csv", encoding='gbk')
    scaler = MinMaxScaler()
    columns_to_normalize = ["太阳总辐照度", "直接法向辐照度", "全球水平辐照度", "温度", "气压"]
    data.loc[:, columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])
    data[["月份", "季节","小时","日子"]] = data[["月份", "季节","小时","日子"]].astype(float)
    
    data_freq = '15min'
    start_time = pd.Timestamp("2023-01-01 00:00:00")
    continuous_timestamps = pd.date_range(start=start_time, periods=len(data), freq=data_freq)
    data["timestamp"] = continuous_timestamps
    df = pd.DataFrame({
        'ds': data["timestamp"],
        'unique_id': 'timeseries_1',
        'y': data["target"],
        'feature1': data["太阳总辐照度"],
        'feature2': data["直接法向辐照度"],
        'feature3': data["全球水平辐照度"],
        'feature4': data["温度"],
        'feature5': data["气压"],
        'feature6': data["小时"],
        'feature7': data["日子"],
        'feature8': data["月份"],
        'feature9': data["季节"]
    })
    return df
def calculate_quantile_score(y_true, quantile_predictions, quantile_levels):
    n_samples, n_quantiles = quantile_predictions.shape
    total_loss = 0.0
    for i, tau in enumerate(quantile_levels):
        y_pred_q = quantile_predictions[:, i]
        error = y_true - y_pred_q
        loss = np.where(error >= 0, tau * error, (tau - 1) * error)
        total_loss += np.mean(loss)
    return total_loss / n_quantiles

def calculate_crps_and_quantile_score(cv_df):
    y_true = cv_df['y'].values
    levels = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    quantile_predictions = []
    quantile_levels = []
    for level in levels:
        lo_col = f'BiTCN-lo-{level}'
        hi_col = f'BiTCN-hi-{level}'
        if lo_col in cv_df.columns:
            quantile_predictions.append(cv_df[lo_col].values)
            q_lo = (1.0 - level / 100.0) / 2.0
            quantile_levels.append(q_lo)
        if hi_col in cv_df.columns:
            quantile_predictions.append(cv_df[hi_col].values)
            q_hi = 1.0 - (1.0 - level / 100.0) / 2.0
            quantile_levels.append(q_hi)
    quantile_predictions = np.array(quantile_predictions).T 
    quantile_levels = np.array(quantile_levels)
    sorted_indices = np.argsort(quantile_levels)
    quantile_levels = quantile_levels[sorted_indices]
    quantile_predictions = quantile_predictions[:, sorted_indices]
    crps_scores = crps_ensemble(y_true, quantile_predictions)
    mean_crps = np.mean(crps_scores)

    quantile_score = calculate_quantile_score(y_true, quantile_predictions, quantile_levels)

    return mean_crps, quantile_score, crps_scores

#**train BITCN model**
def evaluate_BiTCN(individual, df_train_full, horizon, levels):
        params = decode_individual(individual)
        K = params['K']
        Weights = params['Weights']
        df_train_full = df_train_full.iloc[-1050:]#使用验证集
        model_params = {k: v for k, v in params.items() if k not in ['K', 'Weights']}
        BiTCN_model = BiTCN(
            h=horizon,
            loss=RGM(level=levels, n_components=K, num_samples=10000, return_params=True, weighted=True,posterior_regularization=True, posterior_weight=Weights),#, return_params=True, weighted=True,posterior_regularization=True, posterior_weight=0.1
            valid_loss=RGM(level=levels, n_components=K, num_samples=10000, return_params=True, weighted=True,posterior_regularization=True, posterior_weight=Weights),
            hist_exog_list=['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 
                           'feature6', 'feature7', 'feature8', 'feature9'],
            futr_exog_list=['feature6', 'feature7', 'feature8', 'feature9'],
            random_seed=42,
            devices=1,
            n_series=1,
            **model_params
        )
        
        nf = NeuralForecast(models=[BiTCN_model], freq='15min')
        cv_df = nf.cross_validation(
            df=df_train_full,
            n_windows=1050,  
            step_size=1,
            val_size=horizon
        )
        crps_score, quantile_score, individual_crps = calculate_crps_and_quantile_score(cv_df)
        y_true = cv_df['y'].values
        y_pred = cv_df['BiTCN'].values
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mse=mean_squared_error(y_true, y_pred)
        fitness = crps_score
        
        print(f"评估结果:")
        print(f"CRPS: {crps_score:.4f}")
        print(f"Quantile Score: {quantile_score:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"MSE为:{mse}")
        print(f"R2: {r2:.4f}")
        
        return (fitness,)


def train_final_model(best_params,df_train_full, horizon, levels):
    optimal_params = decode_individual(best_params)
    print(f"\n最优参数配置:")
    for key, value in optimal_params.items():
        print(f"  {key}: {value}")
    K = optimal_params['K']
    Weights = optimal_params['Weights']
    
    model_params = {k: v for k, v in optimal_params.items() if k not in ['K', 'Weights']}
    
    final_BITCN = BiTCN(
        h=horizon,
        loss=RGM(level=levels, n_components=K, num_samples=10000, return_params=True, weighted=True,posterior_regularization=True, posterior_weight=Weights),#, return_params=True, weighted=True,posterior_regularization=True, posterior_weight=0.1
        valid_loss=RGM(level=levels, n_components=K, num_samples=10000, return_params=True, weighted=True,posterior_regularization=True, posterior_weight=Weights),
        hist_exog_list=['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 
                           'feature6', 'feature7', 'feature8', 'feature9'],
        futr_exog_list=['feature6', 'feature7', 'feature8', 'feature9'],
        random_seed=42,
        devices=1,
        n_series=1,
        **model_params
    )
    nf_final = NeuralForecast(models=[final_BITCN], freq='15min')
    cv_df_final = nf_final.cross_validation(
        df=df_train_full,
        n_windows=1050,  
        step_size=1,
        val_size=horizon
    )
    final_crps, final_quantile, final_individual_crps = calculate_crps_and_quantile_score(cv_df_final)
    y_true_final = cv_df_final['y'].values
    y_pred_final = cv_df_final['BiTCN'].values
    final_mae = mean_absolute_error(y_true_final, y_pred_final)
    final_r2 = r2_score(y_true_final, y_pred_final)
    final_mse = mean_squared_error(y_true_final, y_pred_final)
    
    print(f"CRPS: {final_crps:.6f}")
    print(f"MAE: {final_mae:.6f}")
    print(f"MSE: {final_mse:.6f}")
    print(f"Quantile Score: {final_quantile:.6f}")
    print(f"R²: {final_r2:.6f}")
    
    return nf_final, cv_df_final, {
        'crps': final_crps,
        'mae': final_mae,
        'mse': final_mse,
        'Quantile Score': final_quantile,
        'r2': final_r2
    }
#**AECLPSO search**
def create_individual():
    individual = []
    
    individual.append(random.uniform(0, 11)) 
    
    individual.append(random.uniform(0, 4))
    
    individual.append(random.uniform(0, 0.5))
    
    individual.append(random.uniform(300, 1000))
    
    individual.append(random.uniform(np.log10(5e-5), np.log10(1e-3)))
    
    individual.append(random.uniform(0, 10))
    
    individual.append(random.uniform(50, 200))
    
    individual.append(random.uniform(0, 4)) 
    
    individual.append(random.uniform(64, 2048))
    
    individual.append(random.uniform(0, 4)) 
    
    individual.append(random.uniform(1, 3))
    
    individual.append(random.uniform(0, 1))
    
    individual.append(random.uniform(2, 8))
    
    individual.append(random.uniform(0.001, 0.2))
    
    return individual

def decode_individual(individual):
    params = {}
    input_size_map = [25,50,75,100,125,150,175, 200, 300, 350, 500, 600]
    hidden_size_map = [16,32, 64, 128, 256]
    batch_size_map = [16, 32, 64, 128, 256]
    inference_windows_batch_size_map = [64, 256, 512, 1024, 2048]
    
    idx = int(round(individual[0]))
    params['input_size'] = input_size_map[idx]

    idx = int(round(individual[1]))
    params['hidden_size'] = hidden_size_map[idx]

    params['dropout'] = individual[2]
    
    params['max_steps'] = int(round(individual[3]))
    
    params['learning_rate'] = 10 ** individual[4]
    
    params['num_lr_decays'] = int(round(individual[5]))
    
    params['val_check_steps'] = int(round(individual[6]))
    
    idx = int(round(individual[7]))
    params['batch_size'] = batch_size_map[idx]
    
    params['windows_batch_size'] = int(round(individual[8]))
    
    idx = int(round(individual[9]))
    params['inference_windows_batch_size'] = inference_windows_batch_size_map[idx]
    
    params['step_size'] = int(round(individual[10]))
    
    params['scaler_type'] = 'standard' if individual[11] < 0.5 else 'minmax'
    
    params['K'] = int(round(individual[12]))
    
    params['Weights'] = individual[13]
    
    print(f"  解码后参数: {params}")
    return params

def apply_constraints(individual):
    individual[0] = np.clip(individual[0], 0, 11)
    individual[1] = np.clip(individual[1], 0, 4)
    individual[2] = np.clip(individual[2], 0, 0.5)
    individual[3] = np.clip(individual[3], 300, 1000)
    individual[4] = np.clip(individual[4], np.log10(5e-5), np.log10(1e-3))
    individual[5] = np.clip(individual[5], 0, 10)
    individual[6] = np.clip(individual[6], 50, 200)
    individual[7] = np.clip(individual[7], 0, 4)
    individual[8] = np.clip(individual[8], 64, 2048)
    individual[9] = np.clip(individual[9], 0, 4)
    individual[10] = np.clip(individual[10], 1, 3)
    individual[11] = np.clip(individual[11], 0, 1)
    individual[12] = np.clip(individual[12], 2, 8)
    individual[13] = np.clip(individual[13], 0.001, 0.2)
    return individual
def aeclpso_optimize_enhanced(df_train_full, horizon, levels, pop_size=20, max_iter=30):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_BiTCN, df_train_full=df_train_full, 
                    horizon=horizon, levels=levels)
    aeclpso = AECLPSO(pop_size=pop_size, max_iter=max_iter)
    population = toolbox.population(n=pop_size)
    velocities = [[random.uniform(-0.1, 0.1) for _ in ind] for ind in population]
    fitnesses = [toolbox.evaluate(ind) for ind in population]
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    pbest = [deepcopy(ind) for ind in population]
    pbest_fitness = [ind.fitness.values[0] for ind in pbest]
    aeclpso._update_elite_archive(population)
    best_individual = None
    best_fitness = float('inf')
    fitness_history = []
    diversity_history = []
    for iteration in range(max_iter):
        print(f"第 {iteration+1}/{max_iter} 代优化")
        current_fitnesses = [ind.fitness.values[0] for ind in population]
        print(f"当前种群适应度统计:")
        print(f"  最优: {min(current_fitnesses):.6f}")
        print(f"  最差: {max(current_fitnesses):.6f}")
        print(f"  平均: {np.mean(current_fitnesses):.6f}")
        print(f"  标准差: {np.std(current_fitnesses):.6f}")
        current_diversity = aeclpso._calculate_diversity(population)
        diversity_history.append(current_diversity)
        if len(diversity_history) > 1:
            diversity_ratio = current_diversity / max(diversity_history[0], 1e-10)
        else:
            diversity_ratio = 1.0
        improvements = 0
        for i in range(pop_size):
            if population[i].fitness.values[0] < pbest_fitness[i]:
                improvements += 1
                if pbest_fitness[i] > 0:
                    aeclpso.improvement_rate[i] = (pbest_fitness[i] - population[i].fitness.values[0]) / pbest_fitness[i]
                else:
                    aeclpso.improvement_rate[i] = 0.5
            else:
                aeclpso.improvement_rate[i] *= 0.9  
        
        success_rate = improvements / pop_size
        aeclpso.convergence_state = aeclpso._determine_convergence_state(
            diversity_ratio, success_rate, iteration
        )
        
        w = aeclpso._adaptive_inertia_weight(iteration, diversity_ratio, success_rate)
        c = clpso._adaptive_learning_coefficients(diversity_ratio, success_rate)
        aeclpso._adaptive_topology(iteration, current_diversity)
        
        print(f"\n算法状态:")
        print(f"  收敛状态: {aeclpso.convergence_state}")
        print(f"  拓扑结构: {aeclpso.topology}")
        print(f"  群体多样性: {current_diversity:.4f}")
        print(f"  多样性比率: {diversity_ratio:.4f}")
        print(f"  成功率: {success_rate:.4f}")
        print(f"  惯性权重: {w:.4f}")
        print(f"  学习系数: {c:.4f}")
        print(f"  精英档案大小: {len(aeclpso.elite_archive)}")
        print(f"  停滞粒子数: {sum(1 for x in aeclpso.stagnation_count if x > aeclpso.stagnation_threshold)}")
        
        for i, (particle, velocity) in enumerate(zip(population, velocities)):
            print(f"\n--- 粒子 {i+1}/{pop_size} ---")
            learning_prob = clpso._adaptive_learning_probability(i, iteration)
            
            if aeclpso.stagnation_count[i] > aeclpso.stagnation_threshold:
                particle[:], velocity[:] = aeclpso._handle_stagnation(i, particle[:], velocity[:])
                print(f"  粒子 {i+1} 触发停滞处理机制")
            for d in range(len(particle)):
                exemplar = aeclpso._select_exemplar(i, d, pbest, learning_prob)
                r = random.random()
                velocity[d] = w * velocity[d] + c * r * (exemplar - particle[d])
                v_max = 0.2 * abs(particle[d]) if particle[d] != 0 else 1.0
                velocity[d] = np.clip(velocity[d], -v_max, v_max)
                particle[d] += velocity[d]
            particle[:] = apply_constraints(particle)
        fitnesses = [toolbox.evaluate(ind) for ind in population]
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        for i, (ind, fit) in enumerate(zip(population, fitnesses)):
            if fit[0] < pbest_fitness[i]:
                pbest[i] = deepcopy(ind)
                pbest_fitness[i] = fit[0]
                aeclpso.stagnation_count[i] = 0
                print(f"  粒子 {i+1} 更新了个体最优: {fit[0]:.6f}")
            else:
                aeclpso.stagnation_count[i] += 1
        aeclpso._update_elite_archive(population)

        current_best_idx = np.argmin(pbest_fitness)
        current_best_fitness = pbest_fitness[current_best_idx]
        
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_individual = deepcopy(pbest[current_best_idx])
            print(f"\n发现新的全局最优: {best_fitness:.6f}")
        
        fitness_history.append(best_fitness)
        print(f"\n迭代 {iteration+1} 总结:")
        print(f"  当前最优适应度: {best_fitness:.6f}")
        print(f"  本代改进粒子数: {improvements}")
        print(f"  平均改进率: {np.mean(clpso.improvement_rate):.4f}")
        
    return best_individual, best_fitness, fitness_history, diversity_history

#main
def main():
    df_train_full = load_and_preprocess_data()
    horizon = 1
    levels = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    pop_size = 10
    max_iter = 50
    print(f"预测天数: {horizon}")
    print(f"种群大小: {pop_size}")
    print(f"最大迭代数: {max_iter}")
    print(f"搜索空间维度: 14")
    print(f"置信水平: {levels}")
    best_params, best_score, fitness_hist, diversity_hist = aeclpso_optimize_enhanced(
        df_train_full, horizon, levels, pop_size=pop_size, max_iter=max_iter
    )
    
    final_model, final_cv_df, final_metrics = train_final_model(
        best_params,df_train_full, horizon, levels
    )
    
    return optimization_results


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    results = main()