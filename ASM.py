import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
To know more details about the ASM method, refer to:

[1] Treiber, M., & Helbing, D. (2003). An adaptive smoothing method for traffic state 
identification from incomplete information. In Interface and transport dynamics: 
Computational Modelling (pp. 343-360). Berlin, Heidelberg: Springer Berlin Heidelberg.

"""

def add_bounded_edges(matrix, boundary_value, row_boundary_thickness, col_boundary_thickness):
    original_rows, original_cols = matrix.shape
    new_rows = original_rows + 2 * row_boundary_thickness
    new_cols = original_cols + 2 * col_boundary_thickness

    # Create a new matrix filled with the boundary value
    new_matrix = np.full((new_rows, new_cols), boundary_value)

    # Insert the original matrix into the center of the new matrix
    new_matrix[row_boundary_thickness:row_boundary_thickness + original_rows,
               col_boundary_thickness:col_boundary_thickness + original_cols] = matrix

    return new_matrix

def generate_weight_matrices(delta=0.12, dx=0.02, dt=4, c_cong=13, c_free=-45, tau=20):
    t = abs(delta / c_cong / 2)
    x_mat = 2*int(delta/dx/2) + 1
    t_mat = int(t / dt * 3600) * 2 + 1
    matrix = np.zeros([x_mat, t_mat])
    matrix_df = pd.DataFrame(matrix)
    st_df = matrix_df.stack().reset_index()
    st_df.columns = ['x', 't', 'weight']
    st_df['time'] = dt * (st_df['t'] - int(t_mat / 2))
    st_df['space'] = dx * (st_df['x'] - int(x_mat / 2))

    def fill_cong_weight(row):
        t_new = row['time'] - row['space'] / (c_cong / 3600)
        if abs(t_new) < tau / 2:
            return np.exp(-(abs(t_new) / tau + abs(row['space']) / delta))
        else:
            return 0

    def fill_free_weight(row):
        t_new = row['time'] - row['space'] / (c_free / 3600)
        if abs(t_new) < tau / 2:
            return np.exp(-(abs(t_new) / tau + abs(row['space']) / delta))
        else:
            return 0

    st_df['cong_weight'] = st_df.apply(fill_cong_weight, axis=1)
    st_df['free_weight'] = st_df.apply(fill_free_weight, axis=1)

    cong_weight_matrix = st_df.pivot(index='t', columns='x', values='cong_weight').values
    free_weight_matrix = st_df.pivot(index='t', columns='x', values='free_weight').values

    return cong_weight_matrix, free_weight_matrix

def smooth_speed_field(raw_data, dx=0.02, dt=4, c_cong=12.5, c_free=-45, tau=20, delta=0.12):
    cong_weight_matrix, free_weight_matrix = generate_weight_matrices( delta=delta,dx=dx, dt=dt, c_cong=c_cong, c_free=c_free, tau=tau)
    half_x_mat = int((cong_weight_matrix.shape[1] - 1) / 2)
    half_t_mat = int((cong_weight_matrix.shape[0] - 1) / 2)
    smooth_data = np.zeros(raw_data.shape)
    raw_data_w_bound = add_bounded_edges(raw_data, np.nan, half_t_mat, half_x_mat)
    for time_idx in tqdm(range(raw_data.shape[0])):
        for space_idx in range(raw_data.shape[1]):
            neighbour_matrix = raw_data_w_bound[time_idx:time_idx + 2 * half_t_mat + 1,
                               space_idx:space_idx + 2 * half_x_mat + 1]
            mask = pd.DataFrame(neighbour_matrix).notna().astype(int).values
            neighbour_fillna = pd.DataFrame(neighbour_matrix).fillna(0).values
            N_cong = np.sum(np.multiply(mask, cong_weight_matrix))
            N_free = np.sum(np.multiply(mask, free_weight_matrix))
            if N_cong == 0:
                v_cong = np.nan
            else:
                v_cong = np.sum(np.multiply(neighbour_fillna, cong_weight_matrix)) / N_cong
            if N_free == 0:
                v_free = np.nan
            else:
                v_free = np.sum(np.multiply(neighbour_fillna, free_weight_matrix)) / N_free
            if N_cong != 0 and N_free != 0:
                w = 0.5 * (1 + np.tanh((37.29 - min(v_cong, v_free)) / 12.43))
                v = w * v_cong + (1 - w) * v_free
            elif N_cong == 0:
                v = v_free
            elif N_free == 0:
                v = v_cong
            elif N_cong == 0 and N_free == 0:
                v = np.nan
            smooth_data[time_idx][space_idx] = v
    return smooth_data
