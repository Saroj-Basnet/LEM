from landlab.components import FlowAccumulator
from IPython.display import clear_output
from landlab import RasterModelGrid
import matplotlib.pylab as plt
from scipy import optimize
from mpmath import *
import numpy as np
import scipy
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import Affine
import os



output_folder="path\\to\\output\\"
os.makedirs(output_folder, exist_ok=True)

def DinfEroder(mg, dt, K_sp, m_sp, n_sp):
    nodes = np.arange(mg.number_of_nodes)
    receiver = mg.at_node['flow__receiver_node']
    A = np.copy(mg.at_node['drainage_area'])
    proportion = mg.at_node['flow__receiver_proportions']
    ele = np.copy(mg.at_node['topographic__elevation'])
    neighbor = np.append(mg.diagonal_adjacent_nodes_at_node, mg.adjacent_nodes_at_node, axis = 1)
    donor = np.zeros_like(neighbor)
    
    receiver_of_neighbor = receiver[neighbor]
    receiver_of_neighbor[:,:,0] = np.where(neighbor == -1 , -1, receiver_of_neighbor[:,:,0])
    receiver_of_neighbor[:,:,1] = np.where(neighbor == -1 , -1, receiver_of_neighbor[:,:,1])
    
    for nei in range(8):
        donor[:, nei] = np.where((receiver_of_neighbor[:, nei, 0] == nodes) |
                                 (receiver_of_neighbor[:, nei, 1] == nodes) , neighbor[:, nei], -1)

    queue = list(np.arange(mg.number_of_nodes)[mg.at_node['flow__sink_flag'] == 1])
    
    processed_nodes = np.zeros_like(nodes, dtype=bool)
    processed_nodes[queue] = True
    
    while len(queue) > 0:
        
        node = queue.pop(0)
        
        for nei in donor[node, donor[node] >= 0]:
            
            if processed_nodes[nei] == False and np.all(processed_nodes[receiver[nei][receiver[nei] >= 0]]):
                processed_nodes[nei] = True
                
                h_0 , h_1 = None, None
                for res, pro in zip(receiver[nei], proportion[nei]):
                    if res != -1:
                        if res // ncols == nei // ncols or res % ncols == nei % ncols:
                            h_0 = ele[res]
                        else:
                            h_1 = ele[res]
                h = ele[nei]
                width = dx
                if h_0 is not None and h_1 is not None:
                    f = lambda h_next: h_next - h + K_sp * (A[nei] / width)**m_sp *\
                                ((h_0 - h_1)**2 + (h_next - h_0)**2)**(n_sp / 2.) * dt / dx**n_sp
                    min_h = min(h_0, h_1)

                elif h_0 is not None:
                    f = lambda h_next: h_next - h + K_sp * (A[nei] / width)**m_sp *\
                                ((h_next - h_0)**2)**(n_sp / 2.) * dt / dx**n_sp
                    min_h = h_0
                elif h_1 is not None:
                    f = lambda h_next: h_next - h + K_sp * (A[nei] / width)**m_sp *\
                                (h_next - h_1)**n_sp * dt / (2**.5 * dx)**n_sp

                    min_h = h_1
                    
                ele[nei] = optimize.brenth(f, h, min_h)
                
                queue.append(nei)
    return ele

def implicit_diffusion(zgiven, dt, D):    
    kd   =                       D*dt/(dx**2)
    d1m1 = np.repeat(-kd, len(mg.core_nodes))
    d1m1[ncols-3:d1m1.size:ncols-2]       = 0
    A    = scipy.sparse.diags([-kd, d1m1, 1 + 4*kd, d1m1, -kd], [-ncols+2, -1, 0, 1, ncols-2], shape=(len(mg.core_nodes), len(mg.core_nodes)))
    return scipy.sparse.linalg.lgmres(A, zgiven, atol=0.000000000001)[0]

def write_tfw(filename, transform):
    with open(filename, 'w') as file:
        file.write(f"{transform.a}\n")
        file.write(f"{transform.b}\n")
        file.write(f"{transform.d}\n")
        file.write(f"{transform.e}\n")
        file.write(f"{transform.c}\n")
        file.write(f"{transform.f}\n")


K_sp = 0.00023 
D = 0.003523 
U  = 0.0001
m_sp = 0.3
n_sp = 1.0
dx = 5
nrows = 300
ncols = 1000

CI = (K_sp*((nrows)*dx)**(m_sp + n_sp))/(D**n_sp * U**(1-n_sp))
Pe1 = CI*(((nrows)*dx))**m_sp
Pe2 = ((K_sp*(((nrows)*dx)/2)**(2*m_sp + 1)))/D
print("CI",round(CI))
print("Pe",round(Pe1))
print("Pe",round(Pe2))

np.random.seed(seed = 701)
z = np.random.rand(nrows * ncols)

mg    =                           RasterModelGrid((nrows,ncols), dx)
_     =               mg.add_zeros('node', 'topographic__elevation')

for edge in (mg.nodes_at_left_edge, mg.nodes_at_right_edge):
    mg.status_at_node[edge] = RasterModelGrid.BC_NODE_IS_CLOSED
for edge in (mg.nodes_at_bottom_edge,mg.nodes_at_top_edge):
    mg.status_at_node[edge] = RasterModelGrid.BC_NODE_IS_FIXED_VALUE

mg.at_node['topographic__elevation']   =     z.reshape(nrows * ncols)
fc = FlowAccumulator(mg, flow_director =          'FlowDirectorDINF')
fc.run_one_step()

count = 0
count_max = 500
min_try = 500
i = -1
t = 0
diff_list = []
simulation_time = 10 * 10 ** 6
dt = 370.00
save_interval = 370.00
next_save_time = 370.00
output_counter = 0

while t <= simulation_time:
    D = D 
    K_sp = K_sp 
    i += 1

    dt = 370.00

    ele_1 = np.copy(mg.at_node['topographic__elevation'])
    erode_done = False

    while erode_done is False:
        try:
            fc = FlowAccumulator(mg, flow_director='FlowDirectorDINF')
            fc.run_one_step()

            mg.at_node['topographic__elevation'] = DinfEroder(mg, dt, K_sp, m_sp, n_sp)

            mg.at_node['topographic__elevation'][mg.core_nodes] = implicit_diffusion(
                mg.at_node['topographic__elevation'][mg.core_nodes] + U * dt, dt, D)
            erode_done = True

        except ValueError:
            mg.at_node['topographic__elevation'] = ele_1
            dt = dt / 2.

    ele_2 = np.copy(mg.at_node['topographic__elevation'])
    t += dt
    ele_diff = np.abs(ele_1 - ele_2).max()
    max_diff_location = np.abs(ele_1 - ele_2).argmax()
    ele_diff_mean = np.abs(ele_1.mean() - ele_2.mean())
    diff_list.append([t, ele_diff, ele_diff_mean, max_diff_location])

    if ele_diff_mean < 0.00005 * U * dt:
        count = count + 1
    else:
        count = 0

    if i >= min_try and ele_diff < U * 0.001 * dt and ele_diff_mean < 0.0001 * U * dt:
        count = count_max

    mean_elevation = mg.at_node['topographic__elevation'].mean()

    if t > simulation_time:
        break
       

    if t >= next_save_time:
        elevation = mg.at_node['topographic__elevation'].reshape(nrows, ncols)

        keep_cols = 1000
        col_start = (ncols - keep_cols) // 2      
        col_end   = col_start + keep_cols        

        dem = elevation[:, col_start:col_end]     

        transform = Affine(dx, 0, 0, 0, -dx, 0) * Affine.translation(col_start, 0)

        profile = {
            'driver': 'GTiff',
            'dtype': rasterio.float32,
            'count': 1,
            'height': nrows,
            'width': dem.shape[1],
            'transform': transform
        }

        filename = os.path.join(output_folder, f'DEM_{output_counter}.tif')

        with rasterio.open(filename, 'w', **profile) as dst:
            dst.write(dem.astype(rasterio.float32), 1)

        write_tfw(filename.replace('.tif', '.tfw'), transform)
        
        output_counter += 1
        
        if output_counter % 100 == 0:
            print(f"Time: {t:.2f} years | Mean Elevation: {mean_elevation:.2f} m | Files saved: {output_counter}")
        
        next_save_time += save_interval