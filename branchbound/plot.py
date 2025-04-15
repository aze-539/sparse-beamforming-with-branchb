import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize

# Original data lists
bb_data = [5.04372503, 8.11583042, 10.3843361, 11.30463792, 12.2385116, 13.11703581, 13.21761111, 13.59284322]
greedy_data = [3.62608037, 6.01481573, 7.37716502, 8.59212492, 9.79184836, 11.20251035, 12.22151576, 13.59284322]
cvx_data = [4.35381451, 5.67187698, 7.25600819, 9.58369699, 10.4781935, 11.48460819, 12.99418134, 13.59284322]
W_data = np.array([[ 0.        +0.j        ,  0.        +0.j        ,
         0.        +0.j        ,  0.        +0.j        ],
       [ 0.        +0.j        ,  0.        +0.j        ,
         0.        +0.j        ,  0.        +0.j        ],
       [-0.59856937+0.j        , -1.51434815+0.j        ,
        -0.67864942+0.j        ,  1.30651164+0.j        ],
       [ 0.92077895-0.02999437j,  0.26168692+0.24742969j,
         0.29079463-1.78868125j,  0.67557066-0.53284071j],
       [ 0.        +0.j        ,  0.        +0.j        ,
         0.        +0.j        ,  0.        +0.j        ],
       [-1.30910836-1.12596181j, -0.09326529-0.39588613j,
         1.08746311-0.23962089j, -0.08828442-0.8456231j ],
       [ 1.04496618-0.34502068j, -1.65856331-0.07338345j,
         0.3962742 -0.0350218j , -0.95013233-0.200415j  ],
       [ 0.        +0.j        ,  0.        +0.j        ,
         0.        +0.j        ,  0.        +0.j        ]])
W_full_data = np.array([[ 0.88262044+0.j        , -0.55177583+0.j        ,
         0.84538629+0.j        ,  0.40217603+0.j        ],
       [ 0.07043162+0.41949126j,  0.3709228 +0.53936468j,
        -0.9916785 +0.25252312j, -0.35233948+0.22859819j],
       [ 0.4816652 -0.42207329j,  1.49714577+0.24309455j,
         0.24553039+0.51109461j,  0.04553639+0.05365134j],
       [-0.37328178-0.87985915j, -0.65537659+0.29954006j,
         0.16277065+1.35037447j, -0.19465515-1.19557265j],
       [ 0.33940455+0.48867843j,  0.20045528+0.95807765j,
         0.60616092+0.01444468j,  0.86906783-0.26449051j],
       [ 0.62369718+1.27124394j,  0.03038506-0.4713865j ,
         0.31674927-0.01525458j, -0.23516967-1.03435019j],
       [-0.35566892-0.30539839j,  0.45706418+0.37847019j,
        -0.51041026-0.41641717j,  0.15889273-0.5338712j ],
       [ 0.47413185+0.3512329j , -0.21712648+0.17134477j,
        -0.47566464+0.2561699j , -0.63051047-0.17551588j]])
greedy_W = np.array([[ 0.        +0.j        ,  0.        +0.j        ,
         0.        +0.j        ,  0.        +0.j        ],
       [ 0.        +0.j        ,  0.        +0.j        ,
         0.        +0.j        ,  0.        +0.j        ],
       [ 0.        +0.j        ,  0.        +0.j        ,
         0.        +0.j        ,  0.        +0.j        ],
       [ 0.        +0.j        ,  0.        +0.j        ,
         0.        +0.j        ,  0.        +0.j        ],
       [-1.54178761+0.j        , -1.00990432+0.j        ,
        -1.40937499+0.j        ,  0.28830711+0.j        ],
       [ 0.29009088-0.6666623j ,  0.80261659+0.51975367j,
        -1.09947732+0.17004451j, -1.21495695-0.32927649j],
       [ 0.45754976+0.28692636j,  0.44834988-1.82270525j,
        -0.55963294+0.57173905j,  0.31263182-0.78102003j],
       [-0.8801073 +1.45946201j,  0.20197723-0.64940924j,
         0.36815692-0.90760553j, -0.87912722-0.07895761j]])
cvx_W = np.array([[8.72665919e-01+0.00000000e+00j, -1.10992986e+00+0.00000000e+00j,
  -1.34782659e+00+0.00000000e+00j,  1.02556261e+00+0.00000000e+00j],
 [1.04425677e-01-6.47230360e-02j, 3.23676571e-02+1.39909193e-01j,
  -5.60117770e-02+2.66334390e-01j, -2.62302216e-01-1.23923433e-01j],
 [0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,
   0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j],
 [-2.66675637e-01+3.93520676e-01j,  2.78787618e-01+4.80990606e-01j,
  -1.53379579e+00+4.59625273e-02j, -1.33215371e+00+2.78139622e-01j],
 [8.74572606e-01-1.01574901e+00j, -1.20124844e+00-8.38933924e-02j,
   5.33504949e-01-8.57424912e-02j, -1.04438242e+00+5.65991847e-01j],
 [0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,
  -1.12035566e-08-1.95968192e-08j, -3.09445649e-09+3.12908580e-08j],
 [0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,
   0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j],
 [-1.51197284e+00-4.84659645e-01j, -8.03161429e-01-1.25932807e+00j,
  -3.39144075e-01+6.07193323e-01j, -6.07459000e-02-1.15829691e-01j]])


def find_sorted_lists_with_indices(nested_list):
    """
    Find sorted sublists and their indices from a nested list
    """
    sorted_lists_with_indices = [(i, lst) for i, lst in enumerate(nested_list) if lst == sorted(lst)]
    return sorted_lists_with_indices


def plot_magnitude_heatmap(matrix1, matrix2, matrix3, matrix4, tol=1e-5):
    """
    plot and save magnitude heatmaps for four matrices with masking
    Args:
        matrix1-4 (ndarray): Input matrices to visualize
        tol (float): Tolerance value for masking small elements
    """
    # Mask elements smaller than tolerance
    matrix1_masked = np.ma.masked_where(np.abs(matrix1) < tol, matrix1)
    matrix2_masked = np.ma.masked_where(np.abs(matrix2) < tol, matrix2)
    matrix3_masked = np.ma.masked_where(np.abs(matrix3) < tol, matrix3)
    matrix4_masked = np.ma.masked_where(np.abs(matrix4) < tol, matrix4)

    # Create custom colormap with white for masked values
    cmap = plt.cm.coolwarm
    colors = ['white'] + [cmap(i) for i in np.linspace(0, 1, cmap.N)]
    custom_cmap = ListedColormap(colors)
    custom_cmap.set_bad(color='white')  # Explicitly set masked color to white

    # Create normalization objects for color mapping
    norm1 = Normalize(vmin=-np.max(np.abs(matrix1)), vmax=np.max(np.abs(matrix1)), clip=False)
    norm2 = Normalize(vmin=-np.max(np.abs(matrix2)), vmax=np.max(np.abs(matrix2)), clip=False)
    norm3 = Normalize(vmin=-np.max(np.abs(matrix3)), vmax=np.max(np.abs(matrix3)), clip=False)

    # Plot and save first heatmap
    fig, ax = plt.subplots(figsize=(6, 6))
    im1 = ax.imshow(np.abs(matrix1_masked), cmap=custom_cmap, norm=norm1, aspect='auto')
    im1.set_array(np.ma.masked_where(np.abs(matrix1_masked) < tol, np.abs(matrix1_masked)))
    plt.colorbar(im1, ax=ax)
    plt.savefig('experiment results/bb_halfNt.png', dpi=100, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    # Plot and save second heatmap
    fig, ax = plt.subplots(figsize=(6, 6))
    im2 = ax.imshow(np.abs(matrix2_masked), cmap=custom_cmap, norm=norm2, aspect='auto')
    im2.set_array(np.ma.masked_where(np.abs(matrix2_masked) < tol, np.abs(matrix2_masked)))
    plt.colorbar(im2, ax=ax)
    plt.savefig('experiment results/greedy_halfNt.png', dpi=100, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    # Plot and save third heatmap
    fig, ax = plt.subplots(figsize=(6, 6))
    im3 = ax.imshow(np.abs(matrix3_masked), cmap=custom_cmap, norm=norm3, aspect='auto')
    im3.set_array(np.ma.masked_where(np.abs(matrix3_masked) < tol, np.abs(matrix3_masked)))
    plt.colorbar(im3, ax=ax)
    plt.savefig('experiment results/cvx_halfNt.png', dpi=100, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    # Plot and save fourth heatmap
    fig, ax = plt.subplots(figsize=(6, 6))
    im4 = ax.imshow(np.abs(matrix4_masked), cmap=custom_cmap, norm=norm3, aspect='auto')
    im4.set_array(np.ma.masked_where(np.abs(matrix4_masked) < tol, np.abs(matrix4_masked)))
    plt.colorbar(im4, ax=ax)
    plt.savefig('experiment results/Nt.png', dpi=100, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


plot_magnitude_heatmap(W_data, greedy_W, cvx_W, W_full_data)
new_bb_data = [-x for x in bb_data]
new_greedy_data = [-x for x in greedy_data]
new_cvx_data = [-x for x in cvx_data]
# Draw the curve plot
plt.plot(range(1, 9), new_bb_data, label='B&B', color='#d62728', linewidth=2, marker='o')  # 红色，圆形标记
plt.plot(range(1, 9), new_greedy_data, label='Greedy', color='#ff7f0e', linewidth=2, marker='s')  # 橙色，方形标记
plt.plot(range(1, 9), new_cvx_data, label='Lccvxr', color='#1f77b4', linewidth=2, marker='^')  # 蓝色，三角形标记

# Set the legend and adjust the font size
plt.legend(loc='best', fontsize=12)  # 增大字体

# Set labels and titles
plt.xlabel('L', fontsize=12)
plt.ylabel('Objective value', fontsize=12)
plt.grid(True)

# Save the figure to file
plt.savefig('experiment results/curve_plot.png', dpi=100, bbox_inches='tight')

# Display the figure
plt.show()







