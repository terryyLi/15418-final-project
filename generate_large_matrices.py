import numpy as np
import os

def generate_sparse_matrix(rows, cols, density=0.1):
    """Generate a sparse matrix with given dimensions and density."""
    # Calculate number of non-zero elements
    num_elements = int(rows * cols * density)
    
    # Generate random positions for non-zero elements
    positions = np.random.choice(rows * cols, size=num_elements, replace=False)
    row_indices = positions // cols
    col_indices = positions % cols
    
    # Generate random values (between 1 and 100 to avoid 0s)
    values = np.random.randint(1, 101, size=num_elements)
    
    # Create the matrix
    matrix = np.zeros((rows, cols))
    matrix[row_indices, col_indices] = values
    
    return matrix

def save_matrix(matrix, filename):
    """Save matrix in the required format."""
    rows, cols = matrix.shape
    with open(filename, 'w') as f:
        f.write('\n')  # Empty line at start
        f.write(f'{rows} {cols}\n')
        for row in matrix:
            f.write(' '.join(map(str, map(int, row))) + '\n')

def main():
    # Parameters
    size = 10000  # Size of the matrices (10000 x 10000)
    density_A = 0.1  # 1% non-zero elements for matrix A
    density_B = 0.1  # 1% non-zero elements for matrix B
    
    # Create output directory if it doesn't exist
    output_dir = 'inputs/testinput/large1'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate matrices
    print(f"Generating {size}x{size} matrices with {density_A*100}% density...")
    matrix_A = generate_sparse_matrix(size, size, density_A)
    matrix_B = generate_sparse_matrix(size, size, density_B)
    
    # Save matrices
    print("Saving matrices...")
    save_matrix(matrix_A, os.path.join(output_dir, 'matrix_A.txt'))
    save_matrix(matrix_B, os.path.join(output_dir, 'matrix_B.txt'))
    print(f"Matrices saved in {output_dir}")
    print(f"Matrix A non-zero elements: {np.count_nonzero(matrix_A)}")
    print(f"Matrix B non-zero elements: {np.count_nonzero(matrix_B)}")

if __name__ == '__main__':
    main()
