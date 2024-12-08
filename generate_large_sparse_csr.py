import numpy as np
import os

def generate_csr_matrix(rows, cols, density=0.001, pattern="random"):
    """
    Generate a sparse matrix in CSR format with given dimensions and density.
    Patterns: 'random', 'banded', 'powerlaw'
    """
    nnz = int(rows * cols * density)  # number of non-zero elements
    row_ptrs = [0]
    col_indices = []
    values = []
    
    if pattern == "random":
        # Distribute non-zeros roughly evenly across rows
        nnz_per_row = max(1, nnz // rows)
        remaining = nnz
        
        for i in range(rows):
            # Adjust nnz_per_row for last rows to match total nnz
            if i == rows - 1:
                row_nnz = remaining
            else:
                row_nnz = min(remaining, np.random.randint(1, nnz_per_row * 2))
            
            # Generate sorted column indices for this row
            row_cols = np.random.choice(cols, size=row_nnz, replace=False)
            row_cols.sort()  # Sort column indices
            
            col_indices.extend(row_cols)
            values.extend(np.random.randint(1, 101, size=row_nnz))
            remaining -= row_nnz
            row_ptrs.append(len(col_indices))
            
            if remaining <= 0:
                # Fill remaining row pointers
                row_ptrs.extend([len(col_indices)] * (rows - len(row_ptrs) + 1))
                break
    
    elif pattern == "banded":
        bandwidth = min(1000, cols // 10)  # Adjust bandwidth based on matrix size
        for i in range(rows):
            band_start = max(0, i - bandwidth//2)
            band_end = min(cols, i + bandwidth//2)
            # Select random columns within the band
            row_nnz = np.random.randint(1, min(20, band_end - band_start))
            row_cols = np.random.choice(range(band_start, band_end), size=row_nnz, replace=False)
            row_cols.sort()
            
            col_indices.extend(row_cols)
            values.extend(np.random.randint(1, 101, size=row_nnz))
            row_ptrs.append(len(col_indices))
    
    return row_ptrs, col_indices, values

def save_csr_matrix(filename, row_ptrs, col_indices, values):
    """Save matrix in CSR format."""
    with open(filename, 'w') as f:
        f.write('\n')  # Empty line at start
        f.write('Row pointers:\n')
        f.write(' '.join(map(str, row_ptrs)) + '\n')
        f.write('Column indices:\n')
        f.write(' '.join(map(str, col_indices)) + '\n')
        f.write('Values:\n')
        f.write(' '.join(map(str, values)) + '\n')

def main():
    # Create different test cases
    test_cases = [
        {
            'name': 'large_random',
            'rows': 100000,
            'cols': 100000,
            'density': 0.0001,  # 0.01% density
            'pattern': 'random'
        },
        {
            'name': 'large_banded',
            'rows': 50000,
            'cols': 50000,
            'density': 0.001,   # 0.1% density
            'pattern': 'banded'
        }
    ]
    
    for case in test_cases:
        print(f"\nGenerating {case['name']} matrices ({case['rows']}x{case['cols']})...")
        output_dir = f"inputs/testinput/{case['name']}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate matrices A and B
        for matrix_name in ['A', 'B']:
            row_ptrs, col_indices, values = generate_csr_matrix(
                case['rows'], case['cols'], case['density'], case['pattern']
            )
            
            filename = os.path.join(output_dir, f'matrix_{matrix_name}_csr.txt')
            save_csr_matrix(filename, row_ptrs, col_indices, values)
            print(f"Matrix {matrix_name} generated with {len(values)} non-zero elements")
            print(f"Actual density: {len(values)/(case['rows']*case['cols'])*100:.4f}%")

if __name__ == '__main__':
    main()
