
# ref: https://github.com/pearu/pearu.github.io/blob/1cdb9eaf60e1313834d7dedb80feb74c1153f3fa/csr_naming_conventions.md
def scipy2torch(src, device='cpu'):
    """Convert scipy.sparse.csr_matrix to torch.sparse_csr_tensor."""
    import torch
    return torch.sparse_csr_tensor(
        torch.from_numpy(src.indptr),
        torch.from_numpy(src.indices),
        torch.from_numpy(src.data),
        device=device,)
