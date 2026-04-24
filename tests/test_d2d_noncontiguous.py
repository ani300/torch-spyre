import torch
import torch._dynamo
import torch_spyre  # noqa: F401


def check(name, cpu_expected, spyre_result):
    actual = spyre_result.to("cpu")
    if torch.allclose(cpu_expected.float(), actual.float(), atol=5e-3, rtol=5e-3):
        print(f"  PASS: {name}")
        return True
    else:
        print(f"  FAIL: {name}")
        diff = (cpu_expected.float() - actual.float()).abs().max().item()
        print(f"    max abs diff: {diff}")
        print(f"    expected[:3,:3]:\n{cpu_expected[:3,:3]}")
        print(f"    actual[:3,:3]:\n{actual[:3,:3]}")
        return False


print("=== Test d2d copy_ with non-contiguous tensors ===\n")

# Test 0: Sanity - contiguous src -> contiguous dst
print("0. Contiguous source -> contiguous dest (sanity)")
cpu_a = torch.randn(128, 64, dtype=torch.float16)
a = cpu_a.to("spyre")
b = torch.empty(128, 64, dtype=torch.float16, device="spyre")
b.copy_(a)
check("contiguous d2d", cpu_a, b)

# Test 1: Transpose source -> contiguous dest
print("1. Transpose source -> contiguous dest")
cpu_a = torch.randn(64, 128, dtype=torch.float16)
a = cpu_a.to("spyre")
b = torch.empty(128, 64, dtype=torch.float16, device="spyre")
b.copy_(a.t())
check("transpose src", cpu_a.t(), b)

# Test 2: Contiguous source -> transpose dest
print("2. Contiguous source -> transpose dest")
cpu_a = torch.randn(64, 128, dtype=torch.float16)
a = cpu_a.to("spyre")
dst_backing = torch.empty(128, 64, dtype=torch.float16, device="spyre")
dst_view = dst_backing.t()  # shape (64, 128), non-contiguous
dst_view.copy_(a)
check("contiguous src -> transpose dst", cpu_a, dst_backing.t())

print("\nDone.")
