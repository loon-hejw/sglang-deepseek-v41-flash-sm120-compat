"""Run inside the built image on an isolated GPU; never executed by CPU gates."""
import pytest
import torch

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires an isolated CUDA test GPU')


@pytest.mark.parametrize('batch',[1,3,8,16])
@pytest.mark.parametrize('nan_row',[False,True])
def test_markov_combine_native_and_cuda_graph(batch,nan_row):
    from sglang.kernels.ops.speculative.dspark import dspark_draft_model as module
    # Use the actual installed Triton kernel, including compiled graph replay.
    name=next(n for n in vars(module) if n.endswith('combine_kernel'))
    kernel=getattr(module,name)
    values=torch.tensor([[1.,3.]],device='cuda').repeat(batch,1)
    indices=torch.tensor([[4,2]],device='cuda',dtype=torch.int32).repeat(batch,1)
    if nan_row:values[0].fill_(float('nan'));indices[0].fill_(2147483647)
    output=torch.empty(batch,device='cuda',dtype=torch.int64)
    def run():kernel[(batch,)](values,indices,output,2,BLOCK_TILES=2)
    # Warm up on a side stream before capture.
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):run()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):run()
    for _ in range(3):graph.replay()
    expected=[2]*batch
    if nan_row:expected[0]=0
    assert output.tolist()==expected
