from adeft_indra.mine import MiningOperation
op = MiningOperation('/Users/albertsteppi/msc/mine/',
                     'sample_op',
                     list(range(100000, 105000)),
                     batch_size=1000, cache_size=300)

def large_scale_op():
    op.mine()
