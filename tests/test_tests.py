import ndistest.test

def test_min_int_gt_prime():

    f1 = lambda a: 0.1*a - 1
    f2 = lambda a: 0.1*a 
    f3 = lambda a: 0.001*a - 1
    
    b1 = ndistest.test.min_int_gt_prime(f1, thresh=0., x0=2, x_max=100, 
               forbiddenmultiples=[12])
    b2 = ndistest.test.min_int_gt_prime(f2, thresh=0., x0=2, x_max=100, 
               forbiddenmultiples=[12])
    b3 = ndistest.test.min_int_gt_prime(f3, thresh=0., x0=2, x_max=100, 
               forbiddenmultiples=[12])
    assert(b1==11), 'should return value 11, instead returned {}'.format(b1)
    assert(b2==1), 'should return min exceeded value 1, instead returned {}'.format(b2)
    assert(b3==97), 'should return max exceeded value 97, instead returned {}'.format(b3)