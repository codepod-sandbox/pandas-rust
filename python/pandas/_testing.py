"""Testing utilities for pandas-rust."""


def assert_frame_equal(left, right, check_dtype=True):
    assert left.shape == right.shape, "Shape mismatch: {} vs {}".format(left.shape, right.shape)
    assert list(left.columns) == list(right.columns), "Columns mismatch"
    if check_dtype:
        assert left.dtypes == right.dtypes, "Dtypes mismatch"
    left_dict = left.to_dict()
    right_dict = right.to_dict()
    for col in left.columns:
        for i, (lv, rv) in enumerate(zip(left_dict[col], right_dict[col])):
            if isinstance(lv, float) and isinstance(rv, float):
                if lv != lv and rv != rv:  # both NaN
                    continue
                assert abs(lv - rv) < 1e-10, "Value mismatch at column '{}', row {}: {} vs {}".format(col, i, lv, rv)
            else:
                assert lv == rv, "Value mismatch at column '{}', row {}: {} vs {}".format(col, i, lv, rv)


def assert_series_equal(left, right, check_dtype=True, check_names=True):
    assert len(left) == len(right), "Length mismatch: {} vs {}".format(len(left), len(right))
    if check_names:
        assert left.name == right.name, "Name mismatch: {} vs {}".format(left.name, right.name)
    if check_dtype:
        assert left.dtype == right.dtype, "Dtype mismatch: {} vs {}".format(left.dtype, right.dtype)
    lvals = left.tolist()
    rvals = right.tolist()
    for i, (lv, rv) in enumerate(zip(lvals, rvals)):
        if isinstance(lv, float) and isinstance(rv, float):
            if lv != lv and rv != rv:
                continue
            assert abs(lv - rv) < 1e-10, "Value mismatch at index {}: {} vs {}".format(i, lv, rv)
        else:
            assert lv == rv, "Value mismatch at index {}: {} vs {}".format(i, lv, rv)
