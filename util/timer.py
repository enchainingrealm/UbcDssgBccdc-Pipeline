from datetime import datetime


def timer(module_factory, df, sizes):
    """
    Measures the trend in the training time of the given classification module
    on the given data, as the size of the training set is increased.
    :param module_factory: a 0-argument lambda returning a new instance of the
    classification module to benchmark
    :param df: the DataFrame containing the training data
    - required columns: columns required by module_factory().retrain
    :param sizes: the sample sizes to measure the training runtime of
    :return: a List of runtimes; the ith runtime corresponds to training on
    sizes[i] rows
    """
    times = []

    for size in sizes:
        print(f"Started evaluating size {size}")

        sampled_df = df.sample(n=size, replace=True)

        time = _timer_helper(module_factory, sampled_df)
        times.append(time)

        print(f"Finished evaluating size {size}")

    return times


def _timer_helper(module_factory, df):
    """
    Returns the runtime of training the given classification module on the
    given data.
    :param module_factory: a 0-argument lambda returning a new instance of the
    classification module to benchmark
    :param df: the DataFrame containing the training data
    - required columns: columns required by module_factory().retrain
    :return: the training time
    """
    start_time = datetime.now()

    module = module_factory()
    module.retrain(df)

    return (datetime.now() - start_time).total_seconds()
