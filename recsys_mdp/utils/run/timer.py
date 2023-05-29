import timeit

timer = timeit.default_timer


def print_with_timestamp(start_time: float, *args):
    elapsed_sec = timer() - start_time
    if elapsed_sec < 1:
        time_format = '5.3f'
    elif elapsed_sec < 10:
        time_format = '5.2f'
    elif elapsed_sec < 1000:
        time_format = '5.1f'
    else:
        time_format = '5.0f'
    print(f'[{elapsed_sec:{time_format}}]', *args)
