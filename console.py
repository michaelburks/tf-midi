import sys

def progress(count, total, description):
    bar_len = 50
    filled_len = bar_len * count / total

    percent = 100 * count / total
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s  %s\r' % (bar, percent, '%', description))
    sys.stdout.flush()
