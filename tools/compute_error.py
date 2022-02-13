import math


def compute_image_error_rate(error_num, classes_num):
    return 1.0 * error_num / classes_num


def compute_bit_error_num(str):
    my_list = eval(str)
    error_sum = 0
    for item in my_list:
        label = int(item["label"])
        pred = int(item["pred"])
        error_num = bin(label^pred).count("1")
        error_sum += error_num
    return error_sum


def compute_bit_error_rate(bit_error_sum, classes_num):
    return bit_error_sum / (1.0 * classes_num * int(math.log(classes_num, 2)))


if __name__ =="__main__":
    error_sum = compute_bit_error_num(
        "[{'label': 650, 'pred': 590}, "
        "{'label': 387, 'pred': 722}, "
        "{'label': 70, 'pred': 69}, "
        "{'label': 162, 'pred': 371},"
        "{'label': 133, 'pred': 581}, "
        "{'label': 519, 'pred': 712}]")
    print("error_rate:"+str(error_sum/(768*8)))  # 0.00390625