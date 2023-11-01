
def weighted_allocate(target_list, weight, fraction_num):
    duplicated_list = []
    for i in range(len(target_list)):
        duplicated_list += [target_list[i] for _ in range(weight[i])]
    num_per_fraction = len(duplicated_list) // fraction_num
    res = []
    for i in range(fraction_num-1):
        list_to_append = list(set(duplicated_list[i*num_per_fraction: (i+1)*num_per_fraction]))
        list_to_append.sort(key=lambda x:x[0])
        res.append(list_to_append)
    list_to_append = list(set(duplicated_list[(fraction_num-1)*num_per_fraction:]))
    list_to_append.sort(key=lambda x:x[0])
    res.append(list_to_append)
    for i in range(len(res)-1):
        if res[i+1][0] == res[i][-1]:
            del res[i+1][0]
    return res


