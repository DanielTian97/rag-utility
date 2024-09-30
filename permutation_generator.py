import itertools

# def get_combinations(lst): # creating a user-defined method
#    combination = [] # empty list 
#    for r in range(1, len(lst) + 1):
#       # to generate combination
#       combination.extend(itertools.combinations(lst, r))
#    return combination

def construct_p_name(l):
    name = ''
    for i in l:
        name += f'_{i}'
    return name

def simple_shuffling(l: list):
    shuffle_result = []
    l.reverse()
    
    dida = True
    for i in l:
        if(dida):
            shuffle_result.append(i)
        else:
            shuffle_result = [i] + shuffle_result
        dida = not(dida)
    return shuffle_result

def get_permutation(lst, p_len, full_permutations=True): # creating a user-defined method

    index = range(len(lst))
    if(p_len > len(lst)):
        print("Permutation length should be less than or equal to the length of the list.")
        return []
    
    print(full_permutations)
    if(full_permutations):
        permutations = list(itertools.permutations(index, p_len))
    else:
        # then p_len should = len(lst)
        
        pmt_0 = tuple(list(range(p_len)))# original order
        pmt_r = tuple(list(range(p_len-1, -1, -1)))# reverse
        pmt_lim = tuple(simple_shuffling(list(range(p_len)))) # lost in the middle
        pmt_r_lim = tuple(simple_shuffling(list(range(p_len-1, -1, -1))))# reversely _ lost in the middle

        permutations = [pmt_0, pmt_r, pmt_lim, pmt_r_lim]
    
    permutation_book = {}
    for p in permutations:
        p_name = construct_p_name(p)
        permutation_book.update({p_name:[lst[i] for i in list(p)]})
    return permutation_book

def get_permutation_simple(lst): # creating a user-defined method
    permutations = list(itertools.permutations(lst))
    return permutations

if __name__=="__main__":
    print("\nPERMUTATION TEST:")
    l1 = [3, 2]
    print(f'List = {l1}')
    all_combinations_1 = get_permutation(l1, len(l1)) # method call
    print(all_combinations_1)
    
    print("\nPERMUTATION TEST (4 SELECTED ONES):")
    l1 = [3, 2, 7, 12, 9]
    print(f'List = {l1}')
    all_combinations_1 = get_permutation(l1, len(l1), FULL_PERMUTATIONS=False) # method call
    print(all_combinations_1)